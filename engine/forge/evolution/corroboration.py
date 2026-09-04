"""A second, narrow pass over assessments that asserted a relationship.

**Why this exists rather than another prompt revision.** Measured 2026-09-04
on `openai/gpt-oss-120b`, three of five held-out failures were the model
returning SUPPORTS where the passage never reported the outcome the claim
asserts. In two of those the passage contained the sentence that should have
blocked the inference — one stated the adoption being measured was voluntary
and self-selected, the other defined the claim's key term differently — and
the model asserted support anyway.

Two prompt revisions did not move it. `assess-prompts/0.2.0` added an explicit
cue and a dedicated rule naming that exact failure, and the case still failed.
On the held-out set, cases the cues describe scored no better than cases they
do not (3/5 each), which is what it looks like when instruction is not the
binding constraint. See `docs/research/assessment-quality.md`.

So this does not tell the model more. It asks it less: one question, about one
claim, with the four other classifications and the whole vocabulary of the
main prompt out of view.

**What it can and cannot do.** It only ever *demotes* — SUPPORTS or REFINES to
INSUFFICIENT_EVIDENCE — so it cannot manufacture a relationship, only decline
one. IRRELEVANT and POTENTIAL_CONFLICT are never checked: the failure being
targeted is over-assertion, and a check that could promote would be a second
place for the same error to enter.

The `quote` requirement is the part doing real work. A model that answers true
must reproduce the sentence, and the sentence is checked against the evidence
it was shown. A boolean is cheap to get wrong; pointing at text that is not
there is a different and rarer failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..domain import AssessmentClass, AssessmentRecord, Claim, Span
from ..ids import text_hash
from ..ingestion.derivation import DerivationKey
from ..llm.base import (
    CompletionRequest,
    LLMError,
    LLMProvider,
    Message,
    ProviderUnavailable,
    StructuredOutputError,
)
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore
from .prompts import (
    CORROBORATION_INSTRUCTION,
    CORROBORATION_PROMPT_VERSION,
    CORROBORATION_SYSTEM,
)
from .schemas import SCHEMA_VERSION, CorroborationCheck

log = get_logger(__name__)

CORROBORATOR_VERSION = "corroborator/0.1.0"

#: The classifications this pass examines. Only assertions are checked,
#: because only assertions can add a wrong belief.
ASSERTIVE = (AssessmentClass.SUPPORTS, AssessmentClass.REFINES)

#: Below this, a `quote` is not treated as found in the evidence. Models
#: normalise whitespace and clip trailing punctuation, so an exact substring
#: test rejects quotes that are genuinely present.
QUOTE_MATCH_FLOOR = 0.80


@dataclass
class CorroborationOutcome:
    """What the pass did, for logging and for the evaluation harness."""

    checked: int = 0
    demoted: int = 0
    upheld: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    #: Claim ids demoted, with why. Surfaced so a reviewer can see what the
    #: check removed rather than only that a number changed.
    demotions: list[dict[str, str]] = field(default_factory=list)
    #: Set when the provider failed. The caller decides what to do; this
    #: module never silently keeps an unverified assertion.
    unavailable: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "demoted": self.demoted,
            "upheld": self.upheld,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "demotions": self.demotions,
            "unavailable": self.unavailable,
        }


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def quote_is_present(quote: str, evidence: str) -> float:
    """How much of `quote` appears in `evidence`, 0.0 to 1.0.

    Token overlap rather than substring: a model reproducing a sentence
    reliably changes whitespace and drops a trailing clause, and an exact test
    would reject quotes that are really there. This is a check against
    fabrication, not a transcription grader.
    """
    q = _normalize(quote)
    if not q:
        return 0.0
    haystack = _normalize(evidence)
    if q in haystack:
        return 1.0
    words = q.split()
    if not words:
        return 0.0
    present = sum(1 for w in words if w in haystack)
    return present / len(words)


class Corroborator:
    """Asks, per assertive assessment, whether the evidence reports the thing."""

    version = CORROBORATOR_VERSION

    def __init__(
        self,
        store: SqliteStore,
        provider: LLMProvider,
        *,
        provider_id: str,
        model_id: str,
    ) -> None:
        self.store = store
        self.provider = provider
        self.provider_id = provider_id
        self.model_id = model_id

    def derivation_key(self, evidence_hash: str, claim_id: str) -> DerivationKey:
        return DerivationKey(
            kind="corroboration",
            content_hash=text_hash(f"{evidence_hash}|{claim_id}"),
            processor_version=self.version,
            model_id=f"{self.provider_id}|{self.model_id}",
            prompt_version=CORROBORATION_PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )

    def review(
        self,
        records: Sequence[AssessmentRecord],
        claims: Sequence[Claim],
        spans: Sequence[Span],
        evidence_hash: str,
    ) -> tuple[list[AssessmentRecord], CorroborationOutcome]:
        """Return the records with unsupported assertions demoted.

        Records that are not assertive pass through untouched and uncounted.
        """
        outcome = CorroborationOutcome()
        by_id = {c.id: c for c in claims}
        evidence = "\n\n".join(s.text for s in spans)
        reviewed: list[AssessmentRecord] = []

        for record in records:
            if record.classification not in ASSERTIVE:
                reviewed.append(record)
                continue

            claim = by_id.get(record.claim_id)
            if claim is None:
                # Cannot check what cannot be read. Leave it alone rather than
                # demoting on a bookkeeping failure.
                reviewed.append(record)
                continue

            outcome.checked += 1
            verdict = self._check(claim, evidence, evidence_hash, outcome)
            if verdict is None:
                # The provider failed mid-pass. Return the ORIGINAL records
                # untouched: a half-reviewed batch, where some assertions were
                # verified and some silently were not, is worse than an
                # unreviewed one, because nothing downstream could tell which
                # is which. `outcome.unavailable` is set; the caller decides.
                return list(records), outcome

            supported, reason = verdict
            if supported:
                outcome.upheld += 1
                reviewed.append(record)
                continue

            outcome.demoted += 1
            outcome.demotions.append(
                {
                    "claim_id": record.claim_id,
                    "was": record.classification.value,
                    "reason": reason,
                }
            )
            log.info(
                "corroboration_demoted",
                claim=record.claim_id[:12],
                was=record.classification.value,
            )
            # The demotion is recorded in the rationale, not just in the
            # classification: a reviewer seeing INSUFFICIENT_EVIDENCE should be
            # able to tell it was demoted and why, rather than assuming the
            # first pass said so.
            reviewed.append(
                record.model_copy(
                    update={
                        "classification": AssessmentClass.INSUFFICIENT_EVIDENCE,
                        "rationale": f"{record.rationale} [corroboration: {reason}]"[:800],
                    }
                )
            )

        if outcome.checked:
            log.info(
                "corroboration_complete",
                checked=outcome.checked,
                demoted=outcome.demoted,
                llm_calls=outcome.llm_calls,
            )
        return reviewed, outcome

    def _check(
        self,
        claim: Claim,
        evidence: str,
        evidence_hash: str,
        outcome: CorroborationOutcome,
    ) -> tuple[bool, str] | None:
        """True/False plus a reason, or None when the provider failed."""
        key = self.derivation_key(evidence_hash, claim.id)
        cached = self.store.get_derivation(key.value())
        if cached is not None:
            outcome.cache_hits += 1
            return bool(cached.get("reports_outcome")), str(cached.get("reason", "cached"))

        request = CompletionRequest(
            messages=[
                Message(role="system", content=CORROBORATION_SYSTEM),
                Message(
                    role="user",
                    content=CORROBORATION_INSTRUCTION.format(
                        claim=claim.statement, evidence=evidence
                    ),
                ),
            ],
            model_role="analysis",
            temperature=0.0,
        )

        try:
            result = self.provider.structured(request, CorroborationCheck)
        except ProviderUnavailable as exc:
            outcome.unavailable = str(exc)
            log.warning("corroboration_unavailable", error=str(exc)[:160])
            return None
        except (StructuredOutputError, LLMError) as exc:
            # A check that cannot be completed must not read as a pass. The
            # assertion it was reviewing stays unverified, and the caller is
            # told so through `unavailable`.
            outcome.unavailable = f"corroboration failed: {exc}"
            log.warning("corroboration_failed", error=str(exc)[:160])
            return None

        outcome.llm_calls += 1
        supported = bool(result.reports_outcome)
        reason = result.rationale.strip()[:200]

        if supported:
            # The quote is the check on the boolean. A yes that cannot point at
            # a sentence in the evidence is not a yes.
            score = quote_is_present(result.quote, evidence)
            if score < QUOTE_MATCH_FLOOR:
                supported = False
                reason = (
                    "claimed the passage reports the outcome but quoted text "
                    f"not found in it (match {score:.2f})"
                )
                log.info("corroboration_quote_unverified", claim=claim.id[:12], score=score)

        self.store.put_derivation(
            key.value(),
            "corroboration",
            key.content_hash,
            {"reports_outcome": supported, "reason": reason, "quote": result.quote},
        )
        return supported, reason
