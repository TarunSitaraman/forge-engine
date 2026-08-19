"""LLM extraction of concept and claim candidates.

Boundaries this module enforces, all of them Phase 1 rules made operational:

* **A model may never produce ``SOURCE_FACT``.** Extraction produces
  ``EXTRACTED_CLAIM`` at best. The domain layer rejects anything stronger, but
  this module never even asks.
* **A model may never produce ``USER_ASSERTION``.** That tier means the user
  said so; a model saying so is a different thing entirely.
* **Nothing is stored without evidence.** A claim whose quote cannot be found
  in the source text is dropped, and the drop is reported.
* **Malformed output is a failure, not a repair opportunity.** Structured
  responses are schema-validated; failures produce ``FAILED`` or ``PARTIAL``.

Extraction is entirely optional. With no provider, ingestion still succeeds —
it just produces spans without candidates.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..domain import (
    Derivation,
    EntityType,
    ExtractionStatus,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    Span,
)
from ..llm.base import (
    CompletionRequest,
    LLMError,
    LLMProvider,
    Message,
    ProviderUnavailable,
    StructuredOutputError,
)
from ..logging import get_logger
from .prompts import (
    CLAIM_INSTRUCTION,
    CONCEPT_INSTRUCTION,
    PROMPT_VERSION,
    SYSTEM,
    TERMINOLOGY_INSTRUCTION,
)
from .schemas import (
    SCHEMA_VERSION,
    ClaimExtractionResponse,
    ConceptExtractionResponse,
    ExtractedClaim,
    ExtractedConcept,
    TerminologyResponse,
)

log = get_logger(__name__)

EXTRACTOR_VERSION = "extractor/0.2.0"

#: Fraction of a quote's words that must appear **in their original order** in
#: the source span for the quote to count as grounded. See `_grounded` for why
#: order is the load-bearing part.
#:
#: Chosen from the observed margin, not by feel. Eliding words *from* a quote
#: does not lower this ratio — only a word the source does not have in that
#: position does — so every legitimate quote measured (hyphenation changes,
#: curly quotes, ellipsis, dropped interior words) scored **1.000**, while
#: quotes reassembled from the span's own vocabulary scored **0.500-0.857**.
#: 0.9 sits in that gap and still tolerates one substituted word in ten, which
#: covers pluralization and similar drift. Tests pin both sides of the margin.
QUOTE_GROUNDING_THRESHOLD = 0.9

#: Minimum span length worth a model call. Low on purpose: see `_select`.
MIN_SPAN_CHARS = 40


@dataclass
class ConceptCandidate:
    """A concept the model proposed, with its evidence."""

    name: str
    kind: str
    span_id: str
    mention: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "span_id": self.span_id, "mention": self.mention}


@dataclass
class ClaimCandidate:
    """A claim the model proposed, grounded in a span."""

    statement: str
    evidence_quote: str
    span_id: str
    concept: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "evidence_quote": self.evidence_quote,
            "span_id": self.span_id,
            "concept": self.concept,
        }


@dataclass
class ExtractionResult:
    """Outcome of extracting from one or more spans."""

    status: ExtractionStatus
    concepts: list[ConceptCandidate] = field(default_factory=list)
    claims: list[ClaimCandidate] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    llm_calls: int = 0
    duration_seconds: float = 0.0
    #: Why spans failed, and why candidates were dropped. Surfaced, never hidden.
    failures: list[dict[str, str]] = field(default_factory=list)
    model_id: str | None = None
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "concepts": [c.to_dict() for c in self.concepts],
            "claims": [c.to_dict() for c in self.claims],
            "terms": self.terms,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "failures": self.failures,
            "model_id": self.model_id,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionResult:
        """Rehydrate a cached result."""
        return cls(
            status=ExtractionStatus(data["status"]),
            concepts=[ConceptCandidate(**c) for c in data.get("concepts", [])],
            claims=[ClaimCandidate(**c) for c in data.get("claims", [])],
            terms=list(data.get("terms", [])),
            llm_calls=0,  # a cache hit costs nothing
            duration_seconds=0.0,
            failures=list(data.get("failures", [])),
            model_id=data.get("model_id"),
            provider=data.get("provider"),
        )


class CandidateExtractor:
    """Extracts concept and claim candidates from spans using an LLM."""

    version = EXTRACTOR_VERSION
    prompt_version = PROMPT_VERSION
    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        model_role: str = "extraction",
        max_spans: int = 12,
        extract_terms: bool = False,
    ) -> None:
        self.provider = provider
        self.model_role = model_role
        #: Cap on spans sent to the model per document. Cost control: a
        #: 300-page PDF must not turn into 3,000 model calls.
        self.max_spans = max_spans
        self.extract_terms = extract_terms

    @property
    def available(self) -> bool:
        if self.provider is None:
            return False
        try:
            return self.provider.health()[0]
        except Exception:
            return False

    def model_id(self) -> str:
        """Resolved model name, for the derivation key. 'none' when disabled.

        A provider may append an ``identity_variant`` — a mode it is running in
        that makes its output non-comparable with the same model in another
        mode, such as a reasoning model with reasoning switched off. It belongs
        in the model id because that is what the derivation key hashes: two
        modes must not share cache entries, and must not be averaged together
        in an evaluation.
        """
        if self.provider is None:
            return "none"
        variant = str(getattr(self.provider, "identity_variant", "") or "")
        resolve = getattr(self.provider, "resolve_model", None)
        if callable(resolve):
            try:
                return f"{resolve(self.model_role)}{variant}"
            except Exception:
                pass
        caps = self.provider.capabilities
        base = caps.available_models[0] if caps.available_models else caps.name
        return f"{base}{variant}"

    # -- extraction --------------------------------------------------------

    def extract(self, spans: Sequence[Span]) -> ExtractionResult:
        """Extract candidates from the most informative spans.

        Returns ``SKIPPED_NO_PROVIDER`` rather than raising when no model is
        available: deterministic ingestion must still succeed.
        """
        if self.provider is None:
            return ExtractionResult(status=ExtractionStatus.SKIPPED_NO_PROVIDER)

        reachable, detail = self.provider.health()
        if not reachable:
            log.info("extraction_skipped_no_provider", detail=detail)
            return ExtractionResult(
                status=ExtractionStatus.SKIPPED_NO_PROVIDER,
                failures=[{"kind": "provider_unavailable", "error": detail}],
            )

        selected = self._select(spans)
        if not selected:
            return ExtractionResult(status=ExtractionStatus.SUCCEEDED)

        started = time.perf_counter()
        result = ExtractionResult(
            status=ExtractionStatus.SUCCEEDED,
            model_id=self.model_id(),
            provider=self.provider.capabilities.name,
        )

        attempted = succeeded = 0
        for span in selected:
            attempted += 1
            span_ok = True
            span_started = time.perf_counter()

            concepts, calls, failure = self._concepts(span)
            result.llm_calls += calls
            if failure:
                result.failures.append(failure)
                span_ok = False
            result.concepts.extend(concepts)

            claims, calls, failure, drops = self._claims(span)
            result.llm_calls += calls
            if failure:
                result.failures.append(failure)
                span_ok = False
            # Dropped ungrounded claims are reported but do NOT fail the span.
            # The grounding check rejecting a fabricated quote is the filter
            # working as designed; treating it as an extraction failure would
            # discard the span's good candidates and prevent caching a result
            # that is, in fact, correct.
            result.failures.extend(drops)
            result.claims.extend(claims)

            if span_ok:
                succeeded += 1

            # Emitted per span, not per document. At the measured local latency
            # a single document is ~25 minutes of silence otherwise, which is
            # indistinguishable from a hang during an overnight run.
            log.info(
                "extraction_span_complete",
                span=attempted,
                of=len(selected),
                ok=span_ok,
                seconds=round(time.perf_counter() - span_started, 1),
                concepts=len(result.concepts),
                claims=len(result.claims),
            )

        if self.extract_terms and selected:
            terms, calls, failure = self._terms(selected[0])
            result.llm_calls += calls
            result.terms = terms
            if failure:
                result.failures.append(failure)

        result.duration_seconds = time.perf_counter() - started
        result.status = _status(attempted, succeeded)
        log.info(
            "extraction_complete",
            status=result.status.value,
            spans=attempted,
            concepts=len(result.concepts),
            claims=len(result.claims),
            llm_calls=result.llm_calls,
        )
        return result

    # -- per-task ----------------------------------------------------------

    def _concepts(self, span: Span) -> tuple[list[ConceptCandidate], int, dict[str, str] | None]:
        parsed, calls, failure = self._call(CONCEPT_INSTRUCTION, span, ConceptExtractionResponse)
        if parsed is None:
            return [], calls, failure

        out: list[ConceptCandidate] = []
        for concept in parsed.concepts:
            out.append(
                ConceptCandidate(
                    name=concept.name.strip(),
                    kind=concept.kind.strip() or "concept",
                    span_id=span.id,
                    mention=concept.mention.strip(),
                )
            )
        return out, calls, failure

    def _claims(
        self, span: Span
    ) -> tuple[list[ClaimCandidate], int, dict[str, str] | None, list[dict[str, str]]]:
        """Returns (claims, calls, call_failure, dropped).

        ``call_failure`` means the model's response was unusable. ``dropped``
        means individual candidates were rejected by the grounding check —
        a different thing, and not a failure of the span.
        """
        parsed, calls, failure = self._call(CLAIM_INSTRUCTION, span, ClaimExtractionResponse)
        if parsed is None:
            return [], calls, failure, []

        out: list[ClaimCandidate] = []
        dropped: list[dict[str, str]] = []
        for claim in parsed.claims:
            # Grounding check: the quote must actually be in the span. This is
            # the difference between "the model cited the text" and "the model
            # produced a plausible-looking citation".
            if not _grounded(claim.evidence_quote, span.text):
                dropped.append(
                    {
                        "kind": "ungrounded_quote",
                        "span_id": span.id,
                        "error": (
                            f"dropped claim whose evidence quote is not present in the "
                            f"span: {claim.evidence_quote[:80]!r}"
                        ),
                    }
                )
                log.warning("dropped_ungrounded_claim", span_id=span.id)
                continue
            out.append(
                ClaimCandidate(
                    statement=claim.statement.strip(),
                    evidence_quote=claim.evidence_quote.strip(),
                    span_id=span.id,
                    concept=claim.concept.strip(),
                )
            )
        return out, calls, failure, dropped

    def _terms(self, span: Span) -> tuple[list[str], int, dict[str, str] | None]:
        parsed, calls, failure = self._call(TERMINOLOGY_INSTRUCTION, span, TerminologyResponse)
        return ([t.strip() for t in parsed.terms] if parsed else []), calls, failure

    def _call(
        self, instruction: str, span: Span, schema: type
    ) -> tuple[Any, int, dict[str, str] | None]:
        """One structured model call. Failures are values, never exceptions."""
        request = CompletionRequest(
            messages=[
                Message(role="system", content=SYSTEM),
                Message(
                    role="user",
                    content=f"{instruction}\n\n--- TEXT START ---\n{span.text}\n--- TEXT END ---",
                ),
            ],
            model_role=self.model_role,
            temperature=0.0,
        )
        try:
            return self.provider.structured(request, schema), 1, None  # type: ignore[union-attr]
        except StructuredOutputError as exc:
            log.warning("extraction_schema_failure", span_id=span.id, error=str(exc)[:200])
            return None, 1, {
                "kind": "schema_violation",
                "span_id": span.id,
                "error": str(exc)[:300],
                "raw_excerpt": exc.raw[:200],
            }
        except ProviderUnavailable as exc:
            return None, 0, {"kind": "provider_unavailable", "span_id": span.id, "error": str(exc)[:200]}
        except LLMError as exc:
            return None, 1, {"kind": "llm_error", "span_id": span.id, "error": str(exc)[:200]}

    def _select(self, spans: Sequence[Span]) -> list[Span]:
        """Choose which spans to spend model calls on.

        Longest-first, because a bare heading carries nothing to extract and
        costs a full call. Restored to document order so the run is
        deterministic.

        The floor is deliberately low. An earlier 120-character threshold
        silently discarded short but entirely meaningful sections — a two-line
        definition of a data structure, for instance — which then never
        appeared as a candidate at all. Dropping content without saying so is
        the wrong failure: better to spend a cheap call than to lose the
        concept.
        """
        usable = [s for s in spans if len(s.text.strip()) >= MIN_SPAN_CHARS]
        chosen = sorted(usable, key=lambda s: -len(s.text))[: self.max_spans]
        return sorted(chosen, key=lambda s: s.ordinal)


# -- provenance ------------------------------------------------------------


def extraction_provenance(
    model_id: str,
    span: Span,
    *,
    tier: ProvenanceTier = ProvenanceTier.EXTRACTED_CLAIM,
    workflow_run_id: str | None = None,
) -> Provenance:
    """Provenance for a model-extracted object.

    Tier is capped at ``EXTRACTED_CLAIM``. ``SOURCE_FACT`` is rejected outright
    rather than silently downgraded — asking for it is a programming error, and
    quietly fixing it would hide the bug.
    """
    if tier in (ProvenanceTier.SOURCE_FACT, ProvenanceTier.USER_ASSERTION):
        raise ValueError(
            f"extraction cannot produce {tier.value}: a model may not assert source "
            f"facts or speak as the user"
        )
    return Provenance(
        tier=tier,
        derivation=Derivation.MODEL,
        agent="CandidateExtractor",
        agent_version=EXTRACTOR_VERSION,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        inputs=(
            ProvenanceInput(
                entity_type=EntityType.SPAN,
                entity_id=span.id,
                tier=ProvenanceTier.SOURCE_FACT,
            ),
        ),
        workflow_run_id=workflow_run_id,
    )


# -- helpers ---------------------------------------------------------------


def _squash(text: str) -> str:
    """Reduce text to lower-case alphanumerics.

    Deletes every distinction a model routinely gets wrong when transcribing a
    quote — whitespace, straight vs curly quotes, hyphens, em dashes, trailing
    punctuation — while preserving the one that carries meaning: the order of
    the characters.
    """
    return "".join(c for c in text.lower() if c.isalnum())


def _tokens(text: str) -> list[str]:
    return [w for w in re.sub(r"[^0-9a-z]+", " ", text.lower()).split() if w]


def _ordered_overlap(quote_words: Sequence[str], text_words: Sequence[str]) -> float:
    """Fraction of the quote's words appearing in the text *in the same order*.

    A longest-common-subsequence ratio. Unlike a set intersection this cannot
    be satisfied by rearranging the source's own vocabulary.
    """
    if not quote_words:
        return 0.0
    previous = [0] * (len(text_words) + 1)
    for q in quote_words:
        current = [0] * (len(text_words) + 1)
        for j, t in enumerate(text_words, 1):
            current[j] = previous[j - 1] + 1 if q == t else max(previous[j], current[j - 1])
        previous = current
    return previous[len(text_words)] / len(quote_words)


def _grounded(quote: str, text: str) -> bool:
    """Is this quote actually present in the span?

    **Order is the whole point.** An earlier version compared bag-of-words
    overlap, which accepts any quote assembled from the span's own vocabulary —
    including one that inverts the meaning. Given a span saying "RAG improves
    accuracy", the fabricated quote "RAG does not improve accuracy" scored 100%
    and was stored as evidence. That defeats the rule this function exists to
    enforce: nothing is stored without evidence.

    Two order-preserving checks, cheapest first:

    1. The quote, stripped to bare alphanumerics, appears as a **substring** of
       the span stripped the same way. This is the common case and absorbs all
       formatting noise. A quote may use ``...`` to elide, in which case each
       segment must appear in order.
    2. Otherwise, a longest-common-subsequence ratio over words, which tolerates
       a model dropping interior words but still requires what remains to be in
       the original sequence.
    """
    if not quote.strip():
        return False

    squashed_text = _squash(text)
    segments = [seg for seg in re.split(r"\.\.\.|…", quote) if _squash(seg)]
    if segments:
        position = 0
        for segment in segments:
            found = squashed_text.find(_squash(segment), position)
            if found < 0:
                break
            position = found + len(_squash(segment))
        else:
            return True

    return _ordered_overlap(_tokens(quote), _tokens(text)) >= QUOTE_GROUNDING_THRESHOLD


def _status(attempted: int, succeeded: int) -> ExtractionStatus:
    if attempted == 0 or succeeded == attempted:
        return ExtractionStatus.SUCCEEDED
    if succeeded == 0:
        return ExtractionStatus.FAILED
    return ExtractionStatus.PARTIAL
