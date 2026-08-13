"""Semantic evidence assessment — the one place Forge actually reasons.

Everything else in the evolution workflow is deterministic: narrowing, claim
retrieval, impact mapping, proposal construction, activation. This module is
where judgement is genuinely required, and therefore the only place a model is
spent.

Four rules are enforced here in code, not requested in the prompt:

1. **Grounding.** Every cited span id must be one that was actually shown to
   the model and exists in the store. A citation to anything else means the
   assessment is rejected — never repaired. Repairing a hallucinated citation
   would mean fabricating the evidence for a knowledge change.
2. **Conservatism.** ``CONTRADICTS`` does not exist in the vocabulary. The
   strongest available judgement is ``POTENTIAL_CONFLICT``, which routes to a
   human.
3. **No silent downgrade.** If the configured provider is unavailable, the
   result is ``SEMANTIC_ANALYSIS_UNAVAILABLE``. Forge does not quietly ask a
   weaker model whether your knowledge is wrong.
4. **Identity is recorded.** Provider, model, prompt version, and schema
   version travel with every assessment, and all four are in its derivation
   key. An assessment is not a fact about the world; it is a fact about what
   *that model* concluded under *those instructions*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..domain import (
    AssessmentClass,
    AssessmentRecord,
    Claim,
    Span,
)
from ..ids import text_hash
from ..ingestion.derivation import CacheStats, DerivationKey
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
from .prompts import ASSESSMENT_INSTRUCTION, PROMPT_VERSION, SYSTEM, claims_block, evidence_block
from .schemas import SCHEMA_VERSION, AssessmentResponse

log = get_logger(__name__)

ASSESSOR_VERSION = "assessor/0.1.0"

#: Claims per model call. Batching keeps cost down, but a large batch degrades
#: quality and makes one malformed field discard a lot of work.
DEFAULT_BATCH_SIZE = 6


class AssessmentOutcome(str, Enum):
    """Why an assessment run ended the way it did.

    These are the failure modes named in the Phase 4 brief, as values rather
    than as log strings, so the workflow can route on them and tests can assert
    them.
    """

    COMPLETED = "completed"
    #: Provider could not be reached or has no credential. Resumable.
    SEMANTIC_ANALYSIS_UNAVAILABLE = "semantic_analysis_unavailable"
    #: Output was malformed, or cited evidence that does not exist.
    ASSESSMENT_REJECTED = "assessment_rejected"
    #: Transport failure that may succeed later.
    RETRYABLE_FAILURE = "retryable_failure"


@dataclass
class AssessmentBatch:
    """Result of assessing new evidence against a set of claims."""

    outcome: AssessmentOutcome = AssessmentOutcome.COMPLETED
    records: list[AssessmentRecord] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    cache: CacheStats = field(default_factory=CacheStats)
    llm_calls: int = 0
    duration_ms: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is AssessmentOutcome.COMPLETED

    def by_classification(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.classification.value] = counts.get(record.classification.value, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "assessments": len(self.records),
            "by_classification": self.by_classification(),
            "rejected": self.rejected,
            "cache": self.cache.to_dict(),
            "llm_calls": self.llm_calls,
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
        }


class EvidenceAssessor:
    """Assesses new evidence against existing claims, with caching and grounding."""

    version = ASSESSOR_VERSION

    def __init__(
        self,
        store: SqliteStore,
        provider: LLMProvider,
        *,
        provider_id: str,
        model_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.store = store
        self.provider = provider
        self.provider_id = provider_id
        self.model_id = model_id
        self.batch_size = batch_size

    # -- entry point -------------------------------------------------------

    def assess(self, spans: Sequence[Span], claims: Sequence[Claim]) -> AssessmentBatch:
        """Assess every claim against the evidence spans.

        Cached assessments are returned without a model call. Only the claims
        that miss the cache are sent, which is what makes re-running a workflow
        nearly free.
        """
        batch = AssessmentBatch()
        started = time.perf_counter()
        if not spans or not claims:
            batch.detail = "nothing to assess"
            return batch

        evidence_hash = self._evidence_hash(spans)
        allowed_spans = {s.id for s in spans}

        pending: list[Claim] = []
        for claim in claims:
            key = self.derivation_key(evidence_hash, claim.id)
            cached = self.store.get_derivation(key.value())
            if cached is not None:
                batch.cache.hit()
                batch.records.append(self._record_from_cache(cached, key.value()))
            else:
                batch.cache.miss()
                pending.append(claim)

        if not pending:
            batch.duration_ms = (time.perf_counter() - started) * 1000
            log.info("assessment_all_cached", claims=len(claims))
            return batch

        for chunk in _chunks(pending, self.batch_size):
            outcome = self._assess_chunk(spans, chunk, evidence_hash, allowed_spans, batch)
            if outcome is not AssessmentOutcome.COMPLETED:
                batch.outcome = outcome
                batch.duration_ms = (time.perf_counter() - started) * 1000
                return batch

        batch.duration_ms = (time.perf_counter() - started) * 1000
        log.info(
            "assessment_complete",
            assessed=len(batch.records),
            llm_calls=batch.llm_calls,
            **batch.by_classification(),
        )
        return batch

    # -- derivation --------------------------------------------------------

    def derivation_key(self, evidence_hash: str, claim_id: str) -> DerivationKey:
        """Everything that can invalidate this judgement.

        Note ``model_id`` carries the *provider* too. Two providers serving the
        same model name are still two different things, and a cached assessment
        must not survive a provider switch.
        """
        return DerivationKey(
            kind="assessment",
            content_hash=text_hash(f"{evidence_hash}|{claim_id}"),
            processor_version=self.version,
            model_id=f"{self.provider_id}|{self.model_id}",
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )

    def _evidence_hash(self, spans: Sequence[Span]) -> str:
        return text_hash("|".join(sorted(s.content_hash for s in spans)))

    # -- one model call ----------------------------------------------------

    def _assess_chunk(
        self,
        spans: Sequence[Span],
        claims: Sequence[Claim],
        evidence_hash: str,
        allowed_spans: set[str],
        batch: AssessmentBatch,
    ) -> AssessmentOutcome:
        request = CompletionRequest(
            messages=[
                Message(role="system", content=SYSTEM),
                Message(
                    role="user",
                    content="\n\n".join(
                        [
                            evidence_block(
                                [(s.id, s.citation(), s.text) for s in spans]
                            ),
                            claims_block(
                                [
                                    (c.id, self._concept_name(c), c.statement)
                                    for c in claims
                                ]
                            ),
                            ASSESSMENT_INSTRUCTION,
                        ]
                    ),
                ),
            ],
            model_role="analysis",
            temperature=0.0,
        )

        try:
            response = self.provider.structured(request, AssessmentResponse)
        except ProviderUnavailable as exc:
            # The safety rule: no substitute model, no partial answer.
            batch.detail = str(exc)
            log.warning("semantic_analysis_unavailable", error=str(exc)[:160])
            return AssessmentOutcome.SEMANTIC_ANALYSIS_UNAVAILABLE
        except StructuredOutputError as exc:
            batch.detail = f"model output did not validate: {exc}"
            batch.rejected.append({"reason": "malformed_output", "detail": str(exc)[:200]})
            log.warning("assessment_rejected", reason="malformed_output")
            return AssessmentOutcome.ASSESSMENT_REJECTED
        except LLMError as exc:
            batch.detail = str(exc)
            log.warning("assessment_retryable_failure", error=str(exc)[:160])
            return AssessmentOutcome.RETRYABLE_FAILURE

        batch.llm_calls += 1
        claim_ids = {c.id for c in claims}

        for assessment in response.assessments:
            problem = self._reject_reason(assessment, claim_ids, allowed_spans)
            if problem is not None:
                batch.rejected.append(
                    {
                        "claim_id": assessment.claim_id,
                        "reason": problem,
                        "classification": assessment.classification.value,
                    }
                )
                log.warning(
                    "assessment_rejected", claim=assessment.claim_id[:12], reason=problem
                )
                continue

            key = self.derivation_key(evidence_hash, assessment.claim_id)
            record = AssessmentRecord(
                claim_id=assessment.claim_id,
                classification=assessment.classification,
                rationale=assessment.rationale,
                evidence_span_ids=tuple(assessment.evidence_span_ids),
                provider_id=self.provider_id,
                model_id=self.model_id,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                derivation_key=key.value(),
            )
            payload = {
                **record.to_dict(),
                "refined_statement": assessment.refined_statement,
                "derivation_key": key.value(),
            }
            self.store.put_derivation(
                key.value(), "assessment", key.content_hash, payload
            )
            batch.cache.write()
            batch.records.append(record)

        return AssessmentOutcome.COMPLETED

    # -- validation --------------------------------------------------------

    def _reject_reason(
        self, assessment: Any, claim_ids: set[str], allowed_spans: set[str]
    ) -> str | None:
        """Why this assessment must be discarded, or ``None`` if it is sound."""
        if assessment.claim_id not in claim_ids:
            return "claim_id was not one of the claims presented"

        unknown = [s for s in assessment.evidence_span_ids if s not in allowed_spans]
        if unknown:
            # The model cited a span it was not shown. It may exist elsewhere
            # in the store or not at all; either way this is fabrication with
            # respect to the question asked.
            return f"cited span(s) not present in the evidence shown: {unknown[:3]}"

        missing = [s for s in assessment.evidence_span_ids if self.store.get_span(s) is None]
        if missing:
            return f"cited span(s) do not exist in the store: {missing[:3]}"

        if (
            assessment.classification is AssessmentClass.REFINES
            and not assessment.refined_statement.strip()
        ):
            # A refinement with no refined statement is not actionable: there
            # would be nothing to propose.
            return "REFINES requires a refined_statement"

        return None

    def _record_from_cache(self, payload: dict[str, Any], key: str) -> AssessmentRecord:
        return AssessmentRecord(
            claim_id=payload["claim_id"],
            classification=AssessmentClass(payload["classification"]),
            rationale=payload["rationale"],
            evidence_span_ids=tuple(payload["evidence_span_ids"]),
            provider_id=payload["provider_id"],
            model_id=payload["model_id"],
            prompt_version=payload["prompt_version"],
            schema_version=payload["schema_version"],
            derivation_key=key,
            cached=True,
        )

    def refined_statement_for(self, record: AssessmentRecord) -> str:
        """Recover the refined statement from the cached payload.

        Kept out of :class:`AssessmentRecord` because it is only meaningful for
        one classification, and a field that is empty for four of five values
        invites treating "" as an answer.
        """
        payload = self.store.get_derivation(record.derivation_key) or {}
        return str(payload.get("refined_statement", "")).strip()

    def _concept_name(self, claim: Claim) -> str:
        if not claim.subject_concept_id:
            return ""
        concept = self.store.get_concept(claim.subject_concept_id)
        return concept.qualified_name if concept else ""


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
