"""Workflow runs — the durable record of how knowledge came to change.

A :class:`WorkflowRun` is not orchestration bookkeeping. It is the answer to
the question a user will eventually ask about every proposal Forge makes:

    "Why did Forge propose this?"

Answering that requires knowing which evidence arrived, which existing
knowledge was considered *and why it was considered*, which model assessed it,
under which prompt and schema, what it concluded, and what a human then
decided. All of that lives here, and it outlives the LangGraph checkpoint —
checkpoints are resumption state and may be pruned; this is history.

Deliberately storage-agnostic and framework-agnostic: nothing here imports
LangGraph. If the orchestrator were replaced tomorrow, this record would be
unchanged, which is the test of whether the framework leaked into the model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..ids import deterministic_id
from .enums import AssessmentClass, ImpactClass, WorkflowStatus
from .provenance import utc_now


class NodeExecution(BaseModel):
    """One node's turn, recorded for observability.

    Duration and call count are per-node because that is the granularity at
    which cost questions are actually asked: "which step is spending the
    tokens?" is unanswerable from a single total.
    """

    model_config = ConfigDict(extra="forbid")

    node: str
    started_at: datetime = Field(default_factory=utc_now)
    duration_ms: float = 0.0
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    note: str | None = None
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "duration_ms": round(self.duration_ms, 2),
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retries": self.retries,
            "failed": self.failed,
            "note": self.note,
        }


class CandidateRecord(BaseModel):
    """A concept the narrowing step selected, and the reason it did.

    The reason is required. A candidate set with no justification is
    indistinguishable from a guess, and the whole point of narrowing
    deterministically before spending a model call is that the narrowing can be
    audited.
    """

    model_config = ConfigDict(extra="forbid")

    concept_id: str
    concept_name: str
    #: e.g. "exact_name", "alias", "heading", "lexical", "graph_neighbour".
    selector: str
    detail: str = ""
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept": self.concept_name,
            "selector": self.selector,
            "detail": self.detail,
            "score": self.score,
        }


class AssessmentRecord(BaseModel):
    """One model judgement about one (evidence, claim) pair.

    Carries its own provider and model identity rather than inheriting the
    run's, because a workflow resumed against a different provider must not be
    able to make two incomparable assessments look like one consistent set.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    classification: AssessmentClass
    rationale: str
    evidence_span_ids: tuple[str, ...]
    provider_id: str
    model_id: str
    prompt_version: str
    schema_version: str
    derivation_key: str
    cached: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "classification": self.classification.value,
            "rationale": self.rationale,
            "evidence_span_ids": list(self.evidence_span_ids),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            # Included so the record round-trips through workflow state: a
            # resumed run must be able to rebuild the record exactly, and the
            # derivation key is what links it back to its cached result.
            "derivation_key": self.derivation_key,
            "cached": self.cached,
        }


class WorkflowRun(BaseModel):
    """One knowledge-evolution run, start to finish."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    document_id: str | None = None
    status: WorkflowStatus = WorkflowStatus.RUNNING

    #: Provider/model the run was *started* with. A resume under a different
    #: provider is recorded rather than hidden — see `provider_changed`.
    provider_id: str = "none"
    model_id: str = "none"
    prompt_version: str = "none"
    schema_version: str = "none"

    evidence_span_ids: tuple[str, ...] = ()
    candidates: tuple[CandidateRecord, ...] = ()
    related_claim_ids: tuple[str, ...] = ()
    assessments: tuple[AssessmentRecord, ...] = ()
    impact: ImpactClass | None = None

    proposal_ids: tuple[str, ...] = ()
    activated_entity_ids: tuple[str, ...] = ()
    revision_ids: tuple[str, ...] = ()

    nodes: tuple[NodeExecution, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @staticmethod
    def make_id(source_id: str, content_hash: str, processor_version: str) -> str:
        """Deterministic identity.

        Re-running evolution over unchanged evidence with unchanged code must
        land on the *same* run, not start a second one — that is what makes
        repeated execution safe rather than merely tolerable.
        """
        return deterministic_id("workflow", source_id, content_hash, processor_version)

    # -- accounting --------------------------------------------------------

    @property
    def llm_calls(self) -> int:
        return sum(n.llm_calls for n in self.nodes)

    @property
    def cache_hits(self) -> int:
        return sum(n.cache_hits for n in self.nodes)

    @property
    def duration_ms(self) -> float:
        return sum(n.duration_ms for n in self.nodes)

    @property
    def awaiting_review(self) -> bool:
        return self.status is WorkflowStatus.WAITING_FOR_REVIEW

    def by_classification(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for assessment in self.assessments:
            key = assessment.classification.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def provider_changed(self, provider_id: str, model_id: str) -> bool:
        """Whether resuming under this provider/model would change identity.

        Used to refuse silent provenance ambiguity on resume: an assessment
        made by one model may not be topped up by another as if they were the
        same judgement.
        """
        if self.provider_id == "none":
            return False
        return (provider_id, model_id) != (self.provider_id, self.model_id)

    def to_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "impact": self.impact.value if self.impact else None,
            "counts": {
                "evidence_spans": len(self.evidence_span_ids),
                "candidates": len(self.candidates),
                "claims_examined": len(self.related_claim_ids),
                "assessments": len(self.assessments),
                "proposals": len(self.proposal_ids),
                "activated": len(self.activated_entity_ids),
                "revisions": len(self.revision_ids),
            },
            "by_classification": self.by_classification(),
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "duration_ms": round(self.duration_ms, 2),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if verbose:
            payload["candidates_detail"] = [c.to_dict() for c in self.candidates]
            payload["assessments_detail"] = [a.to_dict() for a in self.assessments]
            payload["nodes"] = [n.to_dict() for n in self.nodes]
            payload["proposal_ids"] = list(self.proposal_ids)
            payload["activated_entity_ids"] = list(self.activated_entity_ids)
            payload["revision_ids"] = list(self.revision_ids)
        return payload
