"""Typed workflow state.

Two constraints shape everything here.

**It must serialize.** The state is checkpointed after every node so the run can
survive a process exit. That rules out storing services, connections, or domain
objects — anything that cannot round-trip through the checkpointer is a bug
waiting for the first resume.

**It must not carry the corpus.** State holds *identifiers*, and the nodes
resolve them against the store when they need the content. A workflow that
embedded span text and claim objects would checkpoint megabytes per step,
and — worse — would go stale: a resumed run would act on a snapshot of
knowledge rather than on knowledge as it now is.

The one exception is small, already-computed records (candidates and
assessments), stored as plain dicts. They are outputs of the run rather than
copies of the knowledge base, and losing them on resume would mean paying for
the model calls twice.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from ..domain import WorkflowRun, WorkflowStatus
from ..domain.provenance import utc_now


def _replace(_old: Any, new: Any) -> Any:
    """Last write wins.

    LangGraph needs a reducer for every key it merges. Replacement is correct
    here because each node owns the keys it writes: no two nodes append to the
    same list, so an accumulating reducer would only create ambiguity about
    who wrote what.
    """
    return new


class EvolutionState(TypedDict, total=False):
    """State of one knowledge-evolution run."""

    # -- identity ----------------------------------------------------------
    workflow_id: str
    source_id: str
    source_locator: str
    document_id: str | None

    # -- evidence ----------------------------------------------------------
    evidence_span_ids: Annotated[list[str], _replace]
    evidence_hash: str

    # -- narrowing ---------------------------------------------------------
    #: CandidateRecord dicts — concept id, selector, and why it was chosen.
    candidates: Annotated[list[dict[str, Any]], _replace]
    affected_concept_ids: Annotated[list[str], _replace]
    related_claim_ids: Annotated[list[str], _replace]

    # -- reasoning ---------------------------------------------------------
    #: AssessmentRecord dicts. Kept in state so a resume never re-pays for a
    #: model call that already succeeded.
    assessments: Annotated[list[dict[str, Any]], _replace]
    assessment_outcome: str
    impact: str | None

    # -- decision and effect -----------------------------------------------
    proposal_ids: Annotated[list[str], _replace]
    approval_status: str
    approval_note: str
    approved_proposal_ids: Annotated[list[str], _replace]
    activated_entity_ids: Annotated[list[str], _replace]
    revision_ids: Annotated[list[str], _replace]

    # -- provider identity -------------------------------------------------
    provider_id: str
    model_id: str
    prompt_version: str
    schema_version: str

    # -- execution metadata ------------------------------------------------
    status: str
    llm_calls: int
    cache_hits: int
    cache_misses: int
    #: NodeExecution dicts, appended by each node as it finishes.
    node_log: Annotated[list[dict[str, Any]], _replace]
    warnings: Annotated[list[str], _replace]
    errors: Annotated[list[str], _replace]


#: Approval outcomes carried in ``approval_status``.
APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
#: Nothing needed a decision, so the run never paused.
APPROVAL_NOT_REQUIRED = "not_required"


def initial_state(
    *,
    workflow_id: str,
    source_id: str,
    source_locator: str,
    provider_id: str,
    model_id: str,
) -> EvolutionState:
    return EvolutionState(
        workflow_id=workflow_id,
        source_id=source_id,
        source_locator=source_locator,
        document_id=None,
        evidence_span_ids=[],
        evidence_hash="",
        candidates=[],
        affected_concept_ids=[],
        related_claim_ids=[],
        assessments=[],
        assessment_outcome="",
        impact=None,
        proposal_ids=[],
        approval_status=APPROVAL_PENDING,
        approval_note="",
        approved_proposal_ids=[],
        activated_entity_ids=[],
        revision_ids=[],
        provider_id=provider_id,
        model_id=model_id,
        prompt_version="",
        schema_version="",
        status=WorkflowStatus.RUNNING.value,
        llm_calls=0,
        cache_hits=0,
        cache_misses=0,
        node_log=[],
        warnings=[],
        errors=[],
    )


def to_run(state: EvolutionState, run: WorkflowRun) -> WorkflowRun:
    """Project checkpoint state onto the durable :class:`WorkflowRun` record.

    The two exist for different reasons and are kept separate on purpose: the
    checkpoint is resumption state owned by the orchestrator and may be pruned,
    while the run is Forge's own permanent answer to "why did this change?".
    This function is the single point where one becomes the other.
    """
    from ..domain import AssessmentRecord, CandidateRecord, ImpactClass, NodeExecution

    impact = state.get("impact")
    return run.model_copy(
        update={
            "status": WorkflowStatus(state.get("status", run.status.value)),
            "document_id": state.get("document_id"),
            "provider_id": state.get("provider_id", run.provider_id),
            "model_id": state.get("model_id", run.model_id),
            "prompt_version": state.get("prompt_version") or run.prompt_version,
            "schema_version": state.get("schema_version") or run.schema_version,
            "evidence_span_ids": tuple(state.get("evidence_span_ids", [])),
            "candidates": tuple(
                CandidateRecord(
                    concept_id=c["concept_id"],
                    concept_name=c["concept"],
                    selector=c["selector"],
                    detail=c.get("detail", ""),
                    score=c.get("score"),
                )
                for c in state.get("candidates", [])
            ),
            "related_claim_ids": tuple(state.get("related_claim_ids", [])),
            "assessments": tuple(
                AssessmentRecord(
                    claim_id=a["claim_id"],
                    classification=a["classification"],
                    rationale=a["rationale"],
                    evidence_span_ids=tuple(a["evidence_span_ids"]),
                    provider_id=a["provider_id"],
                    model_id=a["model_id"],
                    prompt_version=a["prompt_version"],
                    schema_version=a["schema_version"],
                    derivation_key=a.get("derivation_key", ""),
                    cached=a.get("cached", False),
                )
                for a in state.get("assessments", [])
            ),
            "impact": ImpactClass(impact) if impact else None,
            "proposal_ids": tuple(state.get("proposal_ids", [])),
            "activated_entity_ids": tuple(state.get("activated_entity_ids", [])),
            "revision_ids": tuple(state.get("revision_ids", [])),
            "nodes": tuple(NodeExecution(**n) for n in state.get("node_log", [])),
            "warnings": tuple(state.get("warnings", [])),
            "errors": tuple(state.get("errors", [])),
            "updated_at": utc_now(),
        }
    )
