"""The knowledge-evolution workflow.

**LangGraph is orchestration, not the intelligence itself.** Every node in this
module is a thin adapter over a service that already existed or is plain Python
in this package. Nothing here parses, chunks, hashes, searches, traverses,
validates provenance, or decides safety — those live where they lived before,
and this file would be deleted without any of them changing.

What LangGraph is genuinely buying, and why a hand-rolled loop was not enough:

=====================  =====================================================
persistent state       The run survives a process exit mid-review.
conditional routing    Six of the ten nodes can end the run early.
interruption           ``await_human_review`` stops the graph itself.
resumability           A resumed run continues; it does not restart.
checkpoints            Expensive semantic work is never repeated.
=====================  =====================================================

The graph::

    START -> register_evidence -> identify_affected_concepts
          -> retrieve_related_claims -> assess_evidence -> classify_impact
          -> generate_proposals -> await_human_review -> activate_changes
          -> record_revision -> finalize_workflow -> END

with early exits from almost every node to ``finalize_workflow``, because "no
existing knowledge is affected" is the common case and must be cheap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings
from ..domain import (
    AssessmentClass,
    ImpactClass,
    ProposalStatus,
    Span,
    WorkflowRun,
    WorkflowStatus,
)
from ..identity import IdentityService
from ..llm.base import LLMProvider
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore
from .activation import EvolutionActivator
from .assessor import AssessmentOutcome, EvidenceAssessor
from .candidates import CandidateNarrower
from .claims import ClaimRetriever
from .impact import actionable, classify_impact
from .proposer import EvolutionProposer
from .prompts import PROMPT_VERSION
from .schemas import SCHEMA_VERSION
from .state import (
    APPROVAL_APPROVED,
    APPROVAL_NOT_REQUIRED,
    APPROVAL_REJECTED,
    EvolutionState,
)

log = get_logger(__name__)

WORKFLOW_VERSION = "evolution-workflow/0.1.0"

#: How many times one run may pause for review before giving up. Each resume
#: that leaves proposals undecided pauses again; this bounds that rather than
#: looping forever if a caller resumes in a script without ever deciding.
MAX_REVIEW_ROUNDS = 25


class OrchestratorUnavailable(RuntimeError):
    """LangGraph is not installed.

    Raised only when the evolution workflow is actually requested. Indexing,
    ingestion, retrieval, activation, and the graph must all keep working on a
    clone that never installed the ``agent`` extra.
    """


def _langgraph() -> Any:
    """Import LangGraph lazily, with an actionable error."""
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
    except ImportError as exc:  # pragma: no cover - exercised by the CLI path
        raise OrchestratorUnavailable(
            "the knowledge-evolution workflow requires LangGraph. Install it with "
            "`pip install -e '.[agent]'`. Every other Forge command works without it."
        ) from exc
    return START, END, StateGraph, interrupt, Command


@dataclass
class WorkflowContext:
    """Everything the nodes need. Deliberately **not** part of graph state.

    Services hold database connections and providers, neither of which can be
    checkpointed. Keeping them here — captured by the node closures — is what
    lets the state itself stay small and serializable.
    """

    store: SqliteStore
    settings: Settings
    provider: LLMProvider | None
    provider_id: str
    model_id: str
    run: WorkflowRun
    identity: IdentityService = field(default_factory=IdentityService)
    max_candidates: int = 12

    def narrower(self) -> CandidateNarrower:
        return CandidateNarrower(
            self.store, identity=self.identity, max_candidates=self.max_candidates
        )

    def retriever(self) -> ClaimRetriever:
        return ClaimRetriever(self.store)

    def assessor(self) -> EvidenceAssessor:
        if self.provider is None:  # pragma: no cover - guarded before use
            raise RuntimeError("no provider bound")
        return EvidenceAssessor(
            self.store,
            self.provider,
            provider_id=self.provider_id,
            model_id=self.model_id,
        )

    def proposer(self) -> EvolutionProposer:
        return EvolutionProposer(
            self.store, workflow_id=self.run.id, source_id=self.run.source_id
        )

    def activator(self) -> EvolutionActivator:
        return EvolutionActivator(self.store)


# --------------------------------------------------------------------------
# node instrumentation
# --------------------------------------------------------------------------


def _timed(name: str, fn: Callable[[EvolutionState], dict[str, Any]]):
    """Wrap a node so every execution is measured and recorded.

    Observability is not optional here: "why did Forge propose this?" needs the
    node sequence, and "why did this cost so much?" needs per-node call counts.
    Both are unanswerable if timing is bolted on later.
    """

    def node(state: EvolutionState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = fn(state)
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000
            log.error("node_failed", node=name, error=str(exc)[:200])
            return {
                "status": WorkflowStatus.FAILED.value,
                "errors": [*state.get("errors", []), f"{name}: {type(exc).__name__}: {exc}"],
                "node_log": [
                    *state.get("node_log", []),
                    {"node": name, "duration_ms": duration, "failed": True, "note": str(exc)[:200]},
                ],
            }

        entry = {
            "node": name,
            "duration_ms": (time.perf_counter() - started) * 1000,
            "llm_calls": int(result.pop("_llm_calls", 0)),
            "cache_hits": int(result.pop("_cache_hits", 0)),
            "cache_misses": int(result.pop("_cache_misses", 0)),
            "note": result.pop("_note", None),
        }
        result["node_log"] = [*state.get("node_log", []), entry]
        result["llm_calls"] = state.get("llm_calls", 0) + entry["llm_calls"]
        result["cache_hits"] = state.get("cache_hits", 0) + entry["cache_hits"]
        result["cache_misses"] = state.get("cache_misses", 0) + entry["cache_misses"]
        return result

    return node


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def build_nodes(ctx: WorkflowContext) -> dict[str, Callable[[EvolutionState], dict[str, Any]]]:
    """Construct the node functions over a context. Testable without a graph."""
    START, END, StateGraph, interrupt, Command = _langgraph()

    def _spans(state: EvolutionState) -> list[Span]:
        return [
            s
            for s in (ctx.store.get_span(i) for i in state.get("evidence_span_ids", []))
            if s is not None
        ]

    # -- 1 ---------------------------------------------------------------
    def register_evidence(state: EvolutionState) -> dict[str, Any]:
        """Resolve the source's spans. Deterministic; reuses Phase 2 ingestion output."""
        source_id = state["source_id"]
        documents = ctx.store.documents_for_source(source_id)
        if not documents:
            return {
                "status": WorkflowStatus.FAILED.value,
                "errors": [f"source {source_id[:12]} has no ingested document"],
                "_note": "no document",
            }
        document = documents[-1]
        spans = list(ctx.store.spans_for_document(document.id))
        if not spans:
            return {
                "status": WorkflowStatus.FAILED.value,
                "errors": [f"document {document.id[:12]} has no spans"],
                "_note": "no spans",
            }
        return {
            "document_id": document.id,
            "evidence_span_ids": [s.id for s in spans],
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "_note": f"{len(spans)} evidence span(s)",
        }

    # -- 2 ---------------------------------------------------------------
    def identify_affected_concepts(state: EvolutionState) -> dict[str, Any]:
        """Deterministic narrowing. Makes zero model calls, by construction."""
        result = ctx.narrower().narrow(_spans(state))
        return {
            "candidates": [c.to_dict() for c in result.candidates],
            "affected_concept_ids": result.concept_ids(),
            "warnings": (
                [*state.get("warnings", []), "candidate set was truncated"]
                if result.truncated
                else state.get("warnings", [])
            ),
            "_note": f"{len(result.candidates)} of {result.considered} concept(s)",
        }

    # -- 3 ---------------------------------------------------------------
    def retrieve_related_claims(state: EvolutionState) -> dict[str, Any]:
        """Bounded claim retrieval over the Phase 3 graph."""
        result = ctx.retriever().retrieve(state.get("affected_concept_ids", []))
        return {
            "related_claim_ids": result.ids(),
            "_note": f"{len(result.claims)} claim(s) from {len(result.by_concept)} concept(s)",
        }

    # -- 4 ---------------------------------------------------------------
    def assess_evidence(state: EvolutionState) -> dict[str, Any]:
        """The only node that spends a model call."""
        if ctx.provider is None:
            return {
                "status": WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE.value,
                "assessment_outcome": AssessmentOutcome.SEMANTIC_ANALYSIS_UNAVAILABLE.value,
                "errors": [
                    *state.get("errors", []),
                    "no semantic provider available; knowledge was not assessed",
                ],
                "_note": "provider unavailable",
            }

        claims = [
            c
            for c in (ctx.store.get_claim(i) for i in state.get("related_claim_ids", []))
            if c is not None
        ]
        batch = ctx.assessor().assess(_spans(state), claims)

        update: dict[str, Any] = {
            "assessments": [a.to_dict() for a in batch.records],
            "assessment_outcome": batch.outcome.value,
            "_llm_calls": batch.llm_calls,
            "_cache_hits": batch.cache.hits,
            "_cache_misses": batch.cache.misses,
            "_note": f"{len(batch.records)} assessment(s), {batch.llm_calls} call(s)",
        }
        if not batch.ok:
            update["status"] = (
                WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE.value
                if batch.outcome is AssessmentOutcome.SEMANTIC_ANALYSIS_UNAVAILABLE
                else WorkflowStatus.FAILED.value
            )
            update["errors"] = [
                *state.get("errors", []),
                f"{batch.outcome.value}: {batch.detail}",
            ]
        if batch.rejected:
            update["warnings"] = [
                *state.get("warnings", []),
                *(f"assessment rejected: {r['reason']}" for r in batch.rejected),
            ]
        return update

    # -- 5 ---------------------------------------------------------------
    def classify_impact_node(state: EvolutionState) -> dict[str, Any]:
        """Deterministic. Precedence rules, not a second model call."""
        from ..domain import AssessmentRecord

        records = [AssessmentRecord(**a) for a in state.get("assessments", [])]
        impact = classify_impact(
            records,
            claims_examined=len(state.get("related_claim_ids", [])),
            has_new_evidence=bool(state.get("evidence_span_ids")),
        )
        return {"impact": impact.value, "_note": impact.value}

    # -- 6 ---------------------------------------------------------------
    def generate_proposals(state: EvolutionState) -> dict[str, Any]:
        """Assessments become reviewable proposals. No knowledge is mutated."""
        from ..domain import AssessmentRecord

        records = [AssessmentRecord(**a) for a in state.get("assessments", [])]
        assessor = ctx.assessor() if ctx.provider is not None else None
        refined = {}
        if assessor is not None:
            refined = {
                r.claim_id: assessor.refined_statement_for(r)
                for r in records
                if r.classification is AssessmentClass.REFINES
            }
        batch = ctx.proposer().propose(actionable(records), refined_statements=refined)
        return {
            "proposal_ids": batch.all_ids,
            "_note": f"{len(batch.created)} new, {len(batch.existing)} existing",
        }

    # -- 7 ---------------------------------------------------------------
    def await_human_review(state: EvolutionState) -> dict[str, Any]:
        """Stop the graph and wait for a person.

        ``interrupt()`` is first, so nothing below it runs on the pausing pass.
        On resume the node re-reads each proposal's **actual stored status**
        rather than trusting the resume payload: approval happens through
        `forge proposals approve`, which is the one approval mechanism, and a
        resume must not be able to smuggle in a decision nobody recorded.

        Because of that, resuming with nothing decided pauses *again* rather
        than falling through. A resume is not consent, and a workflow that
        quietly completed with its proposals still pending would report success
        for a decision nobody made.
        """
        for _ in range(MAX_REVIEW_ROUNDS):
            pending = [
                p
                for p in (ctx.store.get_proposal(i) for i in state.get("proposal_ids", []))
                if p is not None and p.status is ProposalStatus.PENDING
            ]
            if not pending:
                break
            interrupt(
                {
                    "workflow_id": state["workflow_id"],
                    "reason": "human approval required before knowledge changes",
                    "impact": state.get("impact"),
                    "proposals": [
                        {
                            "id": p.id,
                            "type": p.type.value,
                            "safety": p.safety.value,
                            "target": p.operation.target,
                            "reason": p.reason,
                        }
                        for p in pending
                    ],
                    "how_to_decide": (
                        "forge proposals approve <id>  |  forge proposals reject <id>, "
                        "then: forge workflow resume <workflow-id>"
                    ),
                }
            )

        decided = [
            p
            for p in (ctx.store.get_proposal(i) for i in state.get("proposal_ids", []))
            if p is not None
        ]
        approved = [p.id for p in decided if p.status is ProposalStatus.APPROVED]
        rejected = [p for p in decided if p.status is ProposalStatus.REJECTED]

        if approved:
            status = APPROVAL_APPROVED
        elif rejected:
            status = APPROVAL_REJECTED
        else:
            status = APPROVAL_NOT_REQUIRED

        return {
            "approval_status": status,
            "approved_proposal_ids": approved,
            "_note": f"{len(approved)} approved, {len(rejected)} rejected",
        }

    # -- 8 ---------------------------------------------------------------
    def activate_changes(state: EvolutionState) -> dict[str, Any]:
        """Apply approved proposals. Deterministic; no model involved."""
        activator = ctx.activator()
        before = ctx.store.count_revisions()
        entity_ids: list[str] = []
        failures: list[str] = []

        for proposal_id in state.get("approved_proposal_ids", []):
            proposal = ctx.store.get_proposal(proposal_id)
            if proposal is None:
                continue
            result = activator.activate(proposal)
            if result.entity_id and result.ok:
                entity_ids.append(result.entity_id)
            if result.outcome.value == "failed":
                failures.append(f"{proposal_id[:12]}: {result.reason}")

        update: dict[str, Any] = {
            "activated_entity_ids": entity_ids,
            "_note": f"{len(entity_ids)} entity change(s), {ctx.store.count_revisions() - before} revision(s)",
        }
        if failures:
            update["status"] = WorkflowStatus.FAILED.value
            update["errors"] = [*state.get("errors", []), *failures]
        return update

    # -- 9 ---------------------------------------------------------------
    def record_revision(state: EvolutionState) -> dict[str, Any]:
        """Collect the revisions the activation produced.

        Activation writes revisions itself — the store does, transactionally,
        which is where it belongs. This node gathers their ids into the run so
        `forge workflow inspect` can show exactly what changed.
        """
        from ..domain import EntityType

        revision_ids: list[str] = []
        for entity_id in state.get("activated_entity_ids", []):
            for revision in ctx.store.revisions_for(EntityType.CLAIM, entity_id):
                if revision.cause in state.get("approved_proposal_ids", []):
                    revision_ids.append(revision.id)
        return {
            "revision_ids": revision_ids,
            "_note": f"{len(revision_ids)} revision(s) linked to this run",
        }

    # -- 10 --------------------------------------------------------------
    def finalize_workflow(state: EvolutionState) -> dict[str, Any]:
        """Set the terminal status. Never converts a failure into a success."""
        current = state.get("status", WorkflowStatus.RUNNING.value)
        if current in (
            WorkflowStatus.FAILED.value,
            WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE.value,
        ):
            return {"_note": f"terminal: {current}"}
        return {"status": WorkflowStatus.COMPLETED.value, "_note": "completed"}

    return {
        "register_evidence": _timed("register_evidence", register_evidence),
        "identify_affected_concepts": _timed(
            "identify_affected_concepts", identify_affected_concepts
        ),
        "retrieve_related_claims": _timed("retrieve_related_claims", retrieve_related_claims),
        "assess_evidence": _timed("assess_evidence", assess_evidence),
        "classify_impact": _timed("classify_impact", classify_impact_node),
        "generate_proposals": _timed("generate_proposals", generate_proposals),
        # Not wrapped: the timing wrapper's try/except would swallow the
        # GraphInterrupt that LangGraph raises to pause the run.
        "await_human_review": await_human_review,
        "activate_changes": _timed("activate_changes", activate_changes),
        "record_revision": _timed("record_revision", record_revision),
        "finalize_workflow": _timed("finalize_workflow", finalize_workflow),
    }


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def _halted(state: EvolutionState) -> bool:
    return state.get("status") in (
        WorkflowStatus.FAILED.value,
        WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE.value,
    )


def route_after_evidence(state: EvolutionState) -> str:
    return "finalize_workflow" if _halted(state) else "identify_affected_concepts"


def route_after_candidates(state: EvolutionState) -> str:
    """No candidate concepts means nothing existing can be affected.

    Routed to ``classify_impact`` rather than straight to the end: "this
    evidence is about something Forge does not yet know" is a *finding*
    (NEW_KNOWLEDGE), and skipping the classifier would report it as no impact
    at all. Classification is deterministic and free, so the short path stays
    cheap.
    """
    if _halted(state):
        return "finalize_workflow"
    if not state.get("affected_concept_ids"):
        return "classify_impact"
    return "retrieve_related_claims"


def route_after_claims(state: EvolutionState) -> str:
    """No claims means there is nothing to assess *against* — and no call to spend."""
    if _halted(state):
        return "finalize_workflow"
    if not state.get("related_claim_ids"):
        return "classify_impact"
    return "assess_evidence"


def route_after_assessment(state: EvolutionState) -> str:
    if _halted(state):
        return "finalize_workflow"
    return "classify_impact"


def route_after_impact(state: EvolutionState) -> str:
    if _halted(state) or state.get("impact") in (
        None,
        ImpactClass.NO_MATERIAL_CHANGE.value,
        ImpactClass.NEW_KNOWLEDGE.value,
    ):
        return "finalize_workflow"
    return "generate_proposals"


def route_after_proposals(state: EvolutionState) -> str:
    if _halted(state) or not state.get("proposal_ids"):
        return "finalize_workflow"
    return "await_human_review"


def route_after_review(state: EvolutionState) -> str:
    """Approved changes are applied; rejection ends the run without effect."""
    if _halted(state) or state.get("approval_status") != APPROVAL_APPROVED:
        return "finalize_workflow"
    return "activate_changes"


def route_after_activation(state: EvolutionState) -> str:
    return "finalize_workflow" if _halted(state) else "record_revision"


def build_graph(ctx: WorkflowContext, *, checkpointer: Any = None) -> Any:
    """Compile the evolution graph."""
    START, END, StateGraph, _interrupt, _Command = _langgraph()

    nodes = build_nodes(ctx)
    graph = StateGraph(EvolutionState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "register_evidence")
    graph.add_conditional_edges("register_evidence", route_after_evidence)
    graph.add_conditional_edges("identify_affected_concepts", route_after_candidates)
    graph.add_conditional_edges("retrieve_related_claims", route_after_claims)
    graph.add_conditional_edges("assess_evidence", route_after_assessment)
    graph.add_conditional_edges("classify_impact", route_after_impact)
    graph.add_conditional_edges("generate_proposals", route_after_proposals)
    graph.add_conditional_edges("await_human_review", route_after_review)
    graph.add_conditional_edges("activate_changes", route_after_activation)
    graph.add_edge("record_revision", "finalize_workflow")
    graph.add_edge("finalize_workflow", END)

    return graph.compile(checkpointer=checkpointer)


#: Exported so documentation and tests describe the same graph rather than two
#: drifting copies of it.
NODE_SEQUENCE: tuple[str, ...] = (
    "register_evidence",
    "identify_affected_concepts",
    "retrieve_related_claims",
    "assess_evidence",
    "classify_impact",
    "generate_proposals",
    "await_human_review",
    "activate_changes",
    "record_revision",
    "finalize_workflow",
)
