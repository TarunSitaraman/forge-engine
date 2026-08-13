"""Facade over the evolution workflow: start, resume, inspect.

Everything the CLI needs, and the only place that knows a checkpointer exists.
Two responsibilities beyond plumbing:

**Durable run records.** The LangGraph checkpoint is resumption state; the
:class:`~forge.domain.WorkflowRun` is Forge's own history. This service keeps
them in step, writing the run after every start and resume so `forge workflow
inspect` works even with LangGraph uninstalled.

**Provider identity on resume.** A run paused under one model and resumed under
another is not the same run. Rather than silently mixing judgements, the resume
is refused unless the caller acknowledges it — see :meth:`resume`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..domain import WorkflowRun, WorkflowStatus
from ..identity import IdentityConfig, IdentityService
from ..ids import text_hash
from ..llm.base import LLMProvider, ProviderUnavailable
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore
from .state import initial_state, to_run
from .workflow import WORKFLOW_VERSION, OrchestratorUnavailable, WorkflowContext, build_graph

log = get_logger(__name__)


@dataclass
class EvolutionOutcome:
    """What one start/resume call did."""

    run: WorkflowRun
    interrupted: bool = False
    interrupt_payload: dict[str, Any] | None = None

    @property
    def status(self) -> WorkflowStatus:
        return self.run.status

    def to_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        payload = self.run.to_dict(verbose=verbose)
        payload["interrupted"] = self.interrupted
        if self.interrupt_payload:
            payload["awaiting"] = self.interrupt_payload
        return payload


class ProviderMismatch(RuntimeError):
    """Resuming would change which model produced this run's assessments."""


class EvolutionService:
    """Runs and resumes knowledge-evolution workflows."""

    version = WORKFLOW_VERSION

    def __init__(
        self,
        store: SqliteStore,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        provider_id: str = "none",
        model_id: str = "none",
        checkpoint_path: Path | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.provider = provider
        self.provider_id = provider_id
        self.model_id = model_id
        # A file, not memory: a checkpoint that does not survive the process
        # cannot support the one feature it exists for.
        self.checkpoint_path = checkpoint_path or (settings.state_dir / "checkpoints.db")
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def _checkpointer(self) -> Any:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - same path as workflow
            raise OrchestratorUnavailable(
                "workflow checkpointing requires LangGraph: pip install -e '.[agent]'"
            ) from exc

        if self._conn is None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.checkpoint_path), check_same_thread=False)
        saver = SqliteSaver(self._conn)
        saver.setup()
        return saver

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- identity ----------------------------------------------------------

    def workflow_id_for(self, source_id: str) -> str:
        """Deterministic run identity.

        Derived from the source's content and the workflow version, so
        re-running evolution over unchanged evidence resumes the same run
        instead of starting a parallel one. This is what makes repeated
        execution safe rather than merely non-fatal.
        """
        source = self.store.get_source(source_id)
        content_hash = source.content_hash if source else "unknown"
        return WorkflowRun.make_id(source_id, content_hash, WORKFLOW_VERSION)

    def _context(self, run: WorkflowRun) -> WorkflowContext:
        identity_path = self.settings.vault_path / "config" / "concept-identity.yaml"
        return WorkflowContext(
            store=self.store,
            settings=self.settings,
            provider=self.provider,
            provider_id=self.provider_id,
            model_id=self.model_id,
            run=run,
            identity=IdentityService(IdentityConfig.load(identity_path)),
        )

    # -- start -------------------------------------------------------------

    def start(self, source_id: str) -> EvolutionOutcome:
        """Begin (or re-enter) evolution for one ingested source."""
        source = self.store.get_source(source_id)
        if source is None:
            raise KeyError(f"unknown source {source_id!r}")

        workflow_id = self.workflow_id_for(source_id)
        existing = self.store.get_workflow(workflow_id)

        if existing is not None and existing.awaiting_review:
            # Re-running a paused workflow must not restart it — that would
            # re-pay for the semantic work and could duplicate proposals.
            log.info("workflow_already_waiting", workflow=workflow_id[:12])
            return self.resume(workflow_id)

        run = existing or WorkflowRun(
            id=workflow_id,
            source_id=source_id,
            provider_id=self.provider_id,
            model_id=self.model_id,
        )
        run = run.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "provider_id": self.provider_id,
                "model_id": self.model_id,
            }
        )
        self.store.put_workflow(run)

        app = build_graph(self._context(run), checkpointer=self._checkpointer())
        state = initial_state(
            workflow_id=workflow_id,
            source_id=source_id,
            source_locator=source.locator,
            provider_id=self.provider_id,
            model_id=self.model_id,
        )
        return self._run(app, run, state)

    # -- resume ------------------------------------------------------------

    def resume(self, workflow_id: str, *, allow_provider_change: bool = False) -> EvolutionOutcome:
        """Continue a paused run from its checkpoint.

        Refuses by default when the configured provider differs from the one
        that produced the run's assessments. Continuing would leave the run
        holding judgements from two models with no way to tell which said what
        — exactly the silent provenance ambiguity the brief forbids. The caller
        may override explicitly, and the change is then recorded as a warning.
        """
        run = self.store.get_workflow(workflow_id)
        if run is None:
            raise KeyError(f"unknown workflow {workflow_id!r}")

        extra_warnings: list[str] = []
        if run.provider_changed(self.provider_id, self.model_id):
            message = (
                f"workflow was assessed by {run.provider_id}/{run.model_id} but the "
                f"configured provider is {self.provider_id}/{self.model_id}. Resuming "
                f"would mix judgements from two models."
            )
            if not allow_provider_change:
                raise ProviderMismatch(message)
            extra_warnings.append(message)
            log.warning("workflow_provider_changed", workflow=workflow_id[:12])

        app = build_graph(self._context(run), checkpointer=self._checkpointer())
        from langgraph.types import Command

        return self._run(
            app,
            run,
            Command(resume={"resumed_at": text_hash(workflow_id)[:8]}),
            extra_warnings=extra_warnings,
        )

    # -- shared ------------------------------------------------------------

    def _run(
        self,
        app: Any,
        run: WorkflowRun,
        payload: Any,
        *,
        extra_warnings: list[str] | None = None,
    ) -> EvolutionOutcome:
        config = {"configurable": {"thread_id": run.id}}
        result = app.invoke(payload, config)

        interrupts = result.get("__interrupt__") or []
        state = {k: v for k, v in result.items() if not k.startswith("__")}

        if interrupts:
            state["status"] = WorkflowStatus.WAITING_FOR_REVIEW.value
        if extra_warnings:
            # Merged into the state's own warnings rather than set on the run
            # beforehand, because `to_run` rebuilds warnings from state and
            # would otherwise drop them.
            state["warnings"] = [*state.get("warnings", []), *extra_warnings]
        updated = to_run(state, run)
        self.store.put_workflow(updated)

        payload_value = None
        if interrupts:
            raw = interrupts[0]
            payload_value = raw.value if hasattr(raw, "value") else raw

        log.info(
            "workflow_run",
            workflow=run.id[:12],
            status=updated.status.value,
            llm_calls=updated.llm_calls,
            interrupted=bool(interrupts),
        )
        return EvolutionOutcome(
            run=updated, interrupted=bool(interrupts), interrupt_payload=payload_value
        )

    # -- inspection --------------------------------------------------------

    def get(self, workflow_id: str) -> WorkflowRun | None:
        run = self.store.get_workflow(workflow_id)
        if run is not None:
            return run
        matches = self.store.find_workflow(workflow_id)
        return matches[0] if len(matches) == 1 else None

    def list(self, *, status: WorkflowStatus | None = None, limit: int = 20) -> list[WorkflowRun]:
        return self.store.list_workflows(status=status, limit=limit)

    def explain(self, workflow_id: str) -> dict[str, Any] | None:
        """Everything behind one run — the "why did Forge propose this?" answer."""
        run = self.get(workflow_id)
        if run is None:
            return None

        source = self.store.get_source(run.source_id)
        payload = run.to_dict(verbose=True)
        payload["source"] = (
            {"id": source.id, "locator": source.locator, "kind": source.kind.value}
            if source
            else None
        )
        payload["proposals"] = [
            {
                "id": p.id,
                "type": p.type.value,
                "status": p.status.value,
                "safety": p.safety.value,
                "target": p.operation.target,
                "before": p.operation.before,
                "after": p.operation.after,
                "reason": p.reason,
                "decided_by": p.decided_by,
            }
            for p in (self.store.get_proposal(i) for i in run.proposal_ids)
            if p is not None
        ]
        payload["evidence"] = [
            {"span_id": span.id, "citation": span.citation(), "text": span.text[:200]}
            for span in (self.store.get_span(i) for i in run.evidence_span_ids[:5])
            if span is not None
        ]
        return payload


def build_service(
    store: SqliteStore,
    settings: Settings,
    *,
    require_semantic: bool = True,
) -> EvolutionService:
    """Construct a service with the configured provider resolved.

    When ``require_semantic`` is false, an unavailable provider yields a
    service with no provider rather than an exception — the workflow then
    reports ``SEMANTIC_ANALYSIS_UNAVAILABLE`` and remains resumable, which is
    the honest outcome and the one the brief specifies.
    """
    from ..llm import get_provider, provider_identity

    try:
        provider = get_provider(settings)
        reachable, detail = provider.health()
        if not reachable:
            raise ProviderUnavailable(detail)
        provider_id, model_id = provider_identity(provider, "analysis")
    except (ProviderUnavailable, Exception) as exc:
        if require_semantic and isinstance(exc, ProviderUnavailable):
            raise
        if not isinstance(exc, ProviderUnavailable):
            raise
        log.warning("semantic_provider_unavailable", error=str(exc)[:160])
        return EvolutionService(store, settings)

    return EvolutionService(
        store, settings, provider=provider, provider_id=provider_id, model_id=model_id
    )
