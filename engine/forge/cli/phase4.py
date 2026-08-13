"""Phase 4 CLI: knowledge evolution.

Two commands and one sub-app, following the existing conventions: `--json`
everywhere, non-zero exit on failure, abbreviated ids accepted, and nothing
that writes to the vault.

Approval deliberately stays with `forge proposals approve`. There is one review
queue and one approval mechanism; adding a second here would mean two places
that can change knowledge and two things to audit.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any, Optional

import typer

from ..domain import WorkflowStatus
from ..evolution.service import ProviderMismatch, build_service
from ..evolution.workflow import OrchestratorUnavailable
from ..llm.base import ProviderUnavailable
from ..storage.sqlite_store import SqliteStore

workflow_app = typer.Typer(
    no_args_is_help=True, help="Inspect and resume knowledge-evolution workflows."
)


def _emit(payload: Any, as_json: bool) -> bool:
    if as_json:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
        return True
    return False


def _service(settings: Any, store: SqliteStore, *, require_semantic: bool = False):
    return build_service(store, settings, require_semantic=require_semantic)


def _resolve_source(store: SqliteStore, token: str) -> str | None:
    """Accept a source id, a vault-relative locator, or a filesystem path."""
    if store.get_source(token) is not None:
        return token
    if (source := store.get_source_by_locator(token)) is not None:
        return source.id
    needle = Path(token).as_posix()
    for source in store.list_sources():
        if source.locator.endswith(needle) or Path(source.locator).name == Path(needle).name:
            return source.id
    return None


def register(app: typer.Typer, settings_factory: Any) -> None:
    """Attach Phase 4 commands."""

    @app.command()
    def evolve(
        source: str = typer.Argument(..., help="Ingested source: id, locator, or path."),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Evaluate how a source's evidence affects existing knowledge.

        Runs the evolution workflow. It pauses for human review whenever it has
        produced a proposal, and nothing is applied until you approve it.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        source_id = _resolve_source(store, source)
        if source_id is None:
            typer.echo(
                f"no ingested source matches {source!r}. Ingest it first: forge ingest {source}",
                err=True,
            )
            store.close()
            raise typer.Exit(code=1)

        try:
            service = _service(settings, store)
            outcome = service.start(source_id)
        except OrchestratorUnavailable as exc:
            typer.echo(str(exc), err=True)
            store.close()
            raise typer.Exit(code=3)
        except ProviderUnavailable as exc:
            typer.echo(f"semantic provider unavailable: {exc}", err=True)
            store.close()
            raise typer.Exit(code=2)

        if _emit(outcome.to_dict(verbose=True), json_out):
            service.close()
            store.close()
            return

        _print_outcome(outcome, store)
        service.close()
        store.close()
        if outcome.status is WorkflowStatus.FAILED:
            raise typer.Exit(code=1)

    app.add_typer(workflow_app, name="workflow")

    @workflow_app.command("list")
    def workflow_list(
        vault: Optional[Path] = typer.Option(None),
        status: Optional[str] = typer.Option(None, help="Filter by workflow status."),
        limit: int = typer.Option(20),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List knowledge-evolution runs."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        try:
            wanted = WorkflowStatus(status) if status else None
        except ValueError:
            typer.echo(f"unknown status {status!r}", err=True)
            store.close()
            raise typer.Exit(code=2)

        runs = store.list_workflows(status=wanted, limit=limit)
        payload = {"workflows": [r.to_dict() for r in runs], "counts": store.count_workflows()}
        if not _emit(payload, json_out):
            if not runs:
                typer.echo("no workflows yet — run `forge evolve <source>`")
            for run in runs:
                source = store.get_source(run.source_id)
                typer.echo(
                    f"  {run.id[:12]}  {run.status.value:<28} "
                    f"{len(run.proposal_ids)} proposal(s)  "
                    f"{source.locator if source else run.source_id[:12]}"
                )
        store.close()

    @workflow_app.command("status")
    def workflow_status(
        workflow_id: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Show one run's current state."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = _service(settings, store)

        run = service.get(workflow_id)
        if run is None:
            typer.echo(f"no workflow {workflow_id!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        if not _emit(run.to_dict(), json_out):
            typer.echo(f"workflow  : {run.id}")
            typer.echo(f"status    : {run.status.value}")
            typer.echo(f"provider  : {run.provider_id} / {run.model_id}")
            typer.echo(f"impact    : {run.impact.value if run.impact else '-'}")
            typer.echo(f"counts    : {run.to_dict()['counts']}")
            typer.echo(f"llm calls : {run.llm_calls}  (cache hits {run.cache_hits})")
            if run.awaiting_review:
                typer.echo("\nawaiting human review. Decide the proposals, then:")
                typer.echo(f"  forge workflow resume {run.id[:12]}")
            for warning in run.warnings:
                typer.echo(f"warning   : {warning}")
            for error in run.errors:
                typer.echo(f"error     : {error}")
        store.close()

    @workflow_app.command("inspect")
    def workflow_inspect(
        workflow_id: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Explain a run: candidates, assessments, proposals, revisions.

        This is the "why did Forge propose this?" command.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = _service(settings, store)

        detail = service.explain(workflow_id)
        if detail is None:
            typer.echo(f"no workflow {workflow_id!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        if _emit(detail, json_out):
            store.close()
            return

        typer.echo(f"workflow  : {detail['id']}  [{detail['status']}]")
        typer.echo(
            f"source    : {(detail.get('source') or {}).get('locator', '?')}"
        )
        typer.echo(
            f"assessed  : {detail['provider_id']} / {detail['model_id']} "
            f"(prompt {detail['prompt_version']}, schema {detail['schema_version']})"
        )
        typer.echo(f"impact    : {detail['impact'] or '-'}")

        typer.echo("\nconcepts considered, and why:")
        for candidate in detail.get("candidates_detail", []):
            typer.echo(f"  {candidate['concept']:<40} [{candidate['selector']}] {candidate['detail']}")

        typer.echo("\nassessments:")
        for assessment in detail.get("assessments_detail", []):
            claim = store.get_claim(assessment["claim_id"])
            typer.echo(
                f"  {assessment['classification']:<22} "
                f"{(claim.statement if claim else assessment['claim_id'])[:60]}"
            )
            typer.echo(f"      {assessment['rationale'][:110]}")
            typer.echo(
                f"      evidence: {', '.join(assessment['evidence_span_ids'][:2])}"
                f"{'  (cached)' if assessment['cached'] else ''}"
            )

        if detail["proposals"]:
            typer.echo("\nproposals:")
            for proposal in detail["proposals"]:
                typer.echo(
                    f"  {proposal['id'][:12]}  {proposal['type']:<18} "
                    f"[{proposal['status']}] {proposal['safety']}"
                )
                if proposal["after"]:
                    typer.echo(f"      before: {(proposal['before'] or '')[:80]}")
                    typer.echo(f"      after : {proposal['after'][:80]}")

        if detail["revision_ids"]:
            typer.echo(f"\nrevisions : {len(detail['revision_ids'])} recorded")
        typer.echo(f"\nnodes     : {' -> '.join(n['node'] for n in detail.get('nodes', []))}")
        typer.echo(
            f"cost      : {detail['llm_calls']} llm call(s), "
            f"{detail['cache_hits']} cache hit(s), {detail['duration_ms']}ms"
        )
        store.close()

    @workflow_app.command("resume")
    def workflow_resume(
        workflow_id: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        allow_provider_change: bool = typer.Option(
            False,
            "--allow-provider-change",
            help="Continue even though a different model would now be used.",
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Resume a paused run after deciding its proposals."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        service = _service(settings, store)
        run = service.get(workflow_id)
        if run is None:
            typer.echo(f"no workflow {workflow_id!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        try:
            outcome = service.resume(run.id, allow_provider_change=allow_provider_change)
        except ProviderMismatch as exc:
            typer.echo(f"{exc}\n\nRe-run with --allow-provider-change if that is intended.", err=True)
            store.close()
            raise typer.Exit(code=2)
        except OrchestratorUnavailable as exc:
            typer.echo(str(exc), err=True)
            store.close()
            raise typer.Exit(code=3)

        if not _emit(outcome.to_dict(verbose=True), json_out):
            _print_outcome(outcome, store)
        service.close()
        store.close()
        if outcome.status is WorkflowStatus.FAILED:
            raise typer.Exit(code=1)


def _print_outcome(outcome: Any, store: SqliteStore) -> None:
    run = outcome.run
    source = store.get_source(run.source_id)
    counts = run.to_dict()["counts"]

    typer.echo("Forge Knowledge Evolution")
    typer.echo("──────────────────────────────")
    typer.echo(f"\nSource:\n  {source.locator if source else run.source_id}")

    typer.echo("\nConcepts affected:")
    if not run.candidates:
        typer.echo("  (none — this evidence does not touch existing knowledge)")
    for candidate in run.candidates:
        typer.echo(f"  ✓ {candidate.concept_name}   [{candidate.selector}]")

    typer.echo(f"\nClaims examined:\n  {counts['claims_examined']}")

    if run.assessments:
        typer.echo("\nSemantic assessments:")
        for label, count in run.by_classification().items():
            typer.echo(f"  {label.replace('_', ' ').title()}: {count}")

    typer.echo(f"\nProposals:\n  {counts['proposals']}")
    if run.activated_entity_ids:
        typer.echo(f"\nApplied:\n  {counts['activated']} change(s), {counts['revisions']} revision(s)")

    typer.echo(f"\nStatus:\n  {run.status.value.upper()}")
    typer.echo(f"\nWorkflow:\n  {run.id}")

    for warning in run.warnings:
        typer.echo(f"\nwarning: {warning}")
    for error in run.errors:
        typer.echo(f"\nerror: {error}")

    if run.awaiting_review:
        typer.echo("\nWaiting for review. Decide the proposals, then resume:")
        typer.echo("  forge proposals list --status pending")
        typer.echo(f"  forge workflow resume {run.id[:12]}")
    else:
        typer.echo("\nUse:")
        typer.echo(f"  forge workflow inspect {run.id[:12]}")
