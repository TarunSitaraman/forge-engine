"""Phase 9 CLI: the six vision questions, from a terminal.

Following the existing conventions: `--json` everywhere, non-zero exit on
failure, and nothing that writes to the vault.

**`question add` is the only command here that writes**, and it writes a
`Question`, which is by definition something the user asked. Everything else
reads. Gap findings in particular are reported and never acted on: whether a
gap matters is a judgement, and Forge does not make those.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any

import typer

from ..domain import (
    Derivation,
    GapKind,
    Provenance,
    ProvenanceTier,
    Question,
    QuestionStatus,
)
from ..research import belief_for_concept, changes_since, gap_report
from ..storage.sqlite_store import SqliteStore

question_app = typer.Typer(no_args_is_help=True, help="Record and review research questions.")

#: What a question asked from the CLI records as its author. A question is a
#: USER_ASSERTION by the domain's own rule, so the agent is the interface the
#: human used, never a model.
CLI_AGENT = "forge-cli"

#: Built once at import rather than inside an argument default, where it is a
#: call evaluated at definition time.
GAP_KIND_HELP = f"Repeatable. One of: {', '.join(k.value for k in GapKind)}"


def _emit(payload: Any, as_json: bool) -> bool:
    if as_json:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
        return True
    return False


def _resolve_concept(store: SqliteStore, needle: str) -> str | None:
    """Accept an id, a full name, or an unambiguous prefix of either.

    Typing a 26-character id by hand is not a thing anyone does, and the other
    phases already accept abbreviations.
    """
    if store.get_concept(needle) is not None:
        return needle
    concepts = list(store.list_concepts())
    lowered = needle.casefold()
    exact = [c for c in concepts if c.canonical_name.casefold() == lowered]
    if len(exact) == 1:
        return exact[0].id
    partial = [
        c
        for c in concepts
        if c.id.startswith(needle) or lowered in c.canonical_name.casefold()
    ]
    return partial[0].id if len(partial) == 1 else None


def register(app: typer.Typer, settings_factory: Any) -> None:
    app.add_typer(question_app, name="question")

    @question_app.command("add")
    def question_add(
        text: str = typer.Argument(..., help="The question, in your own words."),
        vault: Path | None = typer.Option(None),
        concept: list[str] = typer.Option(
            [], "--concept", help="Concept this is about. Repeatable; scopes retrieval."
        ),
        tag: list[str] = typer.Option([], "--tag", help="Repeatable."),
        note: str | None = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Record a research question.

        Stored as a USER_ASSERTION, because you asked it. Forge never invents
        questions: a model may suggest one, but a human asking it is what makes
        it a question the gap report measures against.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        concept_ids: list[str] = []
        for needle in concept:
            resolved = _resolve_concept(store, needle)
            if resolved is None:
                typer.echo(f"no single concept matches {needle!r}", err=True)
                store.close()
                raise typer.Exit(code=2)
            concept_ids.append(resolved)

        question = Question(
            id=Question.make_id(text),
            text=text,
            provenance=Provenance(
                tier=ProvenanceTier.USER_ASSERTION,
                derivation=Derivation.HUMAN,
                agent=CLI_AGENT,
            ),
            tags=tuple(tag),
            note=note,
            concept_ids=tuple(concept_ids),
        )
        existing = store.get_question(question.id)
        store.put_question(question)
        store.close()

        payload = {
            "id": question.id,
            "text": question.text,
            "status": question.status.value,
            "concept_ids": concept_ids,
            "already_recorded": existing is not None,
        }
        if not _emit(payload, json_out):
            verb = "already recorded" if existing else "recorded"
            typer.echo(f"{verb}: {question.text}")
            typer.echo(f"  id      : {question.id}")
            if concept_ids:
                typer.echo(f"  concepts: {', '.join(concept_ids)}")

    @question_app.command("list")
    def question_list(
        vault: Path | None = typer.Option(None),
        status: str | None = typer.Option(None, help="open | partially_answered | answered"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List recorded questions and how many claims answer each."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        try:
            wanted = QuestionStatus(status) if status else None
        except ValueError:
            typer.echo(
                f"unknown status {status!r}; expected one of "
                f"{', '.join(s.value for s in QuestionStatus)}",
                err=True,
            )
            store.close()
            raise typer.Exit(code=2) from None

        rows = [
            {
                "id": q.id,
                "text": q.text,
                "status": q.status.value,
                "answers": len(store.answers_for_question(q.id)),
                "created_at": q.created_at.isoformat(),
            }
            for q in store.list_questions(wanted)
        ]
        store.close()
        if not _emit(rows, json_out):
            if not rows:
                typer.echo("no questions recorded. `forge question add \"...\"`")
                return
            for row in rows:
                mark = "open" if row["answers"] == 0 else f"{row['answers']} answer(s)"
                typer.echo(f"[{row['status']:<19}] {row['text']}")
                typer.echo(f"    {row['id']}  {mark}")

    @app.command()
    def gaps(
        vault: Path | None = typer.Option(None),
        kind: list[str] = typer.Option([], "--kind", help=GAP_KIND_HELP),
        limit: int = typer.Option(20),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """What the knowledge model does not hold. Deterministic, zero model calls.

        Every finding is a structural fact about the graph. Whether one matters
        is your call: nothing here is acted on.

        A kind that describes the whole corpus is summarised in one line rather
        than listed. Pass `--kind` to list it anyway.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        try:
            kinds = [GapKind(k) for k in kind] or None
        except ValueError as exc:
            typer.echo(f"{exc}", err=True)
            store.close()
            raise typer.Exit(code=2) from None

        report = gap_report(store, kinds=kinds, limit=limit)
        store.close()

        if _emit(report.to_dict(), json_out):
            return
        typer.echo(f"{report.total} finding(s), showing {report.returned}\n")
        for summary in report.saturated:
            typer.echo(
                f"  {summary['count']:>5} of {summary['population']}  {summary['kind']}"
                f"\n         corpus-wide, not a finding about any one subject;"
                f" --kind {summary['kind']} to list them\n"
            )
        for gap in report.gaps:
            typer.echo(f"  [{gap.weight:.2f}] {gap.kind.value}")
            typer.echo(f"         {gap.subject_label}")
            typer.echo(f"         {gap.detail}")
        if not report.gaps and not report.saturated:
            typer.echo("  nothing found. An empty store also produces no gaps.")

    @app.command()
    def changes(
        vault: Path | None = typer.Option(None),
        days: int = typer.Option(30, help="Window, in days back from now."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """What changed in your understanding, read from the revision log."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        report = changes_since(store, days=days)
        store.close()

        if _emit(report.to_dict(), json_out):
            return
        typer.echo(f"{report.total} change(s) in the last {days} day(s)")
        if report.truncated:
            typer.echo("  (scan hit its ceiling; these counts are a floor)")
        for entity, ops in sorted(report.by_entity.items()):
            detail = ", ".join(f"{op} {n}" for op, n in sorted(ops.items()))
            typer.echo(f"  {entity:<12} {detail}")
        for label, ids in (
            ("claims created", report.claims_created),
            ("claims superseded", report.claims_superseded),
            ("claims disputed", report.claims_disputed),
            ("syntheses staled", report.syntheses_staled),
        ):
            if ids:
                typer.echo(f"\n  {label}: {len(ids)}")
        if report.causes:
            typer.echo(f"\n  causes: {', '.join(report.causes[:5])}")

    @app.command()
    def belief(
        concept: str = typer.Argument(..., help="Concept id, name, or unambiguous prefix."),
        vault: Path | None = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """What you currently believe about a concept, and what disagrees.

        There is no confidence score and there will not be one: a number a
        model emits about its own certainty is not a measurement. What you get
        instead is every held claim with how it was derived.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        concept_id = _resolve_concept(store, concept)
        if concept_id is None:
            typer.echo(f"no single concept matches {concept!r}", err=True)
            store.close()
            raise typer.Exit(code=2)
        result = belief_for_concept(store, concept_id)
        store.close()
        if result is None:  # pragma: no cover - resolved above
            raise typer.Exit(code=2)

        if _emit(result.to_dict(), json_out):
            return
        typer.echo(f"{result.concept_name}\n")
        if not result.held:
            typer.echo("  nothing is claimed about this concept yet.")
        for claim in result.held:
            typer.echo(f"  held      [{claim.provenance.tier.value}] {claim.statement}")
        for claim in result.disputed:
            typer.echo(f"  disputed  {claim.statement}")
        for claim in result.superseded:
            typer.echo(f"  was held  {claim.statement}")
        if result.supporting_sources:
            typer.echo(f"\n  sources: {', '.join(result.supporting_sources)}")
        if result.unevidenced:
            typer.echo(f"\n  WITHOUT EVIDENCE: {len(result.unevidenced)} held claim(s)")
        if result.dissent:
            typer.echo(f"\n  dissent ({len(result.dissent)}):")
            for item in result.dissent:
                typer.echo(f"    [{item.kind}] {item.detail}")
        elif result.held:
            typer.echo("\n  nothing outstanding against this.")
