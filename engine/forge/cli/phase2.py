"""Phase 2 CLI commands: ingest, search, concepts, documents, proposals.

Registered onto the Phase 1 app in :mod:`forge.cli.main`, so Phase 1 commands
keep working unchanged.

Every command is scriptable: ``--json`` on all of them, and non-zero exit codes
on failure.
"""

from __future__ import annotations

import json as jsonlib
import random
from pathlib import Path
from typing import Any, Optional

import typer

from ..config import Settings
from ..domain import IngestionStatus, ProposalStatus, ProposalType, SafetyClass
from ..embeddings import NullEmbeddingProvider, OllamaEmbeddingProvider
from ..extraction import CandidateExtractor
from ..ingestion import IngestionPipeline, IngestOptions
from ..llm import CALLS, get_provider
from ..proposals import ProposalApplier, ProposalService, audit as audit_grounding
from ..retrieval import SearchQuery, SearchService
from ..storage.sqlite_store import SqliteStore

proposals_app = typer.Typer(
    no_args_is_help=True, help="Review and decide proposed changes. Nothing is applied by default."
)


def _emit(payload: Any, as_json: bool) -> bool:
    if as_json:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
        return True
    return False


def _enum_option(enum_cls: Any, value: str, flag: str) -> Any:
    """Parse a CLI string into an enum member, or exit 2 with the valid set.

    Case-insensitive on purpose: the enum's *names* are upper case and its
    values lower case, so ``--status PENDING`` is the natural thing to type
    and used to raise a raw ``ValueError`` traceback from deep inside
    ``enum``. A wrong filter is a user error, not a crash.
    """
    try:
        return enum_cls(value.strip().lower())
    except ValueError:
        allowed = "|".join(member.value for member in enum_cls)
        typer.echo(f"unknown {flag} {value!r}; expected one of: {allowed}", err=True)
        raise typer.Exit(code=2)


def register(app: typer.Typer, settings_factory: Any) -> None:
    """Attach Phase 2 commands to the main app."""

    @app.command()
    def ingest(
        path: Path = typer.Argument(..., help="File or directory to ingest (.md, .pdf)."),
        vault: Optional[Path] = typer.Option(None),
        extract: bool = typer.Option(
            False, "--extract", help="Run LLM extraction (needs a local model)."
        ),
        force: bool = typer.Option(False, "--force", help="Re-process even if unchanged."),
        embed: bool = typer.Option(False, "--embed", help="Store embeddings if available."),
        max_spans: int = typer.Option(12, help="Max spans sent to the model per document."),
        json_out: bool = typer.Option(False, "--json"),
        verbose: bool = typer.Option(False, "-v"),
    ) -> None:
        """Ingest a PDF or Markdown file into the knowledge model."""
        settings = settings_factory(vault, "INFO" if verbose else "WARNING")
        store = SqliteStore(settings.db_path)
        store.initialize()

        extractor = None
        if extract:
            try:
                extractor = CandidateExtractor(get_provider(settings), max_spans=max_spans)
            except Exception as exc:
                typer.echo(f"could not construct LLM provider: {exc}", err=True)
                raise typer.Exit(code=2)

        CALLS.reset()
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        report = pipeline.ingest_path(
            path, IngestOptions(extract=extract, force=force, max_spans=max_spans)
        )

        if embed:
            _index_embeddings(settings, store, report)

        if _emit(report.to_dict(), json_out):
            store.close()
            return

        for source in report.sources:
            _print_source(source)
        totals = report.totals()
        typer.echo(
            f"\n{totals['sources']} source(s) in {report.duration_seconds:.2f}s | "
            f"{totals['spans']} spans | {totals['concepts_proposed']} concepts | "
            f"{totals['claims_proposed']} claims | {totals['proposals_created']} proposals"
        )
        typer.echo(f"LLM calls: {totals['llm_calls']}  cache: {report.cache.to_dict()}")

        if report.aborted:
            typer.echo(
                f"\nRUN ABORTED after {len(report.sources)} of {report.attempted_targets} "
                f"source(s): {report.abort_reason}",
                err=True,
            )
            typer.echo(
                "The provider rejected the request, so every remaining call would fail\n"
                "identically. Nothing was cached. Check the credential named in\n"
                "FORGE_CLOUD_PRESET's api_key_env (`forge status` shows which), then re-run —\n"
                "completed sources are served from cache.",
                err=True,
            )
            store.close()
            raise typer.Exit(code=2)
        if not extract:
            typer.echo("(deterministic ingestion only — pass --extract for concept/claim candidates)")
        store.close()

        if any(s.status in (IngestionStatus.PARSE_FAILED, IngestionStatus.NOT_FOUND) for s in report.sources):
            raise typer.Exit(code=1)

    @app.command()
    def search(
        query: str = typer.Argument(..., help="Lexical query."),
        vault: Optional[Path] = typer.Option(None),
        limit: int = typer.Option(10),
        source: Optional[str] = typer.Option(None, help="Filter by source path substring."),
        kind: Optional[str] = typer.Option(None, help="Filter by source kind (pdf, markdown)."),
        page: Optional[int] = typer.Option(None, help="Filter by page number."),
        heading: Optional[str] = typer.Option(None, help="Filter by heading path substring."),
        semantic: bool = typer.Option(False, "--semantic", help="Re-rank with embeddings if available."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Search spans. Returns evidence with provenance — not generated prose."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = SearchService(store, embeddings=_embeddings(settings, enabled=semantic))

        hits = service.search(
            SearchQuery(
                text=query,
                limit=limit,
                source_contains=source,
                source_kinds={kind} if kind else set(),
                page=page,
                heading_contains=heading,
                semantic=semantic,
            )
        )
        note = service.degradation_note()

        if _emit(
            {"query": query, "hits": [h.to_dict() for h in hits], "degradation": note}, json_out
        ):
            store.close()
            return

        if semantic and note:
            typer.echo(f"note: {note}\n")
        if not hits:
            typer.echo("no matches")
        for hit in hits:
            typer.echo(f"{hit.score:8.3f}  {hit.citation}")
            typer.echo(f"          {_excerpt(hit.span.text)}")
        store.close()

    @app.command()
    def concepts(
        query: str = typer.Argument("", help="Filter by name or alias."),
        vault: Optional[Path] = typer.Option(None),
        limit: int = typer.Option(50),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List concepts in the knowledge model."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        found = SearchService(store).concepts(query, limit=limit)

        if _emit([c.model_dump(mode="json") for c in found], json_out):
            store.close()
            return
        if not found:
            typer.echo(
                "no concepts stored.\n"
                "Phase 2 proposes concepts rather than creating them — "
                "see `forge proposals list --type new_concept`."
            )
        for concept in found:
            typer.echo(f"{concept.canonical_name}  [{concept.kind.value}]  {concept.vault_path or ''}")
        store.close()

    @app.command()
    def documents(
        query: str = typer.Argument("", help="Filter by path or title."),
        vault: Optional[Path] = typer.Option(None),
        limit: int = typer.Option(50),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List ingested documents and their sources."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        rows = SearchService(store).documents(query, limit=limit)

        payload = [
            {
                "document_id": d.id,
                "source": s.locator if s else None,
                "kind": s.kind.value if s else None,
                "trust_tier": s.trust_tier.value if s else None,
                "title": s.title if s else None,
                "version": d.version,
                "parser": f"{d.parser}@{d.parser_version}",
                "spans": len(store.spans_for_document(d.id)),
            }
            for d, s in rows
        ]
        if _emit(payload, json_out):
            store.close()
            return
        for row in payload:
            typer.echo(
                f"{row['spans']:>4} spans  [{row['kind']}]  {row['source']}  "
                f"({row['trust_tier']})"
            )
        store.close()

    app.add_typer(proposals_app, name="proposals")

    # -- proposals sub-app -------------------------------------------------

    @proposals_app.command("list")
    def proposals_list(
        vault: Optional[Path] = typer.Option(None),
        status: Optional[str] = typer.Option(
            "pending",
            help="pending|approved|activated|rejected|superseded|all (case-insensitive)",
        ),
        type_: Optional[str] = typer.Option(
            None,
            "--type",
            help=(
                "metadata_repair|new_concept|concept_match|new_claim|"
                "claim_evidence|claim_refinement|claim_conflict"
            ),
        ),
        limit: int = typer.Option(20),
        sample: int = typer.Option(
            0,
            help="Show a random sample of this many instead of the first --limit.",
        ),
        seed: int = typer.Option(0, help="Seed for --sample, so a sample is reproducible."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List proposals awaiting or holding a decision.

        `--sample N` draws randomly rather than taking the first N. That matters
        for judging output quality: proposals come back grouped by source, so
        the first N describe one or two documents and tell you nothing about
        the rest. The draw is seeded, so quoting a sample is reproducible.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = ProposalService(store)

        found = service.list(
            status=(
                None
                if status is None or status.strip().lower() == "all"
                else _enum_option(ProposalStatus, status, "--status")
            ),
            type=_enum_option(ProposalType, type_, "--type") if type_ else None,
            limit=max(limit, 100_000) if sample else limit,
        )
        if sample:
            population = list(found)
            found = random.Random(seed).sample(population, min(sample, len(population)))
            typer.echo(f"random sample of {len(found)} from {len(population)} (seed {seed})\n")
        counts = service.counts()

        if _emit(
            {"counts": counts, "proposals": [p.model_dump(mode="json") for p in found]}, json_out
        ):
            store.close()
            return

        typer.echo(f"counts: {counts or '{}'}\n")
        for proposal in found:
            typer.echo(
                f"{proposal.id[:12]}  {proposal.status.value:<9} {proposal.type.value:<16} "
                f"{proposal.safety.value:<24} {proposal.operation.target}"
            )
        if found:
            typer.echo(f"\n{len(found)} shown. `forge proposals show <id>` for detail.")
        store.close()

    @proposals_app.command("audit-grounding")
    def proposals_audit_grounding(
        vault: Optional[Path] = typer.Option(None),
        show_passing: bool = typer.Option(False, "--show-passing", help="Print every row."),
        reject: bool = typer.Option(
            False, "--reject", help="Reject every pending proposal that fails the check."
        ),
        dry_run: bool = typer.Option(True, help="With --reject, preview without deciding."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Re-check stored evidence quotes against the current grounding rule.

        Zero model calls: grounding is a deterministic string check, so a rule
        change applies retroactively to an already-extracted corpus for free.

        Exits 1 if any stored quote fails. Those were admitted under an older,
        looser rule and would be dropped by extraction today — reject them
        rather than approving them.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        try:
            checks = audit_grounding(store)
            failed = [c for c in checks if not c.grounded]

            rejected: list[str] = []
            if reject and not dry_run:
                service = ProposalService(store)
                # Only PENDING can be rejected; anything already decided is
                # someone's decision and is not this command's to overturn.
                for check in failed:
                    if check.status != ProposalStatus.PENDING.value:
                        continue
                    if check.proposal_id in rejected:
                        continue
                    service.reject(
                        check.proposal_id,
                        by="audit-grounding",
                        note="evidence quote is not verbatim in the cited span",
                    )
                    rejected.append(check.proposal_id)
        finally:
            store.close()

        if _emit(
            {
                "checked": len(checks),
                "ungrounded": [c.to_dict() for c in failed],
                "rejected": rejected,
            },
            json_out,
        ):
            raise typer.Exit(code=1 if failed else 0)

        if not checks:
            typer.echo("no proposals carry an evidence quote — nothing to audit")
            return

        for check in checks:
            if check.grounded and not show_passing:
                continue
            mark = "ok  " if check.grounded else "FAIL"
            typer.echo(
                f"{mark} {check.proposal_id[:12]}  {check.status:<9} overlap={check.overlap:.3f}"
            )
            typer.echo(f"     quote: {check.quote[:100]!r}")
            if check.note:
                typer.echo(f"     note : {check.note}")

        rate = len(failed) / len(checks) if checks else 0.0
        typer.echo(
            f"\n{len(checks)} quote(s) checked, {len(failed)} ungrounded ({rate:.2%})."
        )
        if failed:
            if rejected:
                typer.echo(f"{len(rejected)} pending proposal(s) rejected.")
            elif reject:
                pending = {c.proposal_id for c in failed if c.status == ProposalStatus.PENDING.value}
                typer.echo(
                    f"\ndry run — would reject {len(pending)} pending proposal(s).\n"
                    "Re-run with --no-dry-run to apply."
                )
            else:
                typer.echo(
                    "\nThese were admitted under the pre-2026-08-19 bag-of-words rule and\n"
                    "would be dropped by extraction today. Reject rather than approve:\n"
                    "  forge proposals audit-grounding --reject --no-dry-run"
                )
            raise typer.Exit(code=1)

    @proposals_app.command("show")
    def proposals_show(
        proposal_id: str = typer.Argument(..., help="Full or abbreviated proposal id."),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Show one proposal in full, including evidence and affected links."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        proposal, ambiguous = ProposalService(store).resolve(proposal_id)

        if proposal is None:
            _report_unresolved(proposal_id, ambiguous)
            store.close()
            raise typer.Exit(code=1)

        if _emit(proposal.model_dump(mode="json"), json_out):
            store.close()
            return

        typer.echo(f"id       : {proposal.id}")
        typer.echo(f"type     : {proposal.type.value}")
        typer.echo(f"status   : {proposal.status.value}")
        typer.echo(f"safety   : {proposal.safety.value}")
        typer.echo(f"target   : {proposal.operation.target}")
        typer.echo(f"action   : {proposal.operation.action}")
        typer.echo(f"reason   : {proposal.reason}")
        typer.echo(
            f"origin   : {proposal.provenance.derivation.value} via {proposal.provenance.agent}"
            + (f" ({proposal.provenance.model_id})" if proposal.provenance.model_id else "")
        )
        if proposal.operation.before is not None:
            typer.echo("\nchange:")
            typer.echo(f"  - {proposal.operation.before}")
            typer.echo(f"  + {proposal.operation.after}")
        if proposal.evidence_span_ids:
            typer.echo("\nevidence:")
            service = SearchService(store)
            for span_id in proposal.evidence_span_ids:
                hit = service.span(span_id)
                if hit:
                    typer.echo(f"  {hit.citation}")
                    typer.echo(f"    {_excerpt(hit.span.text)}")
        details = proposal.operation.details
        if details.get("affected_links"):
            typer.echo(f"\naffected links: {details['affected_links']}")
        if details.get("candidates"):
            typer.echo("\ncandidates (no automatic selection):")
            for candidate in details["candidates"]:
                typer.echo(
                    f"  {candidate.get('canonical_name')}  "
                    f"[{candidate.get('signal')} {candidate.get('score')}]  "
                    f"{candidate.get('vault_path') or ''}"
                )
        if proposal.is_decided:
            typer.echo(f"\ndecided  : {proposal.decided_at} by {proposal.decided_by}")
            if proposal.decision_note:
                typer.echo(f"note     : {proposal.decision_note}")
        store.close()

    @proposals_app.command("approve")
    def proposals_approve(
        proposal_id: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        note: Optional[str] = typer.Option(None),
        apply: bool = typer.Option(
            False,
            "--apply",
            help="Also write the change to the vault. Off by default; makes a backup and records a revision.",
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Approve a proposal. Recording a decision does not apply it."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = ProposalService(store)

        try:
            decided = service.approve(proposal_id, note=note)
        except KeyError as exc:
            typer.echo(str(exc), err=True)
            store.close()
            raise typer.Exit(code=1)

        applier = ProposalApplier(settings.vault_path, store, settings.state_dir / "backups")
        apply_report = applier.apply([decided], apply=apply)

        if _emit(
            {"proposal": decided.model_dump(mode="json"), "apply": apply_report.to_dict()}, json_out
        ):
            store.close()
            return

        typer.echo(f"approved {decided.id[:12]} ({decided.type.value})")
        _print_apply(apply_report, applied_flag=apply)
        store.close()

    @proposals_app.command("reject")
    def proposals_reject(
        proposal_id: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        note: Optional[str] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Reject a proposal. It will not be regenerated on the next run."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        try:
            decided = ProposalService(store).reject(proposal_id, note=note)
        except KeyError as exc:
            typer.echo(str(exc), err=True)
            store.close()
            raise typer.Exit(code=1)

        if not _emit(decided.model_dump(mode="json"), json_out):
            typer.echo(f"rejected {decided.id[:12]} ({decided.type.value})")
        store.close()

    @proposals_app.command("approve-all")
    def proposals_approve_all(
        vault: Optional[Path] = typer.Option(None),
        safety: str = typer.Option(
            "deterministic_verified",
            help="Safety class to approve in bulk. Ambiguous proposals are refused.",
        ),
        type_: Optional[str] = typer.Option(None, "--type"),
        source: Optional[str] = typer.Option(None, help="Limit to one source id."),
        limit: int = typer.Option(500),
        include_ambiguous: bool = typer.Option(
            False,
            "--include-ambiguous",
            help="Explicitly allow bulk approval of ambiguous proposals. Off by default.",
        ),
        dry_run: bool = typer.Option(True, help="Preview without approving."),
        apply: bool = typer.Option(
            False,
            "--apply",
            help="Also write the approved repairs to the vault. Off by default.",
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Approve many proposals at once, with a guard on ambiguous ones.

        Bulk-approving an ambiguous semantic proposal would be approving a
        decision nobody made, so it requires an explicit flag. Everything else
        is filtered by safety class, which is derived from provenance and
        cannot be asserted by a model.
        """
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        service = ProposalService(store)

        try:
            wanted = SafetyClass(safety)
        except ValueError:
            typer.echo(f"unknown safety class {safety!r}", err=True)
            store.close()
            raise typer.Exit(code=2)

        if wanted is SafetyClass.AMBIGUOUS and not include_ambiguous:
            typer.echo(
                "refusing to bulk-approve ambiguous proposals; pass --include-ambiguous "
                "if that is genuinely what you want",
                err=True,
            )
            store.close()
            raise typer.Exit(code=2)

        candidates = [
            p
            for p in service.list(
                status=ProposalStatus.PENDING,
                type=_enum_option(ProposalType, type_, "--type") if type_ else None,
                source_id=source,
                limit=limit,
            )
            if p.safety is wanted
        ]

        if dry_run:
            payload = {
                "dry_run": True,
                "matched": len(candidates),
                "safety": wanted.value,
                "targets": [p.operation.target for p in candidates[:20]],
            }
            if not _emit(payload, json_out):
                typer.echo(f"{len(candidates)} proposal(s) match safety={wanted.value} — none approved")
                for p in candidates[:10]:
                    typer.echo(f"  {p.id[:12]}  {p.type.value:<16} {p.operation.target}")
                typer.echo("\nRe-run with --no-dry-run to approve them.")
            store.close()
            return

        approved = [service.approve(p.id, note=f"batch approval (safety={wanted.value})") for p in candidates]

        # Approving records a decision; writing to the vault is a second,
        # separately-gated step. --apply runs the same ProposalApplier the
        # single-proposal path uses, so the refusal rules and the per-file
        # backup under <state_dir>/backups apply identically in bulk.
        apply_report = None
        if apply:
            applier = ProposalApplier(
                settings.vault_path, store, settings.state_dir / "backups"
            )
            apply_report = applier.apply(approved, apply=True)

        payload = {
            "approved": len(approved),
            "safety": wanted.value,
            "apply": apply_report.to_dict() if apply_report else None,
        }
        if not _emit(payload, json_out):
            typer.echo(f"approved {len(approved)} proposal(s) with safety={wanted.value}")
            if apply_report is None:
                typer.echo("Nothing was written to the vault or activated.")
            else:
                _print_apply(apply_report, applied_flag=True)
        store.close()

    @proposals_app.command("generate")
    def proposals_generate(
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Generate metadata-repair proposals from the vault's frontmatter defects."""
        from ..corpus.indexer import CorpusIndexer
        from ..proposals import build_repair_proposals, summarize

        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        index = CorpusIndexer(settings).build_index()
        built = list(build_repair_proposals(index.files))
        created, skipped = ProposalService(store).create_many(built)

        payload = {
            "built": len(built),
            "created": created,
            "skipped_existing": skipped,
            "by_safety": summarize(built),
        }
        if not _emit(payload, json_out):
            typer.echo(
                f"built {len(built)} repair proposals: {created} new, {skipped} already known"
            )
            typer.echo(f"by safety: {payload['by_safety']}")
            typer.echo("\nNothing was applied. Review with `forge proposals list`.")
        store.close()


# -- helpers ---------------------------------------------------------------


def _print_source(source: Any) -> None:
    icon = {
        IngestionStatus.INGESTED: "ok",
        IngestionStatus.UNCHANGED: "unchanged",
        IngestionStatus.OCR_REQUIRED: "OCR_REQUIRED",
        IngestionStatus.PARSE_FAILED: "PARSE_FAILED",
        IngestionStatus.NOT_FOUND: "NOT_FOUND",
        IngestionStatus.UNSUPPORTED: "UNSUPPORTED",
    }.get(source.status, source.status.value)

    typer.echo(f"[{icon}] {source.locator}")
    if source.status is IngestionStatus.INGESTED:
        pages = f", {source.pages} pages" if source.pages else ""
        typer.echo(f"    source {source.source_id[:12]} | document {source.document_id[:12]}")
        typer.echo(f"    {source.spans} spans{pages}, {source.chars} chars")
        if source.extraction_status.value not in ("skipped_no_provider",):
            typer.echo(
                f"    extraction: {source.extraction_status.value} | "
                f"{source.concepts_proposed} concepts, {source.claims_proposed} claims, "
                f"{source.proposals_created} proposals"
            )
    elif source.status is IngestionStatus.UNCHANGED:
        # "Unchanged" describes the *source*, not the run. Extraction still
        # runs over an unchanged source that was never extracted, so claiming
        # "no work done" while 2 model calls per span are in flight is simply
        # false — and it hid a 5.7-hour run's real behaviour on 2026-08-19.
        did_extract = source.proposals_created or source.concepts_proposed or source.claims_proposed
        if did_extract:
            typer.echo(f"    source unchanged ({source.spans} spans reused)")
            typer.echo(
                f"    extraction: {source.extraction_status.value} | "
                f"{source.concepts_proposed} concepts, {source.claims_proposed} claims, "
                f"{source.proposals_created} proposals"
            )
        else:
            typer.echo(f"    unchanged ({source.spans} spans already stored) — no work done")
    if source.detail:
        typer.echo(f"    {source.detail}")
    for warning in source.warnings[:5]:
        typer.echo(f"    warning: {warning}")
    for error in source.errors[:5]:
        typer.echo(f"    error: {error}")


def _print_apply(report: Any, *, applied_flag: bool) -> None:
    if report.dry_run:
        eligible = [o for o in report.outcomes if not o.refused_reason]
        if eligible:
            typer.echo(
                f"\nvault write-back NOT performed (pass --apply). "
                f"{len(eligible)} file(s) would change:"
            )
            for outcome in eligible:
                typer.echo(f"  {outcome.path}:{outcome.line}")
                typer.echo(f"    - {outcome.before}")
                typer.echo(f"    + {outcome.after}")
        for outcome in report.refused:
            typer.echo(f"\nnot applicable: {outcome.refused_reason}")
        return

    for outcome in report.applied:
        typer.echo(f"\napplied to {outcome.path}:{outcome.line}")
        typer.echo(f"  backup: {outcome.backup_path}")
    for outcome in report.refused:
        typer.echo(f"\nrefused: {outcome.refused_reason}")


def _report_unresolved(proposal_id: str, ambiguous: list[Any]) -> None:
    if ambiguous:
        typer.echo(f"{proposal_id!r} matches {len(ambiguous)} proposals:", err=True)
        for proposal in ambiguous:
            typer.echo(f"  {proposal.id}  {proposal.operation.target}", err=True)
    else:
        typer.echo(f"no proposal {proposal_id!r}", err=True)


def _embeddings(settings: Settings, *, enabled: bool) -> Any:
    if not enabled:
        return NullEmbeddingProvider()
    return OllamaEmbeddingProvider(settings.llm.base_url)


def _index_embeddings(settings: Settings, store: SqliteStore, report: Any) -> None:
    service = SearchService(store, embeddings=OllamaEmbeddingProvider(settings.llm.base_url))
    if not service.embeddings.available:
        typer.echo("note: embeddings unavailable; lexical retrieval only")
        return
    total = 0
    for source_report in report.sources:
        if source_report.document_id:
            total += service.index_embeddings(store.spans_for_document(source_report.document_id))
    typer.echo(f"embedded {total} spans")


def _excerpt(text: str, limit: int = 110) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
