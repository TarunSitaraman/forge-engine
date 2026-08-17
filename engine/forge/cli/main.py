"""Forge CLI.

Makes the engine demonstrable without a web application.

This module defines the Phase 1 commands and the root ``app``; later phases
register their own commands onto it from :mod:`forge.cli.phase2`, ``phase3``,
and ``phase4``.

    forge index          index the vault, detect changes, write reports
    forge status         engine + provider + derived-state status
    forge corpus-stats   statistics computed from the filesystem
    forge diagnostics    frontmatter / link / convention diagnostics
    forge inspect        everything known about one file
    forge model-test     local-model capability spike

Every command is read-only with respect to the vault. Only ``.forge/`` is
written.
"""

from __future__ import annotations

import json as jsonlib
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from ..config import ConfigError, Settings
from ..corpus import IndexPipeline, analyze_conventions, compute_stats, load_store
from ..corpus.diagnostics import frontmatter_report, link_report
from ..corpus.indexer import CorpusIndexer, detect_changes
from ..llm import CALLS, ProviderUnavailable, get_provider
from ..logging import bind_run, configure_logging, new_run_id
from ..spike import render_markdown, run_spike

app = typer.Typer(
    # Completion is on so `forge <TAB>` works in zsh/bash/fish. Install it once
    # per shell with `forge --install-completion`.
    add_completion=True,
    no_args_is_help=True,
    help="Forge Knowledge OS engine. Read-only with respect to the vault.",
)

err = typer.echo


def _settings(vault: Optional[Path], log_level: str = "WARNING") -> Settings:
    try:
        settings = Settings.load(vault)
    except ConfigError as exc:
        err(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2)
    configure_logging(log_level, settings.log_format)
    bind_run(new_run_id())
    return settings


def _emit(payload: dict[str, Any], as_json: bool) -> bool:
    """Print JSON and report whether output is finished."""
    if as_json:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True))
        return True
    return False


# --------------------------------------------------------------------------


@app.command()
def index(
    vault: Optional[Path] = typer.Option(None, help="Vault path (defaults to repo root)."),
    persist: bool = typer.Option(True, help="Write sources/documents/spans to derived state."),
    reports: bool = typer.Option(True, help="Write JSON reports to .forge/reports/."),
    reset: bool = typer.Option(False, help="Drop derived state first (safe: it rebuilds)."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
    verbose: bool = typer.Option(False, "-v", help="Verbose logging."),
) -> None:
    """Index the vault deterministically and report what changed."""
    settings = _settings(vault, "INFO" if verbose else "WARNING")
    store = load_store(settings)
    if reset:
        store.reset()

    CALLS.reset()
    result = IndexPipeline(settings, store).run(persist=persist, write_reports=reports)
    summary = result.summary()
    summary["llm_calls"] = CALLS.count

    if _emit(summary, json_out):
        return

    c = result.changes.summary()
    typer.echo(f"Indexed {result.index.file_count} files in {result.index.duration_seconds}s")
    typer.echo(f"  fingerprint : {result.index.fingerprint()}")
    typer.echo(
        f"  changes     : {c['new']} new, {c['modified']} modified, "
        f"{c['unchanged']} unchanged, {c['deleted']} deleted"
    )
    typer.echo(
        f"  persisted   : {result.persisted_sources} sources, "
        f"{result.persisted_documents} documents, {result.persisted_spans} spans"
    )
    typer.echo(f"  LLM calls   : {CALLS.count}")
    if result.reports_written:
        typer.echo(f"  reports     : {settings.reports_dir}")
    store.close()


@app.command()
def status(
    vault: Optional[Path] = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show engine, derived-state, and provider status."""
    settings = _settings(vault)
    store = load_store(settings)

    indexer = CorpusIndexer(settings)
    on_disk = indexer.discover()
    previous = {s.locator: s.content_hash for s in store.list_sources()}

    provider_ok, provider_detail = False, "not checked"
    try:
        provider_ok, provider_detail = get_provider(settings).health()
    except ProviderUnavailable as exc:
        provider_detail = str(exc)
    except Exception as exc:  # provider misconfiguration must not break status
        provider_detail = f"{type(exc).__name__}: {exc}"

    payload = {
        "vault_path": str(settings.vault_path),
        "state_dir": str(settings.state_dir),
        "db_exists": settings.db_path.exists(),
        "markdown_files_on_disk": len(on_disk),
        "sources_indexed": len(previous),
        "store_counts": store.counts(),
        "llm": {
            "provider": settings.llm.provider,
            "base_url": settings.llm.base_url,
            "models": settings.llm.models,
            "reachable": provider_ok,
            "detail": provider_detail,
        },
    }

    if not _emit(payload, json_out):
        typer.echo(f"vault          : {settings.vault_path}")
        typer.echo(f"derived state  : {settings.state_dir} (exists: {settings.db_path.exists()})")
        typer.echo(f"markdown files : {len(on_disk)}")
        typer.echo(f"indexed sources: {len(previous)}")
        counts = store.counts()
        typer.echo(
            "  spans={spans} documents={documents} concepts={concepts} "
            "claims={claims} revisions={revisions}".format(**counts)
        )
        typer.echo(f"llm provider   : {settings.llm.provider} ({'OK' if provider_ok else 'UNAVAILABLE'})")
        typer.echo(f"  {provider_detail}")
    store.close()


@app.command(name="corpus-stats")
def corpus_stats(
    vault: Optional[Path] = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    top: int = typer.Option(10, help="How many entries in 'top' lists."),
) -> None:
    """Statistics computed from the filesystem, never from hand-maintained counts."""
    settings = _settings(vault)
    index = CorpusIndexer(settings).build_index()
    stats = compute_stats(index, top_n=top)

    if _emit(stats.to_dict(), json_out):
        return

    typer.echo(f"files              : {stats.file_count}")
    typer.echo(f"lines              : {stats.total_lines:,}")
    typer.echo(f"bytes              : {stats.total_bytes:,}")
    typer.echo(f"frontmatter        : {stats.frontmatter_coverage_pct}% coverage")
    typer.echo(f"canonical: true    : {stats.canonical_count}")
    typer.echo(f"headings           : {stats.heading_total:,}")
    typer.echo(f"code blocks        : {stats.code_blocks:,}")
    typer.echo(f"wikilinks          : {stats.wikilink_total:,}")
    typer.echo(f"markdown links     : {stats.markdown_link_total:,}")
    typer.echo(f"related: entries   : {stats.related_field_total:,}")
    typer.echo(f"duplicate hashes   : {stats.duplicate_hash_groups} group(s)")
    typer.echo("\nby folder:")
    for folder, v in list(stats.by_folder.items())[:top]:
        typer.echo(
            f"  {v['lines']:>7,} lines  {v['files']:>4} files  "
            f"{v['with_frontmatter']:>4} w/fm  {folder}"
        )
    typer.echo("\nfilename styles:")
    for style, n in stats.filename_styles.items():
        typer.echo(f"  {n:>4}  {style}")


@app.command()
def diagnostics(
    what: str = typer.Argument("all", help="all | frontmatter | links | conventions | graph"),
    vault: Optional[Path] = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    limit: int = typer.Option(15, help="Rows shown in text mode."),
) -> None:
    """Report metadata, link, and convention problems. Never modifies the vault."""
    settings = _settings(vault)
    index = CorpusIndexer(settings).build_index()

    payload: dict[str, Any] = {}
    if what in ("all", "frontmatter"):
        payload["frontmatter"] = frontmatter_report(index).to_dict()
    if what in ("all", "links"):
        payload["links"] = link_report(index).to_dict()
    if what in ("all", "conventions"):
        payload["conventions"] = analyze_conventions(index).to_dict()
    if what in ("all", "graph"):
        from ..graph import check_integrity
        from ..storage.sqlite_store import SqliteStore

        store = SqliteStore(settings.db_path)
        store.initialize()
        payload["graph"] = check_integrity(store).to_dict()
        store.close()
    if not payload:
        err(f"unknown diagnostics target {what!r}", err=True)
        raise typer.Exit(code=2)

    if _emit(payload, json_out):
        return

    if "frontmatter" in payload:
        fm = frontmatter_report(index)
        typer.echo("FRONTMATTER")
        s = fm.to_dict()["summary"]
        typer.echo(
            f"  {s['with_frontmatter']}/{s['total_files']} files have frontmatter; "
            f"{s['valid']} valid, {s['invalid']} invalid"
        )
        for code, n in fm.by_code.items():
            typer.echo(f"    {code}: {n}")
        typer.echo(
            f"  {fm.repairable_files} file(s) have verified repair proposals "
            f"(NOT applied — approval required)"
        )

    if "links" in payload:
        lr = link_report(index)
        typer.echo("\nLINKS")
        typer.echo(f"  {lr.total_links} total ({lr.wikilinks} wiki, {lr.markdown_links} markdown)")
        for status_name, n in lr.by_status.items():
            typer.echo(f"    {status_name}: {n}")
        typer.echo(
            f"  unresolved: {lr.unresolved_total} occurrences across "
            f"{lr.unresolved_distinct} distinct targets"
        )
        for target, info in list(lr.unresolved_targets.items())[:limit]:
            cands = f"  candidates={info['candidates']}" if info["candidates"] else ""
            typer.echo(f"    {info['count']:>3}x [{info['status']}] {target!r}{cands}")

    if "conventions" in payload:
        cr = analyze_conventions(index)
        typer.echo(f"\nCONVENTIONS — {cr.resolution_status}")
        for sysid, conf in cr.conformance.items():
            typer.echo(
                f"  {sysid}: {conf['files_in_scope']} files in scope; "
                f"filenames {conf['filename_pct']}%, tags {conf['tags_pct']}%, "
                f"frontmatter {conf['frontmatter_pct']}%"
            )
        for c in cr.conflicts:
            typer.echo(f"    conflict [{c['kind']}]: repo={c['repo_wide']!r} vs dsa={c['dsa_local']!r}")

    if "graph" in payload:
        gr = payload["graph"]
        typer.echo(f"\nGRAPH INTEGRITY — {'clean' if gr['clean'] else str(gr['errors']) + ' error(s)'}")
        typer.echo(f"  checked: {gr['checked']}")
        for code, count in gr["by_code"].items():
            typer.echo(f"    {code}: {count}")
        for finding in gr["findings"][:limit]:
            typer.echo(f"    [{finding['severity']}] {finding['code']} {finding['entity_id'][:12]}: {finding['detail']}")


@app.command()
def inspect(
    path: str = typer.Argument(..., help="Vault-relative path of a Markdown file."),
    vault: Optional[Path] = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    spans: bool = typer.Option(False, help="Also show derived spans."),
) -> None:
    """Show everything deterministically known about one file."""
    settings = _settings(vault)
    indexer = CorpusIndexer(settings)

    target = path[2:] if path.startswith("./") else path
    if not (settings.vault_path / target).is_file():
        err(f"not a file in the vault: {target}", err=True)
        raise typer.Exit(code=1)

    index = indexer.build_index()
    match = index.by_path().get(target)
    if match is None:
        err(f"file not indexed (excluded by configuration?): {target}", err=True)
        raise typer.Exit(code=1)

    payload = match.to_dict()
    payload["links"] = [link.to_dict() for link in match.links]

    if spans:
        source = next(s for s in indexer.to_sources(index) if s.locator == target)
        _, built = indexer.to_document_and_spans(match, source)
        payload["spans"] = [
            {
                "ordinal": sp.ordinal,
                "locator": sp.locator,
                "heading_path": list(sp.heading_path),
                "lines": f"{sp.start_line}-{sp.end_line}",
                "chars": len(sp.text),
            }
            for sp in built
        ]

    if _emit(payload, json_out):
        return

    typer.echo(f"path           : {match.path}")
    typer.echo(f"title          : {match.title}")
    typer.echo(f"content_hash   : {match.content_hash}")
    typer.echo(f"lines / bytes  : {match.line_count} / {match.byte_size}")
    typer.echo(f"frontmatter    : present={match.frontmatter_present} valid={match.frontmatter_valid}")
    typer.echo(f"  keys         : {list(match.frontmatter_keys)}")
    typer.echo(f"  type/status  : {match.doc_type} / {match.status}   canonical={match.canonical}")
    typer.echo(f"  tags         : {list(match.tags)}")
    typer.echo(f"  related      : {list(match.related)}")
    typer.echo(f"headings       : {match.heading_count}")
    typer.echo(f"code blocks    : {match.code_block_count} {list(match.code_languages)}")
    typer.echo(f"links          : {match.wikilink_count} wiki, {match.markdown_link_count} markdown")

    if match.diagnostics:
        typer.echo("diagnostics    :")
        for d in match.diagnostics:
            typer.echo(f"  [{d.severity.value}] {d.code.value}: {d.message}")
    if match.repairs:
        typer.echo("repair proposals (NOT applied):")
        for r in match.repairs:
            typer.echo(f"  line {r.line} verified={r.verified}")
            typer.echo(f"    - {r.original.strip()}")
            typer.echo(f"    + {r.proposed.strip()}")

    unresolved = [link for link in match.links if link.status.value not in ("resolved",)]
    if unresolved:
        typer.echo("unresolved links:")
        for link in unresolved:
            typer.echo(f"  line {link.line} [{link.status.value}] {link.target!r} -> {list(link.candidates)}")

    if spans:
        typer.echo("spans:")
        for sp in payload["spans"]:
            typer.echo(f"  #{sp['ordinal']:>3} {sp['lines']:>12}  {' > '.join(sp['heading_path'])}")


@app.command(name="model-test")
def model_test(
    vault: Optional[Path] = typer.Option(None),
    repetitions: int = typer.Option(3, help="Runs per task; reliability needs more than one."),
    role: str = typer.Option("extraction", help="Model role to exercise."),
    write: bool = typer.Option(True, help="Write docs/research/local-model-capability-spike.md."),
    note: list[str] = typer.Option(
        [], "--note", help="Context to record in the report (repeatable), e.g. hardware or why it did not run."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run the local-model capability spike and record the results honestly."""
    settings = _settings(vault)
    try:
        provider = get_provider(settings)
    except Exception as exc:
        err(f"cannot construct provider: {exc}", err=True)
        raise typer.Exit(code=2)

    report = run_spike(provider, model_role=role, repetitions=repetitions)

    if write:
        out = settings.vault_path / "docs" / "research" / "local-model-capability-spike.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report, notes=note), encoding="utf-8")

    if _emit(report.to_dict(), json_out):
        return

    typer.echo(f"provider : {report.provider}")
    typer.echo(f"model    : {report.model}")
    typer.echo(f"reachable: {report.reachable} — {report.detail}")
    if not report.reachable:
        typer.echo("\nNo capability results: the spike did not run. Nothing has been established.")
        raise typer.Exit(code=1)
    for t in report.tasks:
        typer.echo(
            f"  {t.task:<32} {t.successes}/{t.attempts} "
            f"({t.success_rate:.0%})  median {t.median_latency}s"
        )
    typer.echo(f"overall structured-output success: {report.overall_success_rate:.0%}")


# Phase 2 commands (ingest, search, concepts, documents, proposals) are
# registered here so Phase 1's commands stay exactly as they were.
from .phase2 import register as _register_phase2  # noqa: E402
from .phase3 import register as _register_phase3  # noqa: E402
from .phase4 import register as _register_phase4  # noqa: E402

_register_phase2(app, _settings)
_register_phase3(app, _settings)
_register_phase4(app, _settings)


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
