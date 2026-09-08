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
from typing import Any

import typer

from ..config import ConfigError, Settings, env_file_path
from ..corpus import IndexPipeline, analyze_conventions, compute_stats, load_store
from ..corpus.diagnostics import frontmatter_report, link_report
from ..corpus.indexer import CorpusIndexer
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


def _version() -> str:
    """The installed distribution's version, or a marker that it is not installed.

    Read from package metadata rather than a constant, so it cannot disagree
    with what `pip` reports: a version string maintained by hand is a version
    string that goes stale the first time someone forgets it.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("forge-kb")
    except PackageNotFoundError:  # pragma: no cover - running from a checkout
        return "unknown (not installed as a package)"


def _show_version(value: bool) -> None:
    if not value:
        return
    from .. import api as _api  # noqa: F401  (import only to read its version)
    from ..api import API_VERSION
    from ..storage import SCHEMA_VERSION

    typer.echo(f"forge-kb {_version()}")
    typer.echo(f"  store schema : v{SCHEMA_VERSION}")
    typer.echo(f"  api          : {API_VERSION}")
    raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_show_version, is_eager=True, help="Show the version."
    ),
) -> None:
    """Forge Knowledge OS engine. Read-only with respect to the vault."""


def _settings(vault: Path | None, log_level: str = "WARNING") -> Settings:
    try:
        settings = Settings.load(vault)
    except ConfigError as exc:
        err(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from None
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
def demo(
    path: Path | None = typer.Option(
        None, help="Where to write the sample vault. A temporary directory by default."
    ),
    force: bool = typer.Option(False, "--force", help="Write into a directory that is not empty."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Write a small vault with known defects and show what the engine finds in it.

    Every other command needs a vault, and someone evaluating this tool does not
    have one yet. This writes ten files, runs the real deterministic pipeline
    over them, and reports which of the planted defects came back. Nothing is
    canned: the findings are read out of the same reports `forge diagnostics`
    and `forge bootstrap` produce.

    No model, no API key, no network. The vault it writes is yours to keep;
    the path is printed at the end so you can open `forge dash` on it.
    """
    import tempfile

    from .demo import build, is_occupied, render, write_vault

    target = Path(path) if path else Path(tempfile.mkdtemp(prefix="forge-demo-"))
    if is_occupied(target) and not force:
        err(
            f"{target} is not empty. The demo writes ten files and will not "
            f"write over notes it did not create; pass --force to use it anyway.",
            err=True,
        )
        raise typer.Exit(code=2)
    target.mkdir(parents=True, exist_ok=True)
    write_vault(target)

    settings = _settings(target)
    CALLS.reset()
    checks, counts = build(settings)

    if _emit(
        {
            "vault_path": str(target),
            "llm_calls": CALLS.count,
            "counts": counts,
            "checks": [
                {"title": c.title, "found": c.found, "evidence": c.evidence} for c in checks
            ],
        },
        json_out,
    ):
        return

    for line in render(checks, counts, target, CALLS.count):
        typer.echo(line)

    # A missed defect is a regression in the engine, not a cosmetic problem with
    # the demo, so it has to leave a non-zero status behind for CI to catch.
    if counts["found"] != counts["total"]:
        raise typer.Exit(code=1)


@app.command()
def index(
    vault: Path | None = typer.Option(None, help="Vault path (defaults to repo root)."),
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
    vault: Path | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show engine, derived-state, and provider status."""
    settings = _settings(vault)
    store = load_store(settings)

    indexer = CorpusIndexer(settings)
    on_disk = indexer.discover()
    previous = {s.locator: s.content_hash for s in store.list_sources()}

    provider_ok, provider_detail = False, "not checked"
    # Only a starting point. `settings.llm.models` is the *ollama* role map and
    # is populated with its defaults whatever the provider, so reading it alone
    # made a cloud deployment report `llama3.1:8b` while actually running
    # llama-3.3-70b-versatile (observed on the Mac, 2026-08-29). The provider is
    # asked below; this stands only when there is no provider to ask.
    model_identity = settings.llm.models.get("extraction") or next(
        iter(settings.llm.models.values()), "?"
    )
    # What is *configured* and what is *serving* can differ when a fallback is
    # set. Report the one doing the work, a status line naming a provider that
    # is not answering is worse than no status line.
    serving = settings.llm.provider
    try:
        provider = get_provider(settings)
        provider_ok, provider_detail = provider.health()
        try:
            serving = provider.capabilities.name
        except Exception:  # capabilities may probe the network; keep status alive
            pass
        # The variant is part of the derivation key, so a mode left set in the
        # environment silently produces a different cached corpus. Reasoning
        # off was measured and rejected (provider-availability.md §8) yet stayed
        # exported for a whole extraction run because nothing displayed it.
        # Ask the provider what it would actually run rather than trusting the
        # role map. This line exists to keep a wrong assumption about the active
        # model from governing a long run; it did once, and cost 5.66 hours,
        # so a status line that names the wrong model is the exact failure it
        # was added to prevent.
        resolve = getattr(provider, "resolve_model", None)
        if callable(resolve):
            try:
                model_identity = resolve("extraction")
            except Exception:
                pass  # keep the configured value rather than blanking the line
        model_identity += str(getattr(provider, "identity_variant", "") or "")
    except ProviderUnavailable as exc:
        provider_detail = str(exc)
    except Exception as exc:  # provider misconfiguration must not break status
        provider_detail = f"{type(exc).__name__}: {exc}"

    env_file = env_file_path()
    payload = {
        "vault_path": str(settings.vault_path),
        "state_dir": str(settings.state_dir),
        "env_file": {"path": str(env_file), "exists": env_file.is_file()},
        "db_exists": settings.db_path.exists(),
        "markdown_files_on_disk": len(on_disk),
        "sources_indexed": len(previous),
        "store_counts": store.counts(),
        "llm": {
            "provider": settings.llm.provider,
            "fallback": settings.llm.fallback,
            "serving": serving,
            "base_url": settings.llm.base_url,
            "models": settings.llm.models,
            "model_identity": model_identity,
            "reachable": provider_ok,
            "detail": provider_detail,
        },
    }

    if not _emit(payload, json_out):
        typer.echo(f"vault          : {settings.vault_path}")
        typer.echo(f"derived state  : {settings.state_dir} (exists: {settings.db_path.exists()})")
        typer.echo(f"settings file  : {env_file} ({'loaded' if env_file.is_file() else 'absent'})")
        typer.echo(f"markdown files : {len(on_disk)}")
        typer.echo(f"indexed sources: {len(previous)}")
        counts = store.counts()
        typer.echo(
            "  spans={spans} documents={documents} concepts={concepts} "
            "claims={claims} revisions={revisions}".format(**counts)
        )
        # If the fallback is serving, the configured provider is by definition
        # the one that failed its health check. Reporting its state from the
        # fallback's health would print "ollama (OK)" while ollama is down.
        fallback_active = not serving.startswith(settings.llm.provider)
        state = "UNAVAILABLE" if fallback_active else ("OK" if provider_ok else "UNAVAILABLE")
        typer.echo(f"llm provider   : {settings.llm.provider} ({state})")
        if settings.llm.fallback:
            typer.echo(f"  fallback       : {settings.llm.fallback}")
        if fallback_active:
            typer.echo(
                f"  SERVING        : {serving} "
                f"({'OK' if provider_ok else 'UNAVAILABLE'})  <- fallback is active"
            )
        typer.echo(f"  model identity : {model_identity}")
        if model_identity.endswith("+nothink"):
            typer.echo(
                "  NOTE: reasoning is OFF (FORGE_OLLAMA_THINK=0). Measured and "
                "rejected.\n         See docs/research/provider-availability.md §8."
            )
        typer.echo(f"  {provider_detail}")
    store.close()


@app.command(name="corpus-stats")
def corpus_stats(
    vault: Path | None = typer.Option(None),
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
    vault: Path | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    limit: int = typer.Option(15, help="Rows shown in text mode."),
    html: Path | None = typer.Option(
        None, "--html", help="Write a self-contained HTML report to this path."
    ),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Write a Markdown report to this path."
    ),
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

    # Files are written before any early return, so `--json --html out.html`
    # produces both rather than silently dropping one.
    if html or markdown:
        from ..reporting import render_html, render_markdown

        name = settings.vault_path.name or str(settings.vault_path)
        for target, render in ((html, render_html), (markdown, render_markdown)):
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render(payload, vault_name=name), encoding="utf-8")
            typer.echo(f"wrote {target}")

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
            f"(NOT applied, approval required)"
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
        typer.echo(f"\nCONVENTIONS: {cr.resolution_status}")
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
        typer.echo(f"\nGRAPH INTEGRITY: {'clean' if gr['clean'] else str(gr['errors']) + ' error(s)'}")
        typer.echo(f"  checked: {gr['checked']}")
        for code, count in gr["by_code"].items():
            typer.echo(f"    {code}: {count}")
        for finding in gr["findings"][:limit]:
            typer.echo(f"    [{finding['severity']}] {finding['code']} {finding['entity_id'][:12]}: {finding['detail']}")


@app.command()
def inspect(
    path: str = typer.Argument(..., help="Vault-relative path of a Markdown file."),
    vault: Path | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    spans: bool = typer.Option(False, help="Also show derived spans."),
) -> None:
    """Show everything deterministically known about one file."""
    settings = _settings(vault)
    indexer = CorpusIndexer(settings)

    target = path.removeprefix("./")
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
    vault: Path | None = typer.Option(None),
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
        raise typer.Exit(code=2) from None

    report = run_spike(provider, model_role=role, repetitions=repetitions)

    if write:
        out = settings.vault_path / "docs" / "research" / "local-model-capability-spike.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report, notes=note), encoding="utf-8")

    if _emit(report.to_dict(), json_out):
        return

    typer.echo(f"provider : {report.provider}")
    typer.echo(f"model    : {report.model}")
    typer.echo(f"reachable: {report.reachable}. {report.detail}")
    if not report.reachable:
        typer.echo("\nNo capability results: the spike did not run. Nothing has been established.")
        raise typer.Exit(code=1)
    for t in report.tasks:
        typer.echo(
            f"  {t.task:<32} {t.successes}/{t.attempts} "
            f"({t.success_rate:.0%})  median {t.median_latency}s"
        )
    typer.echo(f"overall structured-output success: {report.overall_success_rate:.0%}")

    # The spike has always recorded why each attempt failed; until 2026-08-29
    # this command printed only the score. A run reading `0/3 median None` four
    # times over told you nothing about whether the model was rejected, the
    # request malformed, or the JSON unparseable, and a reachable provider
    # scoring 0% is a configuration problem far more often than a model one.
    failures: dict[str, list[str]] = {}
    for t in report.tasks:
        for f in t.failures:
            line = f"{f.get('kind', 'unknown')}: {str(f.get('error', '')).strip()}"
            failures.setdefault(line, []).append(t.task)

    if failures:
        typer.echo("\nfailures:", err=True)
        for line, tasks in failures.items():
            typer.echo(f"  x{len(tasks)}  {line[:400]}", err=True)
        if report.overall_success_rate == 0 and len(failures) == 1:
            typer.echo(
                "\nEvery task failed the same way. That is the provider or the request\n"
                "shape, not the model's capability. Nothing here is a measurement of\n"
                "how good the model is.",
                err=True,
            )
        raise typer.Exit(code=1)


# Phase 2 commands (ingest, search, concepts, documents, proposals) are
# registered here so Phase 1's commands stay exactly as they were.
from .phase2 import register as _register_phase2
from .phase3 import register as _register_phase3
from .phase4 import register as _register_phase4
from .phase9 import register as _register_phase9

_register_phase2(app, _settings)
_register_phase3(app, _settings)
_register_phase4(app, _settings)
_register_phase9(app, _settings)


@app.command()
def serve(
    vault: Path | None = typer.Option(None),
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Serve the read-only HTTP API and graph explorer (needs the `api` extra).

    Binds to localhost by default and on purpose. The API publishes a vault's
    entire contents, spans included, and it has no authentication; exposing it
    on 0.0.0.0 publishes your notes to the network. Pass `--host 0.0.0.0`
    deliberately or not at all.
    """
    settings = _settings(vault)
    try:
        import uvicorn

        from ..api import create_app
    except ModuleNotFoundError as exc:
        err(f"{exc}", err=True)
        raise typer.Exit(code=2) from None

    store = load_store(settings)
    counts = store.counts()
    store.close()
    if not counts.get("concepts") and not counts.get("sources"):
        err(
            "the store is empty, so the explorer will have nothing to show. "
            "Run `forge bootstrap --apply` or `forge ingest` first.",
            err=True,
        )

    typer.echo(f"vault    : {settings.vault_path}")
    typer.echo(f"store    : {settings.db_path}")
    typer.echo(f"explorer : http://{host}:{port}/")
    typer.echo(f"api docs : http://{host}:{port}/docs")
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo(
            f"\nWARNING: bound to {host}, not localhost. This API is unauthenticated "
            "and serves the full text of every ingested source."
        )
    uvicorn.run(create_app(settings), host=host, port=port, reload=reload)


@app.command()
def mcp(
    vault: Path | None = typer.Option(None),
) -> None:
    """Serve the knowledge model to an agent over MCP (needs the `mcp` extra).

    Speaks MCP over stdio, which is what an agent host launches. It prints
    nothing to stdout: stdout *is* the protocol channel, and a stray line
    corrupts the stream. Diagnostics go to stderr.

    Read-only, like the HTTP API. Knowledge changes only through proposal and
    activation, which require a human decision.
    """
    settings = _settings(vault)
    try:
        from ..mcp import create_server
    except ModuleNotFoundError as exc:
        err(f"{exc}", err=True)
        raise typer.Exit(code=2) from None

    store = load_store(settings)
    counts = store.counts()
    store.close()
    # stderr on purpose: stdout carries the protocol.
    err(f"forge mcp: {settings.db_path}", err=True)
    err(
        f"  {counts.get('concepts', 0)} concepts, {counts.get('claims', 0)} claims, "
        f"{counts.get('spans', 0)} spans",
        err=True,
    )
    if not counts.get("concepts") and not counts.get("sources"):
        err(
            "  the store is empty; run `forge bootstrap --apply` or `forge ingest` first.",
            err=True,
        )
    create_server(settings).run(transport="stdio")


# Registered last, deliberately: the shell enumerates the commands of the group
# it is given, so every command above must already be attached when it runs.
@app.command()
def shell(
    vault: Path | None = typer.Option(None),
) -> None:
    """Open an interactive Forge shell with slash commands."""
    from .shell import run as _run_shell

    settings = _settings(vault)
    store = load_store(settings)
    on_disk = CorpusIndexer(settings).discover()
    indexed = len(store.list_sources())
    raise typer.Exit(code=_run_shell(app, settings, len(on_disk), indexed))


@app.command()
def tui(
    vault: Path | None = typer.Option(None),
) -> None:
    """Open the full-screen Forge TUI (needs the `tui` extra)."""
    from .tui import Stats, run_tui

    settings = _settings(vault)
    store = load_store(settings)
    on_disk = CorpusIndexer(settings).discover()
    counts = store.counts()
    stats = Stats(
        files=len(on_disk),
        indexed=len(store.list_sources()),
        spans=int(counts.get("spans", 0)),
        llm_calls=0,
    )
    raise typer.Exit(code=run_tui(app, settings, stats))


@app.command()
def dash(
    vault: Path | None = typer.Option(None),
) -> None:
    """Open the Forge dashboard: browse the vault, its graph and its problems.

    The front door. `forge shell` and `forge tui` open on a prompt; this opens
    on your vault, what is in it, what is wrong with it, and what to do next.
    Every screen is deterministic: no model is called and nothing is written.
    """
    from .dashboard import run_dashboard
    from .snapshot import build_snapshot

    settings = _settings(vault)
    raise typer.Exit(code=run_dashboard(settings, build_snapshot(settings)))


# --------------------------------------------------------------------------
# Help panels.
#
# Thirty-eight commands in one alphabetical list is a reference, not an
# interface: it tells a new reader everything except where to start. Grouping
# them answers that, and keeping the map here rather than as a keyword argument
# on thirty-eight decorators means the grouping can be read, and corrected, in
# one place.
#
# The panels are ordered by how far into the tool you are, so `forge --help`
# reads top to bottom as a path: look at a vault, then model what is in it, then
# ask it questions.

HELP_PANELS: dict[str, tuple[str, ...]] = {
    "Start here": ("demo", "index", "dash", "diagnostics", "status"),
    "Read the vault": (
        "corpus-stats",
        "inspect",
        "search",
        "ask",
        "shell",
        "tui",
        "serve",
    ),
    "Build the knowledge model": (
        "bootstrap",
        "ingest",
        "extract-plan",
        "proposals",
        "activate",
        "relationships",
        "identity",
        "embeddings",
    ),
    "Question what it knows": (
        "concepts",
        "concept",
        "claim",
        "documents",
        "graph",
        "belief",
        "changes",
        "gaps",
        "question",
        "evolve",
        "workflow",
    ),
    "Maintain and integrate": ("backup", "restore", "upstream", "mcp"),
    "Measure": ("model-test", "extraction-eval", "retrieval-eval"),
}


def _command_name(info: Any) -> str:
    """The name a command is invoked by, whether or not it was given one."""
    if info.name:
        return info.name
    callback = getattr(info, "callback", None)
    return getattr(callback, "__name__", "").replace("_", "-")


def assign_help_panels(target: typer.Typer, panels: dict[str, tuple[str, ...]]) -> list[str]:
    """Put every command in a panel and order them. Returns the names with no panel.

    Returning the leftovers rather than silently ignoring them is what keeps a
    new command from disappearing into an unlabelled group at the bottom of the
    help: a test asserts the list is empty, so adding a command without deciding
    where it belongs fails the suite instead of shipping.

    The registration list is also sorted to match. Typer draws panels in the
    order it first meets one, which is the order the phase modules happen to
    attach their commands, so without this the map above would describe the
    grouping but not the sequence: `Measure` came third because `model-test` is
    defined early in this file.
    """
    rank: dict[str, tuple[int, int]] = {}
    of: dict[str, str] = {}
    for panel_index, (panel, names) in enumerate(panels.items()):
        for name_index, name in enumerate(names):
            of[name] = panel
            rank[name] = (panel_index, name_index)

    unplaced: list[str] = []
    for info in list(target.registered_commands) + list(target.registered_groups):
        name = _command_name(info)
        panel = of.get(name)
        if panel is None:
            unplaced.append(name)
        else:
            info.rich_help_panel = panel

    # Anything unplaced sorts last rather than first, so a command that has not
    # been given a home does not open the help.
    last = (len(panels), 0)
    target.registered_commands.sort(key=lambda i: rank.get(_command_name(i), last))
    return unplaced


assign_help_panels(app, HELP_PANELS)


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
