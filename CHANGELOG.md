# Changelog

Notable changes, newest first. Dates are the day the work landed on `main`.

The version numbers are [semantic](https://semver.org/), with the caveat
that this is pre-1.0: the CLI's output format and the derived store's schema
may change between minor versions. The store is rebuildable from the vault,
so a schema change costs a re-index and never your notes.

## Unreleased

### Fixed

- **Typing a hyphen, a bracket or an apostrophe into the dashboard's search
  box raised `sqlite3.OperationalError`.** `off-by-one` came back as *no such
  column: by*, and `O(n)`, `don't` and `c++` as FTS5 syntax errors. FTS5's
  `MATCH` takes a query language, not a search string, and
  `api.queries.search_spans` passed typed text into it untranslated, so every
  caller of that function was affected: `forge dash`, the HTTP API and the MCP
  server. `SearchService` had been translating its queries since Phase 2 and
  had a test proving hostile input is safe; the second path around it did not.
  The translator is now `forge.retrieval.fts_query`, public, called in
  `queries.search_spans`, so human text stops at that layer and the store
  still speaks FTS5.
- **Search found nothing until the last letter of a word landed.** FTS5
  matches whole tokens, so `attentio` returned nothing where `attention`
  returned 33 hits: on a 674-file vault the box reported "nothing in the vault
  matches" for eight of the nine keystrokes in a word. The dashboard searches
  on a 250 ms debounce while you type, so it now asks `fts_query` to match the
  final token as a prefix, and results narrow as the word is typed (50 hits at
  `atte`, 41 at `atten`, 33 at `attention`). A trailing space means the word is
  finished and turns the prefix off. This looked like a performance problem and
  was reported as one; every query underneath it returns in under 11 ms.

## 0.2.0 (2026-09-08)

### Added

- **`forge demo`.** Writes a ten-file vault whose defects are known, runs the
  real deterministic pipeline over it, and reports which of the five it found.
  Nothing is canned: every line is read out of the same reports
  `forge diagnostics` and `forge bootstrap` produce, and the command exits
  non-zero if a planted defect goes unreported. Needs no vault, no model, no
  API key and no network, so a first look at the tool costs one command.
- **Help panels.** `forge --help` grouped its 38 commands into one flat list,
  which told a reader everything except where to start. They are now ordered
  by how far into the tool you are: Start here, Read the vault, Build the
  knowledge model, Question what it knows, Maintain and integrate, Measure. A
  test fails if a new command is added without a panel.
- **`scripts/screenshot.py`**, which regenerates the dashboard images in
  `docs/media/` through Textual's own `save_screenshot`, against the vault
  `forge demo` writes. The images in the README are the output of a run
  anyone can repeat.

### Fixed

- **The package did not work on Python 3.10, the version it declared as its
  floor.** `storage/backup.py` imported `UTC` from `datetime`, which exists
  from 3.11, so `pip install forge-kb` on 3.10 succeeded and then raised
  ImportError on the first import. Ruff's `target-version` said py311 while
  the package said 3.10, which is how it was written; the two now match.
- **A claim statement containing a `|` made `stale_reason` name the wrong
  fields.** The fingerprint is `status|tier|statement|superseded_by`, and only
  the statement can hold a pipe, so a table row or a shell pipeline produced
  five parts: `statement` was reported as the text before the pipe,
  `superseded_by` as the text after it, and the real supersession id fell off
  the end. A fingerprint that is not four fields at all, from a restore or an
  older schema, now says only that the content differs rather than guessing.
- **Approving a knowledge proposal reported a failure.** Ten approved
  concepts printed ten `not applicable: safety class model_generated is not
  automatically applicable` lines, so a run that had recorded every approval
  read as one where nothing worked. A concept or a claim was never a
  candidate for vault write-back at any safety class, and the answer is now
  what to do next: `create_concept changes the knowledge graph, not a file;
  run \`forge activate\` to apply it`.
- 18 `raise typer.Exit(...)` inside `except` blocks now say `from None`. The
  message has already been printed; chaining a traceback to it is noise.
- Three tests asserted `pytest.raises(Exception)` and so would have passed on
  an AttributeError from a renamed field, which is the opposite of what they
  were checking. Each names its real exception now.

### Changed

- **CI runs four jobs, not one.** Lint; pytest on 3.10 and 3.13, the declared
  floor and the current ceiling; and a job that builds the wheel, installs it
  alone, and runs `forge demo` from outside the checkout, which proves that
  `pip install forge-kb` followed by the README's first command works on a
  machine that has never seen the repository.
- **Ruff is pinned and its rule set is stated.** An unpinned linter reported
  360 findings on a commit that changed no Python, because CI had installed a
  newer version whose defaults had grown. The rules are now
  E4/E7/E9/F/I/B whatever version runs. All 360 were real and all are fixed.
- `Answer.cited_sources()` pairs each citation number with its own passage
  rather than leaving callers to zip `cited` against `sources()`.
- The dashboard's window and terminal tab are named `forge dash` rather than
  `ForgeDashboard`.
- The README leads with what the tool found rather than what it is, carries a
  screenshot, and its counts were recounted from the filesystem. The Status
  section said "Phases 0-4 complete, Phases 5-10 are not started" while 6, 8,
  9 and 10 had every gate ticked and shipped commands.

## 0.1.0 (2026-09-03)

First release. Deterministic indexing, diagnostics, ingestion with
page- and section-level provenance, proposals and activation, the knowledge
graph, retrieval, the evolution workflow, the HTTP API, the MCP server and
the dashboard. See [`docs/roadmap.md`](docs/roadmap.md) for what each phase
gated on.
