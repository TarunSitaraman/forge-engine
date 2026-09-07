"""The dashboard: what it computes, what it says, and what it costs.

Three layers, tested separately for the reason they are separated.

* **The snapshot** — what counts as a problem, what to suggest to a vault in
  each state, and which numbers are real. No terminal involved.
* **The rendering** — pure functions from those numbers to markup. This is
  where the explorer's real bug lived: it compared a derivation against
  `"MODEL"` when the value serializes lowercase, so every model-derived edge
  rendered as if a human had asserted it. That comparison is pinned here.
* **The application** — driven headless through Textual's pilot, because
  bindings and lazy tab loading are behaviour, not state.

The last test is the point of the whole feature: browsing the entire dashboard
makes zero model calls.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from forge.api.models import ConceptDetail, ConceptSummary, GapResponse, NeighborItem
from forge.api.models import Provenance as ApiProvenance
from forge.cli.dashboard import (
    INSTALL_HINT,
    escape,
    fit,
    render_concept,
    render_gap_header,
    render_header,
    render_overview,
    render_tabbar,
    run_dashboard,
    where_of,
)
from forge.cli.main import app as cli
from forge.cli.shell import FULLSCREEN, REFUSED, Kind, parse, visible_names
from forge.cli.snapshot import (
    NextAction,
    VaultSnapshot,
    browse_concepts,
    browse_search,
    build_snapshot,
    suggest_next,
)
from forge.config import Settings
from forge.llm.base import CALLS
from typer.testing import CliRunner

FIXTURE_VAULT = Path(__file__).resolve().parents[1] / "fixtures" / "vault"


@pytest.fixture
def bare_vault(tmp_path: Path) -> Settings:
    """A vault that exists and has never been indexed."""
    vault = tmp_path / "vault"
    (vault / ".git").mkdir(parents=True)
    (vault / "Note.md").write_text("# Note\n\nSome text.\n", encoding="utf-8")
    return Settings(vault_path=vault, state_dir=tmp_path / "state")


@pytest.fixture
def live_vault(tmp_path: Path) -> Settings:
    """The fixture vault, indexed and bootstrapped through the real CLI.

    Driven through the CLI rather than by writing rows into the store, so what
    the dashboard reads is what `forge index` and `forge bootstrap` actually
    produce. The fixture vault carries the defects on purpose — broken links,
    an ambiguous one, unparseable frontmatter, a duplicate file — so the issue
    views have something real to show.
    """
    vault = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, vault)
    (vault / ".git").mkdir()
    runner = CliRunner()
    for argv in (
        ["index", "--vault", str(vault)],
        ["bootstrap", "--vault", str(vault), "--apply"],
    ):
        result = runner.invoke(cli, argv)
        assert result.exit_code == 0, result.output
    return Settings.load(vault)


# -- the snapshot ----------------------------------------------------------


class TestSnapshot:
    def test_an_unindexed_vault_is_told_to_index_and_nothing_else(self, bare_vault):
        """One instruction, not five.

        A first screen listing everything that could be done to a vault with
        nothing in the store is noise: every later step depends on this one.
        """
        snapshot = build_snapshot(bare_vault)

        assert snapshot.empty
        assert not snapshot.indexed
        assert [a.command for a in snapshot.next_actions] == ["forge index"]

    def test_the_first_suggestion_promises_no_model_call(self, bare_vault):
        """Someone with no API key must be able to tell that this step is free."""
        why = build_snapshot(bare_vault).next_actions[0].why
        assert "no model" in why.lower()

    def test_a_broken_wikilink_becomes_an_issue_naming_the_file(self, live_vault):
        snapshot = build_snapshot(live_vault)

        broken = [i for i in snapshot.issues if i.detail.startswith("[[")]
        assert broken, "the fixture vault's broken links produced no issues"
        assert all(i.where.endswith(".md") for i in broken)
        assert snapshot.links_unresolved == len(broken)

    def test_an_ambiguous_link_carries_the_page_it_probably_meant(self, live_vault):
        """A hint that names a candidate is actionable; "unresolved" is not."""
        hints = [i.hint for i in build_snapshot(live_vault).issues if i.kind == "ambiguous"]
        assert hints, "the fixture vault's ambiguous link produced no issue"
        assert any(h.startswith("probably ") and h.endswith(".md") for h in hints)

    def test_a_file_with_no_frontmatter_is_not_called_a_problem(self, live_vault):
        """The parser rates a missing frontmatter block INFO deliberately.

        Plenty of good notes have none. Counting them would put a three-figure
        number of invented problems on a healthy vault — the real one has 263
        such files and zero actual defects.
        """
        snapshot = build_snapshot(live_vault)

        without_frontmatter = snapshot.files - snapshot.frontmatter_present
        assert without_frontmatter > 0, "the fixture no longer exercises this"
        assert snapshot.metadata_issues < without_frontmatter
        assert all("FM003" not in i.kind for i in snapshot.issues)

    def test_the_issue_list_is_capped_but_the_count_is_not(self, live_vault):
        """A truncated list that reports its own length lies about the vault."""
        snapshot = build_snapshot(live_vault, issue_limit=2)

        assert len(snapshot.issues) == 2
        assert snapshot.issues_total > 2
        assert snapshot.issue_count == snapshot.issues_total

    def test_broken_links_outrank_metadata_problems_under_the_cap(self, live_vault):
        """A broken link costs the graph an edge; a messy frontmatter block does
        not. Under a cap, the one that changes the graph is shown."""
        snapshot = build_snapshot(live_vault, issue_limit=1)
        assert snapshot.issues[0].detail.startswith("[[")

    def test_a_healthy_vault_has_nothing_to_suggest(self):
        """The list has to be able to go quiet, or it is decoration."""
        healthy = VaultSnapshot(
            files=10,
            indexed_sources=10,
            concepts=10,
            claims=4,
            empty=False,
        )
        assert suggest_next(healthy) == []

    def test_extraction_is_suggested_last_and_says_it_needs_a_model(self):
        """It is the only step that does, and that has to be said before it is
        run, not discovered by a 429."""
        snapshot = VaultSnapshot(
            files=5, indexed_sources=5, concepts=5, links_unresolved=1, empty=False
        )
        actions = suggest_next(snapshot)
        assert actions[-1].command.startswith("forge ingest")
        assert "model" in actions[-1].why

    def test_a_missing_store_does_not_stop_the_vault_half_being_read(self, bare_vault):
        """Half a dashboard beats an exception on a vault nobody has set up."""
        snapshot = build_snapshot(bare_vault)
        assert snapshot.files == 1
        assert snapshot.concepts == 0

    def test_the_hubs_are_the_most_connected_concepts_highest_first(self, live_vault):
        snapshot = build_snapshot(live_vault)

        assert snapshot.hubs, "a graph with edges produced no hubs"
        degrees = [degree for _, degree in snapshot.hubs]
        assert degrees == sorted(degrees, reverse=True)
        assert max(degrees) <= snapshot.max_degree

    def test_a_graph_with_no_edges_has_no_hubs(self, bare_vault):
        assert build_snapshot(bare_vault).hubs == []

    def test_building_the_snapshot_makes_no_model_calls(self, live_vault):
        CALLS.reset()
        build_snapshot(live_vault)
        assert CALLS.count == 0


class TestBrowsing:
    def test_the_concept_list_reports_the_total_as_well_as_the_page(self, live_vault):
        items, total = browse_concepts(live_vault)
        assert items
        assert total >= len(items)

    def test_filtering_narrows_the_list(self, live_vault):
        everything, _ = browse_concepts(live_vault)
        name = everything[0].canonical_name
        filtered, total = browse_concepts(live_vault, name)
        assert total <= len(everything)
        assert any(c.canonical_name == name for c in filtered)

    def test_the_browser_and_the_service_layer_return_the_same_concepts(
        self, live_vault
    ):
        """The dashboard is a fourth face on one service layer, not a fourth
        implementation. If these ever disagree, that has stopped being true."""
        from forge.api import queries
        from forge.cli.snapshot import open_store

        with open_store(live_vault) as store:
            page = queries.list_concepts(store, limit=200)

        items, total = browse_concepts(live_vault)
        assert [c.id for c in items] == [c.id for c in page.items]
        assert total == page.total

    def test_an_empty_search_box_is_not_an_error(self, live_vault):
        """`queries.search_spans` raises on an empty query, which is right for an
        API — a caller sent something malformed. Here it is a box nobody has
        typed in yet, and a dialog saying so would be absurd."""
        assert browse_search(live_vault, "") == []
        assert browse_search(live_vault, "   ") == []

    def test_search_finds_text_that_is_in_the_vault(self, live_vault):
        hits = browse_search(live_vault, "traversal")
        assert hits, "lexical search found nothing in an indexed vault"
        assert all(hit.text for hit in hits)


# -- rendering -------------------------------------------------------------


def _prov(derivation: str, tier: str = "SOURCE_FACT", model_id: str | None = None):
    return ApiProvenance(tier=tier, derivation=derivation, model_id=model_id)


class TestRendering:
    def test_the_overview_names_the_command_to_run(self, bare_vault):
        text = render_overview(build_snapshot(bare_vault))
        assert "forge index" in text

    def test_the_overview_says_the_screen_cost_no_model_call(self, bare_vault):
        assert "No model was called" in render_overview(build_snapshot(bare_vault))

    def test_an_unindexed_vault_gets_an_explanation_not_a_wall_of_zeros(
        self, bare_vault
    ):
        text = render_overview(build_snapshot(bare_vault))
        assert "Nothing has been indexed yet" in text
        assert "never writes to your vault" in text

    def test_a_clean_vault_is_told_so_rather_than_shown_an_empty_list(self):
        clean = VaultSnapshot(files=3, indexed_sources=3, concepts=3, claims=1, empty=False)
        clean.next_actions = suggest_next(clean)
        assert "Nothing outstanding" in render_overview(clean)

    def test_the_overview_names_the_pages_the_vault_hangs_off(self):
        """545 concepts is a number. "Boundary Errors, 163 edges" is a way in."""
        snapshot = VaultSnapshot(concepts=3, empty=False)
        snapshot.hubs = [("Boundary Errors", 163), ("Hash Map", 100)]

        text = render_overview(snapshot)

        assert "MOST CONNECTED" in text
        assert "Boundary Errors" in text
        assert "163" in text

    def test_a_long_reason_wraps_under_its_own_command(self):
        """Without a hanging indent the second line starts at the left margin
        and reads as a new paragraph rather than the rest of a sentence."""
        snapshot = VaultSnapshot(files=1, indexed_sources=1, concepts=1, empty=False)
        snapshot.next_actions = [
            NextAction("forge index", "a reason long enough to need two lines " * 4)
        ]

        body = render_overview(snapshot, width=70)

        reason_lines = [ln for ln in body.split("\n") if "long enough" in ln]
        assert len(reason_lines) > 1, "the reason did not wrap"
        assert all("]    " in ln for ln in reason_lines), "a wrapped line lost its indent"

    def test_a_vault_path_with_brackets_survives_rendering(self, tmp_path):
        """A folder named `[archive]` is legal, and an unescaped bracket is
        swallowed as a style tag — taking the rest of the line with it."""
        snapshot = VaultSnapshot(name="[archive]", path="/home/x/[archive]")
        text = render_overview(snapshot)
        assert r"\[archive]" in text

    def test_the_header_carries_the_llm_call_counter(self):
        """The counter is the deterministic claim, checkable at a glance."""
        assert "0 llm calls" in render_header(VaultSnapshot(name="v"))

    def test_the_tab_bar_marks_the_current_tab_and_carries_counts(self):
        snapshot = VaultSnapshot(concepts=42, issues_total=7)
        bar = render_tabbar("concepts", snapshot)
        assert "Concepts 42" in bar
        assert "Issues 7" in bar
        assert "[b #c8d3f5]Concepts 42" in bar, "the current tab is not distinguished"

    @pytest.mark.parametrize("derivation", ["model", "MODEL", "Model"])
    def test_model_derived_relationships_are_marked_whatever_the_casing(
        self, derivation
    ):
        """The explorer shipped `derivation === "MODEL"` against a value that
        serializes lowercase, so every generated edge rendered as if a human had
        asserted it — the exact distinction the provenance model exists for.
        Casing is pinned here so it cannot come back.
        """
        detail = ConceptDetail(
            concept=ConceptSummary(
                id="c1",
                canonical_name="Heap",
                qualified_name="Heap",
                kind="structure",
                provenance=_prov("deterministic"),
            ),
            relationships=[
                NeighborItem(
                    concept_id="c2",
                    label="Priority Queue",
                    link_type="RELATED_TO",
                    direction="out",
                    provenance=_prov(derivation, "MODEL_INFERENCE", model_id="qwen3:8b"),
                )
            ],
        )

        text = render_concept(detail)

        assert "qwen3:8b" in text, "a model-derived edge does not name its model"
        assert "deterministic" not in text.split("RELATED_TO")[1]

    def test_a_deterministic_concept_says_deterministic_and_names_no_model(self):
        detail = ConceptDetail(
            concept=ConceptSummary(
                id="c1",
                canonical_name="Heap",
                qualified_name="Heap",
                kind="structure",
                provenance=_prov("deterministic"),
            )
        )
        assert "deterministic" in render_concept(detail)

    def test_a_concept_with_no_claims_says_what_would_produce_some(self):
        detail = ConceptDetail(
            concept=ConceptSummary(
                id="c1",
                canonical_name="Heap",
                qualified_name="Heap",
                kind="structure",
                provenance=_prov("deterministic"),
            )
        )
        text = render_concept(detail)
        assert "CLAIMS (0)" in text
        assert "--extract" in text

    def test_an_isolated_concept_says_so_rather_than_showing_an_empty_heading(self):
        """72 of the real vault's 545 concepts are isolated. That is a finding,
        and an empty list under a heading does not read as one."""
        detail = ConceptDetail(
            concept=ConceptSummary(
                id="c1",
                canonical_name="Orphan",
                qualified_name="Orphan",
                kind="concept",
                provenance=_prov("deterministic"),
            )
        )
        assert "isolated" in render_concept(detail)

    def test_the_gap_header_names_the_kinds_it_collapsed(self):
        """A collapsed kind that is not mentioned is a silent omission."""
        report = GapResponse(
            total=600,
            returned=50,
            by_kind={"unevidenced_concept": 600},
            saturated=[
                {
                    "kind": "unevidenced_concept",
                    "count": 600,
                    "population": 601,
                    "detail": "applies to nearly every concept",
                }
            ],
        )
        text = render_gap_header(report)
        assert "unevidenced_concept" in text
        assert "collapsed" in text
        assert "forge gaps --kind unevidenced_concept" in text

    def test_a_hit_with_no_heading_still_names_its_page(self):
        """`L1-L8` locates a span in a file the reader was never told the name
        of. Every frontmatter block in a vault cites like that."""
        from forge.api.models import SearchHit

        hit = SearchHit(
            span_id="s1",
            score=-12.0,
            citation="L1-L8",
            text="---\ntype: problem\n---",
            source_locator="DSA/04_Problems/Stock Span.md",
        )
        assert where_of(hit) == "Stock Span > L1-L8"

    def test_a_hit_that_already_names_its_heading_is_left_alone(self):
        from forge.api.models import SearchHit

        hit = SearchHit(
            span_id="s1",
            score=-12.0,
            citation="Stock Span > Classification",
            text="…",
            source_locator="DSA/04_Problems/Stock Span.md",
        )
        assert where_of(hit) == "Stock Span > Classification"

    def test_fit_truncates_on_one_line(self):
        assert fit("a b\nc", 100) == "a b c"
        assert fit("abcdefgh", 4) == "abc…"

    def test_escape_leaves_ordinary_text_alone(self):
        assert escape("plain text") == "plain text"
        assert escape("[[Heap]]") == r"\[\[Heap]]"


# -- the surfaces know what they cannot launch -----------------------------


class TestRefusal:
    def test_the_shell_may_launch_a_full_screen_app(self):
        """It is line-oriented: `/dash` works and returns to the prompt."""
        assert parse("/dash", ["dash"]).kind is Kind.COMMAND

    def test_a_full_screen_app_refuses_to_launch_another(self):
        """Two Textual applications on one terminal, the second started from a
        worker thread with stdout already redirected, leaves no way back to
        either."""
        action = parse("/dash", ["dash"], refused=(*REFUSED, *FULLSCREEN))
        assert action.kind is Kind.REFUSED
        assert "full-screen" in action.message

    def test_what_cannot_be_run_is_not_offered(self):
        assert "dash" not in visible_names(["dash", "index"], (*REFUSED, *FULLSCREEN))
        assert "dash" in visible_names(["dash", "index"])


def test_without_the_extra_the_dashboard_explains_itself(bare_vault, monkeypatch, capsys):
    """An ImportError three frames down names a module, not a remedy."""
    monkeypatch.setattr("forge.cli.dashboard.textual_available", lambda: False)

    code = run_dashboard(bare_vault, VaultSnapshot())

    assert code == 2
    assert "pip install" in capsys.readouterr().out
    assert "forge-kb[tui]" in INSTALL_HINT


# -- the application -------------------------------------------------------

textual = pytest.importorskip("textual", reason="TUI extra not installed")


class TestApplication:
    """Driven headless through Textual's pilot.

    Bindings, lazy loading and worker hand-off are behaviour; asserting on the
    app's attributes alone would pass with every key unbound.
    """

    def _run(self, settings, body):
        from forge.cli.dashboard import build_dashboard

        snapshot = build_snapshot(settings)

        async def go():
            app = build_dashboard(settings, snapshot)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                return await body(app, pilot)

        return asyncio.run(go())

    @staticmethod
    async def _settle(app, pilot, delay: float = 0.4):
        """Wait out the debounce timer, then the worker it started."""
        await pilot.pause(delay)
        await app.workers.wait_for_complete()
        await pilot.pause()

    def test_it_opens_on_the_overview_not_a_prompt(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import ContentSwitcher, Static

            switcher = app.query_one("#body", ContentSwitcher)
            body_text = app.query_one("#overview-body", Static).content
            return switcher.current, str(body_text)

        current, text = self._run(live_vault, body)
        assert current == "overview"
        assert "VAULT" in text and "GRAPH" in text

    def test_number_keys_switch_tabs(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import ContentSwitcher

            seen = []
            for key in ("2", "4", "1"):
                await pilot.press(key)
                await pilot.pause()
                seen.append(app.query_one("#body", ContentSwitcher).current)
            return seen

        assert self._run(live_vault, body) == ["concepts", "issues", "overview"]

    def test_the_concepts_tab_fills_on_first_visit(self, live_vault):
        """Not at startup: the slowest tab must not decide when the window
        appears."""

        async def body(app, pilot):
            from textual.widgets import DataTable

            before = app.query_one("#concept-table", DataTable).row_count
            await pilot.press("2")
            await self._settle(app, pilot)
            return before, app.query_one("#concept-table", DataTable).row_count

        before, after = self._run(live_vault, body)
        assert before == 0
        assert after > 0

    def test_a_filter_narrows_the_concept_table(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import DataTable, Input

            await pilot.press("2")
            await self._settle(app, pilot)
            everything = app.query_one("#concept-table", DataTable).row_count
            app.query_one("#concept-filter", Input).value = "zzzz-no-such-concept"
            await self._settle(app, pilot)
            return everything, app.query_one("#concept-table", DataTable).row_count

        everything, filtered = self._run(live_vault, body)
        assert everything > 0
        assert filtered == 0

    def test_typing_in_a_box_does_not_trigger_a_binding(self, live_vault):
        """Every binding is a single key, and every one of them is a character
        somebody will type into the filter box. "q" is the dangerous one: a
        search for "queue" that quits after the first letter is not a UI.

        This guards the behaviour, not a mechanism. The bindings are declared
        without `priority` to say the focused widget comes first, but measured
        on Textual 8.2.8 a focused Input wins either way, so the test passes
        under both — it is here to catch the version where that changes.
        """

        async def body(app, pilot):
            from textual.widgets import ContentSwitcher, Input

            await pilot.press("2")
            await self._settle(app, pilot)
            app.query_one("#concept-filter", Input).focus()
            await pilot.pause()
            for key in ("3", "q", "r"):
                await pilot.press(key)
            await pilot.pause()
            return (
                app.query_one("#body", ContentSwitcher).current,
                app.query_one("#concept-filter", Input).value,
                app.is_running,
            )

        current, typed, running = self._run(live_vault, body)
        assert current == "concepts"
        assert typed == "3qr"
        assert running, "a keystroke in the filter box closed the application"

    def test_leaving_the_search_tab_frees_the_number_keys(self, live_vault):
        """The trap this fixes: switching tabs hides the focused widget, and
        Textual hands focus to the next one in the DOM — which was the search
        box. Visiting Search once left every number key going into that box,
        with no visible way out.
        """

        async def body(app, pilot):
            from textual.widgets import ContentSwitcher

            await pilot.press("3")  # search — the box takes focus, deliberately
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            seen = []
            for key in ("4", "5", "1"):
                await pilot.press(key)
                await self._settle(app, pilot, 0.2)
                seen.append(app.query_one("#body", ContentSwitcher).current)
            return seen

        assert self._run(live_vault, body) == ["issues", "gaps", "overview"]

    def test_the_footer_says_what_the_keys_do_while_a_box_has_focus(self, live_vault):
        """Every binding is one character, so inside a text box they all mean
        something else. A footer still advertising them is worse than none."""

        async def body(app, pilot):
            from textual.widgets import Static

            await pilot.press("3")
            await pilot.pause()
            typing = str(app.query_one("#keys", Static).content)
            await pilot.press("escape")
            await pilot.pause()
            return typing, str(app.query_one("#keys", Static).content)

        typing, browsing = self._run(live_vault, body)
        assert "esc" in typing and "leave the box" in typing
        assert "tabs" in browsing

    def test_search_results_are_ranked_not_scored(self, live_vault):
        """SQLite's bm25 is negative and unbounded: "-12.89" in a column headed
        Score tells a reader only that something is broken."""

        async def body(app, pilot):
            from textual.widgets import DataTable, Input

            await pilot.press("3")
            await pilot.pause()
            app.query_one("#search-input", Input).value = "traversal"
            await self._settle(app, pilot, 0.5)
            table = app.query_one("#search-table", DataTable)
            first = table.get_row_at(0)
            return [str(c.label) for c in table.columns.values()], first

        headers, first_row = self._run(live_vault, body)
        assert headers[0] == "#"
        assert first_row[0] == "1"
        assert not any("-1" in str(cell)[:2] for cell in first_row)

    def test_moving_the_cursor_shows_the_concept_detail(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import DataTable, Static

            await pilot.press("2")
            await self._settle(app, pilot)
            app.query_one("#concept-table", DataTable).focus()
            await pilot.press("down")
            await self._settle(app, pilot)
            return str(app.query_one("#concept-detail", Static).content)

        detail = self._run(live_vault, body)
        assert "CLAIMS" in detail
        assert "RELATED" in detail

    def test_search_finds_a_span_and_shows_its_text(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import DataTable, Input, Static

            await pilot.press("3")
            await pilot.pause()
            app.query_one("#search-input", Input).value = "traversal"
            await self._settle(app, pilot, 0.5)
            table = app.query_one("#search-table", DataTable)
            table.focus()
            await pilot.pause()
            return table.row_count, str(app.query_one("#search-detail", Static).content)

        rows, detail = self._run(live_vault, body)
        assert rows > 0
        assert detail, "highlighting a hit showed nothing"

    def test_the_issues_tab_lists_the_broken_links(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import DataTable

            await pilot.press("4")
            await pilot.pause()
            return app.query_one("#issue-table", DataTable).row_count

        assert self._run(live_vault, body) > 0

    def test_a_clean_vault_is_told_it_is_clean(self, bare_vault):
        async def body(app, pilot):
            from textual.widgets import Static

            await pilot.press("4")
            await pilot.pause()
            return str(app.query_one("#issue-status", Static).content)

        assert "No problems found" in self._run(bare_vault, body)

    def test_the_gaps_tab_computes_on_first_visit(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import Static

            await pilot.press("5")
            await self._settle(app, pilot)
            return str(app.query_one("#gap-status", Static).content)

        status = self._run(live_vault, body)
        assert "gap" in status.lower()

    def test_refreshing_re_reads_the_vault(self, live_vault):
        async def body(app, pilot):
            from textual.widgets import Static

            await pilot.press("r")
            await self._settle(app, pilot)
            return str(app.query_one("#titlebar", Static).content)

        assert "files" in self._run(live_vault, body)

    def test_browsing_the_whole_dashboard_makes_no_model_calls(self, live_vault):
        """The product claim, asserted rather than described.

        Every tab, a filter and a search — the entire surface — on a vault with
        no API key configured and no network available.
        """

        async def body(app, pilot):
            from textual.widgets import Input

            for key in ("2", "3", "4", "5", "1"):
                await pilot.press(key)
                await self._settle(app, pilot, 0.2)
            app.query_one("#search-input", Input).value = "traversal"
            await self._settle(app, pilot, 0.5)
            return CALLS.count

        CALLS.reset()
        assert self._run(live_vault, body) == 0
