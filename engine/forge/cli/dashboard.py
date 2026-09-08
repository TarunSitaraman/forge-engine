"""The Forge dashboard: `forge dash`.

**The front door.** `forge shell` and `forge tui` both open on a prompt, which
answers "what do I type?" with silence. This opens on the vault: what is in it,
what is wrong with it, and what to do next, before anything is typed. Someone
who has never used Forge can run one command and see their own notes described
back to them.

**Nothing here calls a model, and that is the product argument.** Every number
and every list on every tab comes from the vault on disk and the derived store.
It works on the whole corpus, with no API key, no network and no rate limit,
and it works the same on a 6-file vault and a 670-file one. Model-derived
knowledge (claims, syntheses) appears when it exists and is absent without
complaint when it does not, rather than the screen pretending the vault is
broken because nothing has been extracted yet.

**The substance is not in here.** What counts as an issue, what to suggest to a
vault that is not indexed, and how a concept reads once rendered all live in
`snapshot.py` as plain functions over plain data, and are tested there. This
module is layout, key handling, and keeping slow work off the UI thread. The
split is deliberate: a Textual app is awkward to test, and almost nothing worth
getting right needs a terminal to check.

**Textual is an optional extra** (`pip install 'forge-kb[tui]'`), for the same
reason `forge tui` is: indexing a vault must not require a UI framework.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, ClassVar

from ..config import Settings
from .snapshot import (
    VaultSnapshot,
    browse_concept,
    browse_concepts,
    browse_gaps,
    browse_search,
    build_snapshot,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..api.models import ConceptDetail, GapResponse, SearchHit

REQUIRED = "textual"

INSTALL_HINT = (
    "forge dash needs the optional TUI extra.\n"
    "  pip install 'forge-kb[tui]'\n"
    "or, for a pipx install:\n"
    "  pipx inject forge-kb textual\n"
    "\n"
    "`forge status` prints the same overview with nothing extra installed."
)

#: The tabs, in order. The number key that selects a tab is its position, so
#: this tuple is the keyboard layout as well as the tab bar.
TABS: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("concepts", "Concepts"),
    ("search", "Search"),
    ("issues", "Issues"),
    ("gaps", "Gaps"),
)

# One palette, shared with `forge tui` by convention rather than by import,
# they are separate screens and a shared stylesheet would couple their layouts.
BG = "#0b0e14"
PANEL = "#11151c"
TEXT = "#c8d3f5"
MUTED = "#8f9aae"
DIM = "#4d5566"
FAINT = "#3b4252"
RULE = "#1f2430"
ACCENT = "#4d9de0"
WARN = "#d7875f"
BAD = "#d75f5f"
GOOD = "#5f8787"


def textual_available() -> bool:
    """Whether the optional extra is installed."""
    try:  # pragma: no cover - trivial
        import textual  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Rendering. Pure functions over the snapshot and the API models, returning
# Rich markup; no widgets, no Textual import, directly testable.
# --------------------------------------------------------------------------


def escape(text: str) -> str:
    """Vault text is data, not markup.

    A note is entitled to contain `[[Heap]]` or `[dsa/pattern]`, and an
    unescaped bracket is swallowed as a style tag, taking the rest of the line
    with it. Every string that came from the vault goes through here.
    """
    return str(text).replace("[", r"\[")


def fit(text: str, width: int) -> str:
    """Truncate to `width`, with an ellipsis, on one line.

    Cells are truncated rather than wrapped so a row stays a row: a table where
    one long path makes a five-line row is unreadable at a glance, which is the
    only thing a table is for.
    """
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[: max(0, width - 1)] + "…"


def _cell(label: str, value: Any, *, tone: str = TEXT, label_width: int = 15) -> str:
    """One `label      value` pair, padded before it is coloured.

    Padding has to be computed on the bare strings: markup tags are characters
    to `str.format` and invisible on screen, so padding the marked-up string
    lines up nothing.
    """
    shown = f"{value:,}" if isinstance(value, int) else str(value)
    return f"[{DIM}]{label:<{label_width}}[/][{tone}]{shown:>9}[/]"


def _row(left: str, right: str = "") -> str:
    return f"  {left}    {right}"


def render_header(snapshot: VaultSnapshot, *, llm_calls: int = 0) -> str:
    """The title bar: what this vault is, in one line.

    `llm calls` is here for the same reason it is in `forge tui`, the design
    asserts that everything on this screen is deterministic, and a counter that
    sits at zero while you browse the whole vault is that claim, checkable.
    """
    parts = [
        f"[{GOOD}]{snapshot.files:,} files[/]",
        f"[{GOOD}]{snapshot.spans:,} spans[/]",
        f"[{GOOD}]{snapshot.concepts:,} concepts[/]",
        f"[{GOOD}]{snapshot.edges:,} edges[/]",
        f"[{DIM}]{llm_calls} llm calls[/]",
    ]
    return f"[b {TEXT}]# {escape(snapshot.name)}[/]   " + "   ".join(parts)


def render_tabbar(current: str, snapshot: VaultSnapshot) -> str:
    """The tab strip, with counts, so a tab says whether it is worth opening.

    A bare `Issues` tab has to be visited to learn it is empty. `Issues 0` does
    not, and `Issues 42` earns the visit.
    """
    counts = {
        "concepts": snapshot.concepts,
        "issues": snapshot.issues_total,
    }
    cells = []
    for position, (key, label) in enumerate(TABS, start=1):
        count = counts.get(key)
        text = label if count is None else f"{label} {count:,}"
        if key == current:
            cells.append(f"[{ACCENT}]{position}[/] [b {TEXT}]{text}[/]")
        else:
            cells.append(f"[{FAINT}]{position}[/] [{MUTED}]{text}[/]")
    return "   ".join(cells)


def _wrap(text: str, width: int, indent: str = "  ") -> list[str]:
    """Wrap prose to the terminal, with the continuation lines indented.

    Without this a two-line reason starts its second line hard against the left
    margin, which reads as a new paragraph rather than as the rest of a
    sentence. Textual would wrap it for us and would not indent it.
    """
    return textwrap.wrap(
        text,
        width=max(40, width),
        initial_indent=indent,
        subsequent_indent=indent,
    ) or [indent]


def render_overview(snapshot: VaultSnapshot, width: int = 96) -> str:
    """The opening screen: the vault, the graph, what is known, what to do.

    Ordered by what a reader can act on. The counts come first because they are
    what makes it *their* vault rather than a demo, and the next actions come
    last because they are what to do once the counts have been read.
    """
    lines: list[str] = [f"[{DIM}]{escape(snapshot.path)}[/]", ""]

    if snapshot.empty:
        lines.append(f"[{WARN}]Nothing has been indexed yet.[/]")
        lines += [
            f"[{MUTED}]{line}[/]"
            for line in _wrap(
                "Indexing reads every Markdown file, resolves its wikilinks and "
                "records what it found. It makes no model calls and touches no "
                "network, and it never writes to your vault; everything derived "
                "lives in .forge/ and can be deleted and rebuilt.",
                width,
                indent="",
            )
        ]
        lines.append("")

    unresolved_tone = WARN if snapshot.links_unresolved else GOOD
    lines += [
        f"[b {MUTED}]VAULT[/]",
        _row(
            _cell("files", snapshot.files),
            _cell("wikilinks", snapshot.links_total),
        ),
        _row(
            _cell(
                "frontmatter",
                f"{snapshot.frontmatter_valid:,}/{snapshot.files:,}",
                tone=MUTED,
            ),
            _cell("unresolved", snapshot.links_unresolved, tone=unresolved_tone),
        ),
        _row(
            _cell("indexed", snapshot.indexed_sources),
            _cell(
                "duplicates",
                snapshot.duplicate_files,
                tone=WARN if snapshot.duplicate_files else GOOD,
            ),
        ),
        "",
        f"[b {MUTED}]GRAPH[/]",
        _row(
            _cell("concepts", snapshot.concepts),
            _cell("spans", snapshot.spans),
        ),
        _row(
            _cell("edges", snapshot.edges),
            # "no edges" rather than "isolated", and muted rather than warned:
            # in a vault whose `_index.md` hubs do the linking, a concept with
            # no edges is usually linked perfectly well from a page that is not
            # a concept. Calling that isolated was wrong on the real corpus by
            # 71 out of 72.
            _cell("no edges", snapshot.isolated_concepts, tone=MUTED),
        ),
        _row(
            _cell("mean degree", f"{snapshot.mean_degree:.1f}", tone=MUTED),
            _cell("max degree", snapshot.max_degree, tone=MUTED),
        ),
        _row(
            _cell(
                "unreferenced",
                "not counted"
                if snapshot.unreferenced_concepts is None
                else snapshot.unreferenced_concepts,
                tone=(
                    DIM
                    if snapshot.unreferenced_concepts is None
                    else WARN
                    if snapshot.unreferenced_concepts
                    else GOOD
                ),
            ),
        ),
        "",
    ]

    if snapshot.hubs:
        # Not decoration: on a vault of 545 concepts these five are what the
        # rest hangs off, and they are the natural place to start reading.
        lines.append(f"[b {MUTED}]MOST CONNECTED[/]")
        for name, degree in snapshot.hubs:
            lines.append(
                f"  [{TEXT}]{escape(fit(name, 40)):<42}[/]"
                f"[{DIM}]{degree:>4} edges[/]"
            )
        lines.append("")

    lines += [
        f"[b {MUTED}]KNOWLEDGE[/]",
        _row(
            _cell("claims", snapshot.claims),
            _cell(
                "questions",
                f"{snapshot.open_questions:,}/{snapshot.questions:,} open",
                tone=MUTED,
            ),
        ),
        _row(
            _cell("syntheses", snapshot.syntheses),
            _cell(
                "stale",
                snapshot.stale_syntheses,
                tone=WARN if snapshot.stale_syntheses else GOOD,
            ),
        ),
        _row(
            _cell(
                "proposals",
                f"{snapshot.proposals_pending:,} pending",
                tone=WARN if snapshot.proposals_pending else MUTED,
            ),
        ),
        "",
    ]

    if snapshot.next_actions:
        lines.append(f"[b {MUTED}]NEXT[/]")
        for action in snapshot.next_actions:
            lines.append(f"  [{ACCENT}]{escape(action.command)}[/]")
            lines += [
                f"[{DIM}]{escape(line)}[/]"
                for line in _wrap(action.why, width - 4, indent="    ")
            ]
        lines.append("")
    else:
        settled = (
            f"  [{GOOD}]Nothing outstanding.[/] "
            f"[{DIM}]Every file is indexed, every link resolves.[/]"
        )
        lines += [f"[b {MUTED}]NEXT[/]", settled, ""]

    lines.append(
        f"[{FAINT}]Everything on this screen was computed from your files and the "
        f"derived store. No model was called.[/]"
    )
    return "\n".join(lines)


def _provenance_note(provenance: Any) -> str:
    """`deterministic` or the model that said it: never both, never neither.

    The distinction is the whole point of the provenance model, and a UI that
    renders a bootstrapped edge and an inferred one identically throws away the
    one thing the store is careful about.
    """
    if provenance is None:
        return f"[{DIM}]no provenance[/]"
    derivation = str(getattr(provenance, "derivation", "")).lower()
    tier = str(getattr(provenance, "tier", ""))
    model = getattr(provenance, "model_id", None)
    if derivation == "model":
        return f"[{WARN}]{escape(tier)}[/] [{DIM}]{escape(model or 'model')}[/]"
    if derivation == "human":
        return f"[{ACCENT}]{escape(tier)}[/] [{DIM}]you[/]"
    return f"[{GOOD}]{escape(tier)}[/] [{DIM}]deterministic[/]"


def render_concept(detail: ConceptDetail) -> str:
    """One concept: where it came from, what is claimed about it, what it touches.

    Claims are shown even when there are none, with the command that would
    produce some. An empty section that explains itself is worth more than a
    section that disappears, which just looks like the feature is missing.
    """
    concept = detail.concept
    title = (
        f"[b {TEXT}]{escape(concept.canonical_name)}[/]  "
        f"[{DIM}]{escape(concept.kind)}[/]  {_provenance_note(concept.provenance)}"
    )
    lines = [title]
    if concept.vault_path:
        lines.append(f"[{DIM}]{escape(concept.vault_path)}[/]")
    if concept.aliases:
        lines.append(
            f"[{DIM}]also: {escape(', '.join(concept.aliases))}[/]"
        )
    lines.append("")

    lines.append(f"[b {MUTED}]CLAIMS ({len(detail.claims)})[/]")
    if detail.claims:
        for claim in detail.claims:
            marker = BAD if (claim.status or "").upper() == "DISPUTED" else TEXT
            lines.append(f"  [{marker}]{escape(fit(claim.statement, 300))}[/]")
            lines.append(
                f"    {_provenance_note(claim.provenance)}  "
                f"[{DIM}]{claim.evidence_count} evidence[/]"
            )
    else:
        lines.append(
            f"  [{DIM}]nothing extracted yet, "
            f"forge ingest <path> --extract reads this page with a model[/]"
        )
    lines.append("")

    lines.append(f"[b {MUTED}]RELATED ({len(detail.relationships)})[/]")
    if detail.relationships:
        for neighbor in detail.relationships:
            arrow = "→" if neighbor.direction == "out" else "←"
            label = neighbor.label or neighbor.concept_id
            lines.append(
                f"  [{DIM}]{arrow}[/] [{MUTED}]{escape(neighbor.link_type):<14}[/] "
                f"[{TEXT}]{escape(fit(label, 40))}[/]  "
                f"{_provenance_note(neighbor.provenance)}"
            )
    else:
        lines.append(
            f"  [{WARN}]isolated[/] [{DIM}]no page in the vault links to this one, "
            f"and it links to none[/]"
        )

    if detail.origin_spans:
        lines += ["", f"[b {MUTED}]FIRST SEEN[/]"]
        span = detail.origin_spans[0]
        lines.append(f"  [{DIM}]{escape(span.citation)}[/]")
        lines.append(f"  [{MUTED}]{escape(fit(span.text, 400))}[/]")
    return "\n".join(lines)


def where_of(hit: SearchHit) -> str:
    """Where a hit is, in a form that means something on its own.

    A span with no heading above it cites as `L1-L8`, which locates it in a file
    the reader has not been told the name of. Frontmatter blocks are all like
    this, and they are a good share of any vault's spans.
    """
    citation = hit.citation or hit.span_id
    if ">" in citation or not hit.source_locator:
        return citation
    stem = hit.source_locator.rsplit("/", 1)[-1].removesuffix(".md")
    return f"{stem} > {citation}"


def render_hit(hit: SearchHit) -> str:
    """One search result, in full, with where it came from."""
    header = f"[b {TEXT}]{escape(where_of(hit))}[/]"
    tier = f"  [{DIM}]{escape(hit.trust_tier or 'unverified')}[/]"
    source = (
        f"[{DIM}]{escape(hit.source_locator)}[/]\n" if hit.source_locator else ""
    )
    return f"{header}{tier}\n{source}\n[{MUTED}]{escape(hit.text)}[/]"


def render_gap_header(report: GapResponse) -> str:
    """The gap summary, including the kinds that were deliberately collapsed.

    A kind that applies to half the corpus is a property of the corpus, not a
    to-do list, and listing 300 of them buries the ones worth acting on. Saying
    so is the difference between a filter and a silent omission.
    """
    by_kind = ", ".join(f"{k} {v}" for k, v in sorted(report.by_kind.items()))
    lines = [
        f"[{MUTED}]{report.total:,} gap(s)[/]  [{DIM}]{escape(by_kind)}[/]"
        if by_kind
        else f"[{GOOD}]no gaps found[/]"
    ]
    for saturated in report.saturated:
        lines.append(
            f"[{DIM}]{escape(saturated.kind)}: {saturated.count:,} of "
            f"{saturated.population:,}, collapsed, "
            f"forge gaps --kind {escape(saturated.kind)} lists them[/]"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The application.
# --------------------------------------------------------------------------


def build_dashboard(settings: Settings, snapshot: VaultSnapshot):
    """Construct the Textual application.

    A factory rather than a module-level class, so importing this module does
    not require Textual: `forge dash` can then fail with an instruction naming
    the extra instead of an ImportError from three frames down.
    """
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import ContentSwitcher, DataTable, Input, Static

    class ForgeDashboard(App):
        # Names the terminal tab and window. Without it Textual uses the
        # class name, so the tab reads "ForgeDashboard" while the command
        # the user typed was `forge dash`.
        TITLE = "forge dash"

        CSS = f"""
        Screen {{ background: {BG}; }}

        #titlebar {{ height: 1; padding: 0 1; color: {TEXT}; background: {BG}; }}
        #tabbar   {{ height: 1; padding: 0 1; background: {PANEL}; }}
        #body     {{ height: 1fr; background: {BG}; }}
        #keys     {{ height: 1; padding: 0 1; color: {FAINT}; background: {PANEL}; }}

        .status {{ height: auto; max-height: 3; padding: 0 1; color: {DIM}; }}

        #overview {{ padding: 1 1; background: {BG}; scrollbar-size-vertical: 1; }}

        Input {{
            height: 1;
            background: {PANEL};
            border: none;
            padding: 0 1;
            color: {TEXT};
        }}
        Input:focus {{ border: none; }}

        DataTable {{
            height: 1fr;
            background: {BG};
            scrollbar-size-vertical: 1;
            scrollbar-size-horizontal: 1;
        }}
        DataTable > .datatable--header {{ background: {BG}; color: {MUTED}; }}
        DataTable > .datatable--cursor {{ background: {RULE}; color: {TEXT}; }}

        .detail {{
            height: 45%;
            border-top: solid {RULE};
            padding: 0 1;
            background: {BG};
            scrollbar-size-vertical: 1;
        }}
        """

        # Number keys select tabs, and single letters act. Deliberately not
        # `priority`: the focused widget gets the key first, so typing "q" into
        # the search box types a q rather than quitting mid-word. Measured on
        # Textual 8.2.8 a focused Input wins even against a priority binding, so
        # this expresses the intent rather than being the thing that enforces
        # it: `test_typing_in_a_box_does_not_trigger_a_binding` guards the
        # behaviour, and passes under either choice on this version.
        BINDINGS: ClassVar[list] = [
            Binding("q", "quit", "quit"),
            Binding("ctrl+c", "quit", "", show=False),
            Binding("1", "show('overview')", "overview"),
            Binding("2", "show('concepts')", "concepts"),
            Binding("3", "show('search')", "search"),
            Binding("4", "show('issues')", "issues"),
            Binding("5", "show('gaps')", "gaps"),
            Binding("r", "refresh", "refresh"),
            Binding("slash", "find", "find"),
            Binding("escape", "leave_input", "", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.snapshot = snapshot
            self.current = "overview"
            #: Tabs whose contents have been loaded. Concepts and gaps are
            #: filled on first visit rather than at startup, so the window
            #: appears immediately on a large vault instead of after the
            #: slowest query on the slowest tab.
            self._loaded: set[str] = set()
            self._hits: dict[str, Any] = {}
            self._debounce: Any = None

        # -- layout --------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Static(render_header(self.snapshot), id="titlebar")
            yield Static(render_tabbar(self.current, self.snapshot), id="tabbar")
            with ContentSwitcher(initial="overview", id="body"):
                with VerticalScroll(id="overview"):
                    yield Static(render_overview(self.snapshot), id="overview-body")
                with Vertical(id="concepts"):
                    yield Input(
                        placeholder="filter concepts by name", id="concept-filter"
                    )
                    yield Static("", id="concept-status", classes="status")
                    yield DataTable(id="concept-table", cursor_type="row")
                    with VerticalScroll(id="concept-detail-wrap", classes="detail"):
                        yield Static("", id="concept-detail")
                with Vertical(id="search"):
                    yield Input(
                        placeholder="search the text of every indexed page",
                        id="search-input",
                    )
                    yield Static("", id="search-status", classes="status")
                    yield DataTable(id="search-table", cursor_type="row")
                    with VerticalScroll(id="search-detail-wrap", classes="detail"):
                        yield Static("", id="search-detail")
                with Vertical(id="issues"):
                    yield Static("", id="issue-status", classes="status")
                    yield DataTable(id="issue-table", cursor_type="row")
                with Vertical(id="gaps"):
                    yield Static("", id="gap-status", classes="status")
                    yield DataTable(id="gap-table", cursor_type="row")
            yield Static(self._keys(), id="keys")

        def _keys(self) -> str:
            """The footer, which changes with focus.

            While a text box has focus every binding is a character being typed
            "q" in the middle of "queue" must not quit, so the keys really
            are different, and a footer that claims otherwise is worse than no
            footer.
            """
            if isinstance(self.focused, Input):
                return (
                    f"[{FAINT}]type[/] to search    "
                    f"[{FAINT}]esc[/] leave the box, then 1-5 for tabs"
                )
            return (
                f"[{FAINT}]1-5[/] tabs    "
                f"[{FAINT}]/[/] find    "
                f"[{FAINT}]↑↓[/] browse    "
                f"[{FAINT}]esc[/] back    "
                f"[{FAINT}]r[/] refresh    "
                f"[{FAINT}]q[/] quit"
            )

        # `focused` is a reactive on the Screen, not on the App, so a
        # `watch_focused` here is never called; it looks like it works and
        # silently does nothing. These events do bubble to the App.
        def on_descendant_focus(self, event: Any) -> None:
            self._refresh_keys()

        def on_descendant_blur(self, event: Any) -> None:
            # Blur arrives before the new focus is set, so read it afterwards.
            self.call_after_refresh(self._refresh_keys)

        def _refresh_keys(self) -> None:
            from textual.css.query import NoMatches

            try:
                self.query_one("#keys", Static).update(self._keys())
            except NoMatches:  # before compose has run
                pass

        def on_mount(self) -> None:
            self.query_one("#concept-table", DataTable).add_columns(
                "Concept", "Kind", "Origin", "Page"
            )
            # Rank, not the score itself: SQLite's bm25 is negative and
            # unbounded, and "-12.89" tells a reader nothing except that
            # something is wrong. The order is the part that means something.
            self.query_one("#search-table", DataTable).add_columns(
                "#", "Where", "Text"
            )
            self.query_one("#issue-table", DataTable).add_columns(
                "Kind", "What", "File", "Hint"
            )
            self.query_one("#gap-table", DataTable).add_columns(
                "Kind", "Subject", "Why", "Weight"
            )
            self._fill_issues()

        # -- tabs ----------------------------------------------------------

        #: Where focus goes when a tab opens. Switching tabs hides the focused
        #: widget, and Textual then hands focus to whatever is next in the DOM,
        #: which was the search box, three tabs away. The effect was that
        #: visiting Search once left every number key going into that box, and
        #: the only way out was a key nobody had been told about. Focusing the
        #: new tab's own widget is what stops that.
        FOCUS: ClassVar[dict[str, str]] = {
            "overview": "#overview",
            "concepts": "#concept-table",
            "search": "#search-input",
            "issues": "#issue-table",
            "gaps": "#gap-table",
        }

        def action_show(self, tab: str) -> None:
            self.current = tab
            self.query_one("#body", ContentSwitcher).current = tab
            self.query_one("#tabbar", Static).update(
                render_tabbar(tab, self.snapshot)
            )
            if tab == "concepts" and "concepts" not in self._loaded:
                self._loaded.add("concepts")
                self._load_concepts("")
            if tab == "gaps" and "gaps" not in self._loaded:
                self._loaded.add("gaps")
                self._load_gaps()
            self.query_one(self.FOCUS[tab]).focus()

        def action_find(self) -> None:
            """`/` goes where finding happens, whichever tab that is."""
            if self.current == "concepts":
                self.query_one("#concept-filter", Input).focus()
            else:
                self.action_show("search")

        def action_leave_input(self) -> None:
            """Escape hands focus back to the list, so the number keys work again.

            The one way out of a text box, and the reason the footer changes to
            say so while one has focus.
            """
            table = {"concepts": "#concept-table", "search": "#search-table"}.get(
                self.current
            )
            if table:
                self.query_one(table, DataTable).focus()

        # -- refresh -------------------------------------------------------

        def action_refresh(self) -> None:
            self.query_one("#titlebar", Static).update(
                f"[{WARN}]re-reading the vault…[/]"
            )
            self._rebuild()

        @work(thread=True, exclusive=True, group="snapshot")
        def _rebuild(self) -> None:
            fresh = build_snapshot(settings)
            self.call_from_thread(self._apply_snapshot, fresh)

        def _apply_snapshot(self, fresh: VaultSnapshot) -> None:
            self.snapshot = fresh
            self.query_one("#titlebar", Static).update(render_header(fresh))
            self.query_one("#tabbar", Static).update(
                render_tabbar(self.current, fresh)
            )
            self.query_one("#overview-body", Static).update(render_overview(fresh))
            self._fill_issues()
            # A rebuilt store means the loaded tabs are stale. Dropping them
            # from the set makes the next visit reload rather than show numbers
            # that disagree with the header.
            self._loaded.discard("gaps")
            if "concepts" in self._loaded:
                self._load_concepts(self.query_one("#concept-filter", Input).value)

        # -- concepts ------------------------------------------------------

        def _load_concepts(self, query: str) -> None:
            self.query_one("#concept-status", Static).update(f"[{DIM}]loading…[/]")
            self._concepts_worker(query)

        @work(thread=True, exclusive=True, group="concepts")
        def _concepts_worker(self, query: str) -> None:
            items, total = browse_concepts(settings, query)
            self.call_from_thread(self._show_concepts, items, total, query)

        def _show_concepts(self, items: list[Any], total: int, query: str) -> None:
            table = self.query_one("#concept-table", DataTable)
            table.clear()
            for concept in items:
                table.add_row(
                    escape(fit(concept.canonical_name, 38)),
                    escape(fit(concept.kind, 16)),
                    escape(str(concept.provenance.derivation)),
                    escape(fit(concept.vault_path or "—", 48)),
                    key=concept.id,
                )
            self.query_one("#concept-status", Static).update(
                self._list_status(len(items), total, query, noun="concept")
            )
            self.query_one("#concept-detail", Static).update(
                f"[{DIM}]select a concept to see its claims and relationships[/]"
            )

        def _list_status(self, shown: int, total: int, query: str, *, noun: str) -> str:
            if not total:
                if query:
                    return f"[{WARN}]nothing matches {escape(repr(query))}[/]"
                return (
                    f"[{WARN}]no {noun}s yet[/] "
                    f"[{DIM}]forge bootstrap --apply builds the graph from your "
                    f"filenames and wikilinks, with no model calls[/]"
                )
            if shown < total:
                return f"[{DIM}]showing {shown:,} of {total:,} {noun}s[/]"
            return f"[{DIM}]{total:,} {noun}s[/]"

        # -- search --------------------------------------------------------

        def on_input_changed(self, event: Input.Changed) -> None:
            """Debounce: a query runs when typing pauses, not per keystroke.

            Span search touches the source of every hit, which is a hundred-odd
            queries, fast, but not fast enough to run between two keystrokes
            without the box going gummy.
            """
            if self._debounce is not None:
                self._debounce.stop()
            value = event.value
            if event.input.id == "search-input":
                self._debounce = self.set_timer(0.25, lambda: self._load_search(value))
            elif event.input.id == "concept-filter":
                self._debounce = self.set_timer(0.15, lambda: self._load_concepts(value))

        def _load_search(self, query: str) -> None:
            if not query.strip():
                self.query_one("#search-table", DataTable).clear()
                self.query_one("#search-status", Static).update(
                    f"[{DIM}]lexical search over {self.snapshot.spans:,} spans. "
                    f"Deterministic: no embeddings, no model.[/]"
                )
                return
            self.query_one("#search-status", Static).update(f"[{DIM}]searching…[/]")
            self._search_worker(query)

        @work(thread=True, exclusive=True, group="search")
        def _search_worker(self, query: str) -> None:
            hits = browse_search(settings, query)
            self.call_from_thread(self._show_search, hits, query)

        def _show_search(self, hits: list[Any], query: str) -> None:
            table = self.query_one("#search-table", DataTable)
            table.clear()
            self._hits = {hit.span_id: hit for hit in hits}
            for rank, hit in enumerate(hits, start=1):
                table.add_row(
                    str(rank),
                    escape(fit(where_of(hit), 44)),
                    escape(fit(hit.text, 90)),
                    key=hit.span_id,
                )
            self.query_one("#search-status", Static).update(
                f"[{DIM}]{len(hits):,} span(s) matching {escape(repr(query))}[/]"
                if hits
                else f"[{WARN}]nothing in the vault matches {escape(repr(query))}[/]"
            )
            self.query_one("#search-detail", Static).update("")

        # -- issues and gaps -----------------------------------------------

        def _fill_issues(self) -> None:
            table = self.query_one("#issue-table", DataTable)
            table.clear()
            for issue in self.snapshot.issues:
                table.add_row(
                    escape(fit(issue.kind, 18)),
                    escape(fit(issue.detail, 44)),
                    escape(fit(issue.where, 42)),
                    escape(fit(issue.hint, 46)),
                )
            total = self.snapshot.issues_total
            shown = len(self.snapshot.issues)
            if not total:
                status = (
                    f"[{GOOD}]No problems found.[/] "
                    f"[{DIM}]Every wikilink resolves and every frontmatter block "
                    f"parses.[/]"
                )
            elif shown < total:
                status = f"[{DIM}]showing {shown:,} of {total:,} problems[/]"
            else:
                status = f"[{DIM}]{total:,} problem(s)[/]"
            self.query_one("#issue-status", Static).update(status)

        def _load_gaps(self) -> None:
            self.query_one("#gap-status", Static).update(
                f"[{DIM}]walking the graph…[/]"
            )
            self._gaps_worker()

        @work(thread=True, exclusive=True, group="gaps")
        def _gaps_worker(self) -> None:
            report = browse_gaps(settings)
            self.call_from_thread(self._show_gaps, report)

        def _show_gaps(self, report: Any) -> None:
            table = self.query_one("#gap-table", DataTable)
            table.clear()
            # Gaps of one kind carry one explanation, so printing it on every
            # row spends sixty columns saying the same sentence 72 times. It is
            # shown when the kind changes, which reads as a grouped list.
            previous = None
            for gap in report.gaps:
                repeat = gap.kind == previous
                table.add_row(
                    escape(fit(gap.kind, 24)),
                    escape(fit(gap.subject_label, 44)),
                    "" if repeat else escape(fit(gap.detail, 58)),
                    f"{gap.weight:.2f}",
                )
                previous = gap.kind
            self.query_one("#gap-status", Static).update(render_gap_header(report))

        # -- selection -----------------------------------------------------

        def on_data_table_row_highlighted(self, event: Any) -> None:
            """Detail follows the cursor, so browsing needs no second keystroke."""
            key = getattr(event.row_key, "value", None)
            if key is None:
                return
            if event.data_table.id == "concept-table":
                self._detail_worker(key)
            elif event.data_table.id == "search-table":
                hit = self._hits.get(key)
                if hit is not None:
                    self.query_one("#search-detail", Static).update(render_hit(hit))

        @work(thread=True, exclusive=True, group="detail")
        def _detail_worker(self, concept_id: str) -> None:
            try:
                detail = browse_concept(settings, concept_id)
            except Exception as exc:  # noqa: BLE001 - a missing concept is a message, not a crash
                self.call_from_thread(
                    self.query_one("#concept-detail", Static).update,
                    f"[{BAD}]{escape(f'{type(exc).__name__}: {exc}')}[/]",
                )
                return
            self.call_from_thread(
                self.query_one("#concept-detail", Static).update,
                render_concept(detail),
            )

    return ForgeDashboard()


def run_dashboard(settings: Settings, snapshot: VaultSnapshot | None = None) -> int:
    """Entry point. Returns a process exit code.

    The snapshot is built before the app starts rather than inside `on_mount`:
    on a 670-file vault it takes about two seconds, and two seconds of an empty
    window is worse than two seconds of a terminal that has not changed yet.
    """
    if not textual_available():
        print(INSTALL_HINT)
        return 2
    build_dashboard(settings, snapshot or build_snapshot(settings)).run()
    return 0
