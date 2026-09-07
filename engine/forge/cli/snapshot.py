"""What the dashboard shows, computed without a terminal.

Kept apart from the Textual app on purpose. A TUI is awkward to test and easy
to get subtly wrong, and almost everything worth getting right here is the
*data*: what counts as an issue, what to tell someone whose vault is not
indexed yet, which numbers are real. Those are tested directly; the app is a
rendering of this.

**Nothing here calls a model.** Every number comes from the vault on disk or
the derived store, which is the point: this works on the whole corpus, today,
with no API key and no rate limit. Model-derived knowledge shows up when it
exists and is absent without complaint when it does not.

**The browsing functions go through `forge.api.queries`**, the same service
layer the HTTP API and the MCP server wrap. That is deliberate: a fourth
hand-written copy of "get a concept with its claims and neighbours" would agree
with the other three on the day it was written and drift by the next change.
What is *not* shared is the store's lifetime — each call opens and closes its
own connection, because these run on Textual worker threads and a SQLite
connection belongs to the thread that made it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..api.models import (
        ConceptDetail,
        ConceptSummary,
        GapResponse,
        SearchHit,
    )
    from ..storage.sqlite_store import SqliteStore


@dataclass
class NextAction:
    """One thing worth doing next, with the reason and the exact command.

    A dashboard that shows an empty vault and says nothing is a dead end. These
    are ordered, and the first is the one to do.
    """

    command: str
    why: str


@dataclass
class Issue:
    """One concrete problem, with where it is."""

    kind: str
    detail: str
    where: str = ""
    hint: str = ""


@dataclass
class VaultSnapshot:
    """Everything the dashboard needs to render its overview.

    Assembled in one pass so the screen appears at once rather than filling in
    panel by panel.
    """

    name: str = ""
    path: str = ""
    # -- the vault on disk
    files: int = 0
    links_total: int = 0
    links_unresolved: int = 0
    frontmatter_present: int = 0
    frontmatter_valid: int = 0
    duplicate_files: int = 0
    # -- the derived store
    indexed_sources: int = 0
    spans: int = 0
    concepts: int = 0
    edges: int = 0
    #: Concepts with no edges. A fact about the graph's shape, not a defect:
    #: the pages that do the linking in a hub-and-spoke vault are not concepts,
    #: so their links never become edges.
    isolated_concepts: int = 0
    #: Concepts no page in the vault links to, from any page at all. This is
    #: the one that means something is unreachable. `None` until
    #: `forge bootstrap --apply` has counted.
    unreferenced_concepts: int | None = None
    mean_degree: float = 0.0
    max_degree: int = 0
    #: The most-connected concepts, name and degree, highest first. The vault
    #: organises itself around a handful of pages, and naming them is both a
    #: fact about the graph and the most useful place to start reading.
    hubs: list[tuple[str, int]] = field(default_factory=list)
    # -- knowledge, which may legitimately be empty
    claims: int = 0
    questions: int = 0
    open_questions: int = 0
    syntheses: int = 0
    stale_syntheses: int = 0
    proposals_pending: int = 0
    # -- derived views
    #: The first `issue_limit` issues, unresolved links first. `issues_total`
    #: is how many there really are, so a capped list never reads as the whole
    #: truth.
    issues: list[Issue] = field(default_factory=list)
    issues_total: int = 0
    #: Diagnostics above INFO only. A file with no frontmatter at all is not
    #: counted: the parser rates that INFO on purpose, because plenty of
    #: perfectly good notes have none, and calling 263 of them "problems"
    #: would be the dashboard inventing work.
    metadata_issues: int = 0
    next_actions: list[NextAction] = field(default_factory=list)
    #: Set when the store holds nothing yet, so the app can say so plainly
    #: instead of rendering a screen of zeros.
    empty: bool = True

    @property
    def indexed(self) -> bool:
        return self.indexed_sources > 0

    @property
    def seeded(self) -> bool:
        return self.concepts > 0

    @property
    def issue_count(self) -> int:
        """Every problem found, whether or not it fit under the display cap."""
        return self.issues_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "files": self.files,
            "indexed_sources": self.indexed_sources,
            "spans": self.spans,
            "concepts": self.concepts,
            "edges": self.edges,
            "isolated_concepts": self.isolated_concepts,
            "unreferenced_concepts": self.unreferenced_concepts,
            "mean_degree": round(self.mean_degree, 2),
            "max_degree": self.max_degree,
            "hubs": [{"name": n, "degree": d} for n, d in self.hubs],
            "links_total": self.links_total,
            "links_unresolved": self.links_unresolved,
            "frontmatter_present": self.frontmatter_present,
            "frontmatter_valid": self.frontmatter_valid,
            "duplicate_files": self.duplicate_files,
            "claims": self.claims,
            "questions": self.questions,
            "open_questions": self.open_questions,
            "syntheses": self.syntheses,
            "stale_syntheses": self.stale_syntheses,
            "proposals_pending": self.proposals_pending,
            "issues_total": self.issues_total,
            "metadata_issues": self.metadata_issues,
            "issues": [
                {"kind": i.kind, "detail": i.detail, "where": i.where, "hint": i.hint}
                for i in self.issues
            ],
            "next_actions": [
                {"command": a.command, "why": a.why} for a in self.next_actions
            ],
            "empty": self.empty,
        }


@contextmanager
def open_store(settings: Settings) -> Iterator[SqliteStore]:
    """A store connection scoped to one call.

    Every function here opens its own. That looks wasteful next to a long-lived
    handle, and is not: opening SQLite costs well under a millisecond, while a
    connection shared across Textual's worker threads is a
    `ProgrammingError` waiting for the first background refresh. The HTTP API
    made exactly this mistake and fixed it the same way.
    """
    from ..storage.sqlite_store import SqliteStore

    store = SqliteStore(settings.db_path)
    try:
        store.initialize()
        yield store
    finally:
        store.close()


def build_snapshot(
    settings: Settings, *, issue_limit: int = 200, hub_limit: int = 5
) -> VaultSnapshot:
    """Read the vault and the store, once.

    Deliberately tolerant of a half-set-up vault: an unindexed vault, a store
    with sources but no concepts, and a fully seeded one all produce a valid
    snapshot. The alternative is a dashboard that only works once you already
    know how to use Forge.
    """
    from ..corpus.indexer import CorpusIndexer
    from ..domain import ProposalStatus, QuestionStatus
    from ..graph import KnowledgeGraph
    from ..parsing.frontmatter import CODE_DESCRIPTIONS, Severity
    from ..parsing.links import LinkStatus

    snapshot = VaultSnapshot(
        name=settings.vault_path.name or str(settings.vault_path),
        path=str(settings.vault_path),
    )

    index = CorpusIndexer(settings).build_index()
    snapshot.files = index.file_count
    snapshot.duplicate_files = len(index.duplicate_hashes)

    # Two categories, kept apart until the end so broken links — the ones that
    # cost the graph an edge — are never pushed off the list by a long tail of
    # missing frontmatter.
    link_issues: list[Issue] = []
    metadata_issues: list[Issue] = []
    for indexed_file in index.files:
        if indexed_file.frontmatter_present:
            snapshot.frontmatter_present += 1
        if indexed_file.frontmatter_valid:
            snapshot.frontmatter_valid += 1
        for link in indexed_file.links:
            snapshot.links_total += 1
            if link.status in (LinkStatus.RESOLVED, LinkStatus.CASE_MISMATCH):
                continue
            snapshot.links_unresolved += 1
            candidates = getattr(link, "candidates", None) or []
            link_issues.append(
                Issue(
                    kind=link.status.value,
                    detail=f"[[{link.target}]]",
                    where=indexed_file.path,
                    hint=(
                        f"probably {candidates[0]}"
                        if candidates
                        else "no page in the vault matches this"
                    ),
                )
            )
        # An INFO diagnostic is an observation, not a problem, and folding it in
        # would put a four-figure number on a healthy vault.
        repairable = any(getattr(r, "verified", False) for r in indexed_file.repairs)
        for diagnostic in indexed_file.diagnostics:
            if diagnostic.severity is Severity.INFO:
                continue
            metadata_issues.append(
                Issue(
                    kind=diagnostic.code.value,
                    detail=diagnostic.message,
                    where=indexed_file.path,
                    hint=(
                        "a verified repair exists: forge diagnostics frontmatter"
                        if repairable
                        else CODE_DESCRIPTIONS.get(diagnostic.code, "")
                    ),
                )
            )

    with open_store(settings) as store:
        counts = store.counts()
        snapshot.indexed_sources = counts.get("sources", 0)
        snapshot.spans = counts.get("spans", 0)
        snapshot.concepts = counts.get("concepts", 0)
        snapshot.edges = counts.get("claim_links", 0)
        snapshot.claims = counts.get("claims", 0)
        snapshot.questions = counts.get("questions", 0)
        snapshot.syntheses = counts.get("syntheses", 0)
        snapshot.empty = not any(counts.values())

        if snapshot.concepts:
            metrics = KnowledgeGraph(store).metrics()
            snapshot.isolated_concepts = metrics.isolated_nodes
            snapshot.mean_degree = metrics.mean_degree
            snapshot.max_degree = metrics.max_degree
            snapshot.hubs = _hubs(store, limit=hub_limit)
            unreferenced = store.unreferenced_concepts()
            snapshot.unreferenced_concepts = (
                None if unreferenced is None else len(unreferenced)
            )
        if snapshot.questions:
            snapshot.open_questions = len(
                [q for q in store.list_questions() if q.status is not QuestionStatus.ANSWERED]
            )
        if snapshot.syntheses:
            snapshot.stale_syntheses = len(store.list_syntheses(stale=True))
        snapshot.proposals_pending = len(
            store.list_proposals(status=ProposalStatus.PENDING, limit=1000)
        )

    found = link_issues + metadata_issues
    snapshot.metadata_issues = len(metadata_issues)
    snapshot.issues_total = len(found)
    snapshot.issues = found[:issue_limit]
    snapshot.next_actions = suggest_next(snapshot)
    return snapshot


def _hubs(store: SqliteStore, *, limit: int = 5) -> list[tuple[str, int]]:
    """The most-connected concepts, by undirected degree.

    Counted in Python over every edge rather than in SQL, because the whole
    edge table is 2,752 rows on the largest vault this has been run against and
    reads in 58ms — a query worth adding to the store is one that would not.
    """
    degree: dict[str, int] = {}
    for link in store.all_links():
        degree[link.from_id] = degree.get(link.from_id, 0) + 1
        degree[link.to_id] = degree.get(link.to_id, 0) + 1
    if not degree:
        return []
    names = {c.id: c.canonical_name for c in store.list_concepts()}
    ranked = sorted(
        ((names.get(cid, cid), n) for cid, n in degree.items()),
        key=lambda pair: (-pair[1], pair[0].casefold()),
    )
    return ranked[:limit]


def suggest_next(snapshot: VaultSnapshot) -> list[NextAction]:
    """What to do next, in order, given the state of this vault.

    This is the onboarding path. A first-time user should never see a screen of
    zeros with no way forward, and an experienced one should see the same list
    go quiet once there is nothing outstanding.
    """
    actions: list[NextAction] = []

    if not snapshot.indexed:
        actions.append(
            NextAction(
                "forge index",
                f"{snapshot.files} file(s) on disk, none indexed yet. "
                "Deterministic, no model, no network.",
            )
        )
        return actions

    if snapshot.files > snapshot.indexed_sources:
        actions.append(
            NextAction(
                "forge index",
                f"{snapshot.files - snapshot.indexed_sources} file(s) on disk are not "
                "in the store; re-indexing is cheap and skips unchanged files.",
            )
        )

    if not snapshot.seeded:
        actions.append(
            NextAction(
                "forge bootstrap --apply",
                "no concept graph yet. Filenames become concepts and wikilinks "
                "become edges, with zero model calls.",
            )
        )

    if snapshot.links_unresolved:
        actions.append(
            NextAction(
                "forge diagnostics links",
                f"{snapshot.links_unresolved} wikilink(s) point at nothing. Each one "
                "is a connection the graph does not have.",
            )
        )

    if snapshot.metadata_issues:
        actions.append(
            NextAction(
                "forge diagnostics frontmatter",
                f"{snapshot.metadata_issues} frontmatter block(s) do not parse cleanly, "
                "so those files' tags and type are not in the index.",
            )
        )

    if snapshot.proposals_pending:
        actions.append(
            NextAction(
                "forge proposals list",
                f"{snapshot.proposals_pending} proposal(s) waiting on your decision. "
                "Nothing changes until you approve it.",
            )
        )

    if snapshot.stale_syntheses:
        actions.append(
            NextAction(
                "forge gaps",
                f"{snapshot.stale_syntheses} synthesis/syntheses were written from "
                "claims that have since changed.",
            )
        )

    if snapshot.seeded and not snapshot.claims:
        actions.append(
            NextAction(
                "forge ingest <file> --extract",
                "the graph knows your concepts' names but nothing about them. "
                "Extraction needs a model and is the one step that does.",
            )
        )

    return actions


# --------------------------------------------------------------------------
# Browsing. Thin wrappers over the shared service layer; see the module
# docstring for why they are wrappers and not a fourth implementation.
# --------------------------------------------------------------------------


def browse_concepts(
    settings: Settings, query: str = "", *, limit: int = 200
) -> tuple[list[ConceptSummary], int]:
    """Concepts matching a name substring, plus how many matched in total.

    The total matters: a list capped at 200 that says nothing about the cap is
    a list that quietly lies about the size of the vault.
    """
    from ..api import queries

    with open_store(settings) as store:
        page = queries.list_concepts(store, q=query or None, limit=limit)
        return list(page.items), page.total


def browse_concept(settings: Settings, concept_id: str) -> ConceptDetail:
    """One concept with its origin, claims and relationships."""
    from ..api import queries

    with open_store(settings) as store:
        return queries.get_concept(store, concept_id)


def browse_search(settings: Settings, query: str, *, limit: int = 50) -> list[SearchHit]:
    """Lexical span search. Empty and whitespace-only queries return nothing.

    `queries.search_spans` raises `BadRequest` on an empty query, which is right
    for an API — a caller asked for something malformed. Here the empty query is
    just an empty search box, and an error dialog for "you have not typed
    anything yet" is the wrong answer.
    """
    from ..api import queries

    if not query.strip():
        return []
    with open_store(settings) as store:
        return queries.search_spans(store, query, limit=limit)


def browse_gaps(settings: Settings, *, limit: int = 100) -> GapResponse:
    """What the graph does not hold. Deterministic, and the slowest view.

    Slow enough to be worth running off the UI thread: it walks every concept's
    neighbourhood. That is why the dashboard computes it lazily, on the first
    visit to the tab, rather than as part of the opening snapshot.
    """
    from ..api import queries

    with open_store(settings) as store:
        return queries.list_gaps(store, limit=limit)
