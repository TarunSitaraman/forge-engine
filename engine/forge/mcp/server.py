"""MCP tools over the same queries the HTTP API serves.

**Every tool is a wrapper around `forge.api.queries`.** Not a re-implementation:
Phase 8's gate is that the two interfaces have identical semantics and that no
capability lives in only one of them, and two implementations of "get a claim
with its evidence" would satisfy that on the day they were written and drift by
the next change. `queries.CAPABILITIES` names each one once, this module builds
a tool per name, and a test compares the tool list against the registry and
against the HTTP layer's operation ids.

**Provenance travels with every result.** The tools return the same pydantic
models the HTTP API does, so the MCP output schema carries the provenance
fields as part of the published contract rather than as a convention an agent
has to discover. An agent that receives a claim can see it was model-derived; a
path result carries the provenance of every edge it crossed.

**A connection per call.** Same reasoning as the HTTP app: `sqlite3`
connections are thread-bound, and a long-lived server has no business holding
one open across calls it does not control the threading of.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ..api import queries
from ..api.models import (
    BeliefResponse,
    ChangeResponse,
    ClaimDetail,
    ClaimPage,
    ConceptDetail,
    ConceptPage,
    EvidenceItem,
    GapResponse,
    NeighborItem,
    PathResponse,
    QuestionDetail,
    QuestionEvidenceItem,
    QuestionSummary,
    RevisionItem,
    SearchHit,
    SourcePage,
    SourceSummary,
    SpanDetail,
    StatsResponse,
    SynthesisSummary,
)
from ..config import Settings
from ..storage import SqliteStore
from . import MCP_SERVER_VERSION

INSTRUCTIONS = """\
Forge is a personal engineering knowledge base with explicit provenance.

Every result carries where it came from. A `provenance.tier` of USER_ASSERTION
means a human asserted it; EXTRACTED_CLAIM, MODEL_INFERENCE and SYNTHESIS mean
a model produced it and it is not evidence. Spans and sources carry a
`trust_tier` instead, because they are quoted source material rather than
derived knowledge. Report that distinction to the user rather than flattening
it: a model's inference and a human's assertion are different kinds of thing.

To go from a claim to what supports it, call `get_claim`, which returns the
evidence chain and each span's verbatim text in one call.

The research tools answer questions a search box cannot. `get_belief` gives
what is held about a concept *and* what disagrees with it; `list_gaps` reports
what the model does not hold, by graph query rather than by guessing;
`get_changes` reads the revision log. None of them calls a model, so none of
them can invent a finding.

There are no confidence scores anywhere, deliberately. A number a model emits
about its own certainty is not a measurement. Judge a claim by its provenance
tier and its evidence, and say so that way.

Everything here is read-only. Knowledge changes only through Forge's proposal
and activation path, which requires a human decision.
"""


def create_server(settings: Settings | None = None, *, db_path: Path | None = None) -> MCPServer:
    """Build the server over the store at `db_path`, defaulting to the settings'."""
    settings = settings or Settings.load()
    database = Path(db_path) if db_path else settings.db_path

    # Once, before serving. Never per call.
    bootstrap = SqliteStore(database)
    bootstrap.initialize()
    bootstrap.close()

    server = MCPServer(
        name="forge",
        title="Forge knowledge base",
        version=MCP_SERVER_VERSION,
        instructions=INSTRUCTIONS,
    )

    @contextmanager
    def opened() -> Iterator[SqliteStore]:
        store = SqliteStore(database)
        try:
            yield store
        finally:
            store.close()

    def guard(fn, *args, **kwargs):
        """Run a query, translating domain errors into `ToolError`.

        `ToolError`'s message reaches the agent; anything else becomes an
        opaque "error executing tool", which is the SDK's intent and the right
        treatment for a genuine bug. A missing concept is not a bug, and the
        agent can act on being told which id was not found.
        """
        try:
            return fn(*args, **kwargs)
        except (queries.NotFound, queries.BadRequest) as exc:
            raise ToolError(str(exc)) from None

    # -- tools -------------------------------------------------------------
    # Named exactly as `queries.CAPABILITIES` names them, so the registry, the
    # HTTP operation ids and this list can be compared directly.

    @server.tool(name="get_stats")
    def get_stats() -> StatsResponse:
        """Counts of what the knowledge store holds, and the shape of its graph.

        Call this first to see whether the store has been populated at all.
        """
        with opened() as store:
            return guard(queries.get_stats, store)

    @server.tool(name="list_concepts")
    def list_concepts(
        q: str | None = None, kind: str | None = None, limit: int = 50, offset: int = 0
    ) -> ConceptPage:
        """List concepts, optionally filtered by a name substring or a kind.

        `q` is a case-insensitive substring of the canonical name. Results are
        alphabetical and paged; `total` is the full match count, not the window.
        """
        with opened() as store:
            return guard(queries.list_concepts, store, q=q, kind=kind, limit=limit, offset=offset)

    @server.tool(name="get_concept")
    def get_concept(concept_id: str) -> ConceptDetail:
        """Everything justifying one concept: origin, claims, relationships.

        Includes how the concept came to exist, which proposal created it, the
        claims made about it, and the concepts it links to with each edge's
        rationale.
        """
        with opened() as store:
            return guard(queries.get_concept, store, concept_id)

    @server.tool(name="list_concept_neighbors")
    def list_concept_neighbors(concept_id: str, limit: int = 50) -> list[NeighborItem]:
        """Concepts one edge away, each with the edge's provenance and rationale.

        The rationale distinguishes a human-authored link from a computed
        similarity; report which it was rather than presenting both as "related".
        """
        with opened() as store:
            return guard(queries.list_concept_neighbors, store, concept_id, limit=limit)

    @server.tool(name="list_claims")
    def list_claims(concept_id: str | None = None, limit: int = 50, offset: int = 0) -> ClaimPage:
        """List claims, optionally only those about one concept."""
        with opened() as store:
            return guard(queries.list_claims, store, concept_id=concept_id, limit=limit, offset=offset)

    @server.tool(name="get_claim")
    def get_claim(claim_id: str) -> ClaimDetail:
        """A claim together with the source text that evidences it.

        This is the call to make before repeating a claim to a user: it returns
        each supporting span verbatim, with its citation and source locator, so
        the claim can be quoted against its source rather than paraphrased.
        """
        with opened() as store:
            return guard(queries.get_claim, store, claim_id)

    @server.tool(name="get_claim_evidence")
    def get_claim_evidence(claim_id: str) -> list[EvidenceItem]:
        """Just the evidence chain for a claim, without the claim itself."""
        with opened() as store:
            return guard(queries.get_claim_evidence, store, claim_id)

    @server.tool(name="get_span")
    def get_span(span_id: str) -> SpanDetail:
        """One span of a source document, verbatim, with its exact location."""
        with opened() as store:
            return guard(queries.get_span, store, span_id)

    @server.tool(name="list_sources")
    def list_sources(limit: int = 50, offset: int = 0) -> SourcePage:
        """Ingested sources, with the trust tier each was assigned."""
        with opened() as store:
            return guard(queries.list_sources, store, limit=limit, offset=offset)

    @server.tool(name="get_source")
    def get_source(source_id: str) -> SourceSummary:
        """One ingested source document, with the trust tier it was assigned.

        A source is the file or paper spans were read from. Its trust tier is
        about the source's authority and is a different question from a claim's
        provenance tier: a faithful extraction from a weak source is still a
        faithful extraction.
        """
        with opened() as store:
            return guard(queries.get_source, store, source_id)

    @server.tool(name="list_entity_revisions")
    def list_entity_revisions(entity_type: str, entity_id: str) -> list[RevisionItem]:
        """How one entity changed over time, oldest first.

        `entity_type` is one of Source, Document, Span, Concept, Claim,
        EvidenceLink, ClaimLink, Proposal, Workflow. Each row's `cause` is what
        triggered the change, which is how "why did this change?" is answered.
        """
        with opened() as store:
            return guard(queries.list_entity_revisions, store, entity_type, entity_id)

    @server.tool(name="list_recent_revisions")
    def list_recent_revisions(limit: int = 50) -> list[RevisionItem]:
        """The most recent changes across every entity."""
        with opened() as store:
            return guard(queries.list_recent_revisions, store, limit=limit)

    @server.tool(name="search_spans")
    def search_spans(q: str, limit: int = 20) -> list[SearchHit]:
        """Lexical full-text search over source spans.

        Deterministic: no model and no embeddings, so it matches words rather
        than meaning. A hit is quoted source text, not generated prose.
        """
        with opened() as store:
            return guard(queries.search_spans, store, q, limit=limit)

    @server.tool(name="find_path")
    def find_path(source: str, target: str, max_depth: int = 4) -> PathResponse:
        """Shortest path between two concepts, within `max_depth` hops.

        `found: false` means "no path within max_depth", which is not the same
        as "no path": the search is bounded and cannot establish the latter.
        Say so that way. Each edge carries who asserted it and why.
        """
        with opened() as store:
            return guard(queries.find_path, store, source, target, max_depth=max_depth)

    # -- research intelligence (Phase 9) -----------------------------------

    @server.tool(name="get_belief")
    def get_belief(concept_id: str) -> BeliefResponse:
        """What is currently believed about a concept, and what disagrees.

        Use this rather than `list_claims` when the user asks what they think
        about something. It separates claims still held from ones disputed or
        superseded, lists the sources the held ones rest on, and reports
        dissent: disputed claims, superseded ones, and conflicts still awaiting
        a human. Report the dissent; a belief presented without it is the
        failure this system exists to prevent.

        There is no confidence score, on purpose. Judge by provenance tier and
        by how many sources support a claim.
        """
        with opened() as store:
            return guard(queries.get_belief, store, concept_id)

    @server.tool(name="list_questions")
    def list_questions(status: str | None = None) -> list[QuestionSummary]:
        """Research questions the user has recorded.

        `status` is `open`, `partially_answered` or `answered`. Omit it for all
        of them. Questions are always asked by a human; Forge does not invent
        them.
        """
        with opened() as store:
            return guard(queries.list_questions, store, status=status)

    @server.tool(name="get_question")
    def get_question(question_id: str) -> QuestionDetail:
        """One research question with the claims that answer it."""
        with opened() as store:
            return guard(queries.get_question, store, question_id)

    @server.tool(name="get_question_evidence")
    def get_question_evidence(question_id: str, limit: int = 10) -> list[QuestionEvidenceItem]:
        """Source spans bearing on a question that are not yet cited in an answer.

        Retrieval scoped by the question rather than by a keyword, and it
        deliberately excludes evidence already used, so what comes back is what
        has not been read into the answer yet.
        """
        with opened() as store:
            return guard(queries.get_question_evidence, store, question_id, limit=limit)

    @server.tool(name="list_gaps")
    def list_gaps(kinds: str | None = None, limit: int = 50) -> GapResponse:
        """What the knowledge model does not hold, by deterministic graph query.

        Every finding is a structural fact: a concept with no claims, an open
        question with no answers, a claim resting on one source, an unresolved
        dispute, a concept with no edges. Whether a gap matters is the user's
        call, so report them and do not act on them.

        `weight` orders the report and is not a probability. `saturated` names
        kinds that describe the whole corpus rather than any one subject; pass
        that kind in `kinds` (comma-separated) to list its instances.
        """
        with opened() as store:
            return guard(queries.list_gaps, store, kinds=kinds, limit=limit)

    @server.tool(name="get_changes")
    def get_changes(days: int = 30) -> ChangeResponse:
        """What changed in the knowledge model over the last `days` days.

        Read from the revision log, so a claim created, disputed and superseded
        inside the window shows as three movements rather than one net diff.
        `causes` says what triggered the changes.
        """
        with opened() as store:
            return guard(queries.get_changes, store, days=days)

    @server.tool(name="list_syntheses")
    def list_syntheses(stale: bool | None = None) -> list[SynthesisSummary]:
        """Generated aggregates over claims, with whether each has gone stale.

        A synthesis is the object most likely to be mistaken for evidence. It
        is not evidence. If `stale` is true the claims it was written from have
        since changed, and `stale_reason` says which; do not repeat a stale
        synthesis without saying so.
        """
        with opened() as store:
            return guard(queries.list_syntheses, store, stale=stale)

    @server.tool(name="get_synthesis")
    def get_synthesis(synthesis_id: str) -> SynthesisSummary:
        """One synthesis by id, with its staleness and the claims behind it."""
        with opened() as store:
            return guard(queries.get_synthesis, store, synthesis_id)

    return server
