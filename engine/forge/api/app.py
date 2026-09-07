"""FastAPI application: read-only views over the knowledge store.

**The routes are wrappers, not implementations.** Every one calls a function in
`forge.api.queries`, which the MCP server calls too. That is what Phase 8's
"no capability lives only in one interface" means in practice: there is one
implementation, and adding a capability to one interface but not the other
fails a test rather than shipping.

Each route's `operation_id` is the capability's name from
`queries.CAPABILITIES`, so the mapping is in the OpenAPI schema and can be
compared against the MCP tool list without a private registry of its own.

**Every route is a GET, and that is enforced rather than observed.** A test
walks the OpenAPI schema and fails on any other method. Knowledge changes
through proposal and activation, which require a human decision; an HTTP write
path would be a second way in without that gate.

**Zero model calls.** The API is a view over what is already stored. `/stats`
publishes the process call counter so a client can see that, and a test asserts
it stays at zero across every route.
"""

from __future__ import annotations

from pathlib import Path as FilePath
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from ..config import Settings
from ..storage import SqliteStore
from . import API_VERSION, queries
from .models import (
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
from .queries import MAX_PAGE_SIZE, BadRequest, NotFound

STATIC_DIR = FilePath(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None, *, db_path: FilePath | None = None) -> FastAPI:
    """Build the app over the store at `db_path`, defaulting to the settings'.

    **A connection per request, not one shared connection.** `sqlite3`
    connections are bound to the thread that created them, and FastAPI runs
    sync endpoints in a threadpool, so a single shared store raises
    `ProgrammingError: SQLite objects created in a thread can only be used in
    that same thread` on the first route that touches the database from a
    worker. Found on the first smoke run of this app, on `/stats`.

    Opening per request is the honest fix rather than passing
    `check_same_thread=False`, which silences the check without making the
    connection safe to share. A local SQLite connection costs a file open and
    two pragmas; the schema is initialized once at startup, never per request.

    `db_path` is injectable so tests point the app at a store they seeded
    themselves rather than whatever happens to be on disk.
    """
    settings = settings or Settings.load()
    database = FilePath(db_path) if db_path else settings.db_path

    app = FastAPI(
        title="Forge",
        version=API_VERSION,
        description=(
            "Read-only views over a Forge knowledge store. Every route is a GET "
            "and makes zero model calls; knowledge changes only through proposal "
            "and activation, which require a human decision."
        ),
    )
    app.state.settings = settings
    app.state.db_path = database

    # Once, in this thread, before anything is served. Running it per request
    # would re-check migrations on every call.
    _bootstrap = SqliteStore(database)
    _bootstrap.initialize()
    _bootstrap.close()

    def get_store():
        """One connection per request, closed when the request ends."""
        store = SqliteStore(app.state.db_path)
        try:
            yield store
        finally:
            store.close()

    @app.exception_handler(NotFound)
    def _not_found(request, exc: NotFound):  # pragma: no cover - exercised via routes
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(BadRequest)
    def _bad_request(request, exc: BadRequest):  # pragma: no cover - via routes
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # -- meta --------------------------------------------------------------

    @app.get("/health", tags=["meta"], operation_id="health")
    def health() -> dict[str, Any]:
        """Liveness. Interface plumbing, deliberately not a knowledge capability."""
        return {"ok": True, "api_version": API_VERSION, "vault": str(settings.vault_path)}

    @app.get("/stats", response_model=StatsResponse, tags=["meta"], operation_id="get_stats")
    def get_stats(store: SqliteStore = Depends(get_store)):
        return queries.get_stats(store)

    # -- concepts ----------------------------------------------------------

    @app.get(
        "/concepts", response_model=ConceptPage, tags=["concepts"], operation_id="list_concepts"
    )
    def list_concepts(
        store: SqliteStore = Depends(get_store),
        q: str | None = Query(None, description="Case-insensitive substring of the name."),
        kind: str | None = Query(None),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        return queries.list_concepts(store, q=q, kind=kind, limit=limit, offset=offset)

    @app.get(
        "/concepts/{concept_id}",
        response_model=ConceptDetail,
        tags=["concepts"],
        operation_id="get_concept",
    )
    def get_concept(concept_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_concept(store, concept_id)

    @app.get(
        "/concepts/{concept_id}/neighbors",
        response_model=list[NeighborItem],
        tags=["concepts"],
        operation_id="list_concept_neighbors",
    )
    def list_concept_neighbors(
        concept_id: str,
        store: SqliteStore = Depends(get_store),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    ):
        return queries.list_concept_neighbors(store, concept_id, limit=limit)

    # -- claims and their evidence ----------------------------------------

    @app.get("/claims", response_model=ClaimPage, tags=["claims"], operation_id="list_claims")
    def list_claims(
        store: SqliteStore = Depends(get_store),
        concept_id: str | None = Query(None),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        return queries.list_claims(store, concept_id=concept_id, limit=limit, offset=offset)

    @app.get(
        "/claims/{claim_id}",
        response_model=ClaimDetail,
        tags=["claims"],
        operation_id="get_claim",
    )
    def get_claim(claim_id: str, store: SqliteStore = Depends(get_store)):
        """Claim plus its full evidence chain.

        The Phase 6 gate is reaching the exact source span from a claim in
        **one interaction**, so the span text and its source locator are in
        this response rather than behind a second request.
        """
        return queries.get_claim(store, claim_id)

    @app.get(
        "/claims/{claim_id}/evidence",
        response_model=list[EvidenceItem],
        tags=["claims"],
        operation_id="get_claim_evidence",
    )
    def get_claim_evidence(claim_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_claim_evidence(store, claim_id)

    # -- spans and sources -------------------------------------------------

    @app.get(
        "/spans/{span_id}", response_model=SpanDetail, tags=["sources"], operation_id="get_span"
    )
    def get_span(span_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_span(store, span_id)

    @app.get("/sources", response_model=SourcePage, tags=["sources"], operation_id="list_sources")
    def list_sources(
        store: SqliteStore = Depends(get_store),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        return queries.list_sources(store, limit=limit, offset=offset)

    @app.get(
        "/sources/{source_id}",
        response_model=SourceSummary,
        tags=["sources"],
        operation_id="get_source",
    )
    def get_source(source_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_source(store, source_id)

    # -- history -----------------------------------------------------------

    @app.get(
        "/revisions/{entity_type}/{entity_id}",
        response_model=list[RevisionItem],
        tags=["history"],
        operation_id="list_entity_revisions",
    )
    def list_entity_revisions(
        entity_type: str, entity_id: str, store: SqliteStore = Depends(get_store)
    ):
        try:
            return queries.list_entity_revisions(store, entity_type, entity_id)
        except BadRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get(
        "/revisions",
        response_model=list[RevisionItem],
        tags=["history"],
        operation_id="list_recent_revisions",
    )
    def list_recent_revisions(
        store: SqliteStore = Depends(get_store),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    ):
        return queries.list_recent_revisions(store, limit=limit)

    # -- retrieval and traversal ------------------------------------------

    @app.get(
        "/search", response_model=list[SearchHit], tags=["retrieval"], operation_id="search_spans"
    )
    def search_spans(
        q: str = Query(..., min_length=1),
        store: SqliteStore = Depends(get_store),
        limit: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    ):
        return queries.search_spans(store, q, limit=limit)

    @app.get("/path", response_model=PathResponse, tags=["retrieval"], operation_id="find_path")
    def find_path(
        store: SqliteStore = Depends(get_store),
        source: str = Query(..., description="Concept id to start from."),
        target: str = Query(..., description="Concept id to reach."),
        max_depth: int = Query(4, ge=1, le=6),
    ):
        return queries.find_path(store, source, target, max_depth=max_depth)

    # -- research intelligence (Phase 9) -----------------------------------

    @app.get(
        "/beliefs/{concept_id}",
        response_model=BeliefResponse,
        tags=["research"],
        operation_id="get_belief",
    )
    def get_belief(concept_id: str, store: SqliteStore = Depends(get_store)):
        """What is currently believed about a concept, with supports and dissent."""
        return queries.get_belief(store, concept_id)

    @app.get(
        "/questions",
        response_model=list[QuestionSummary],
        tags=["research"],
        operation_id="list_questions",
    )
    def list_questions(
        store: SqliteStore = Depends(get_store),
        status: str | None = Query(None, description="open | partially_answered | answered"),
    ):
        return queries.list_questions(store, status=status)

    @app.get(
        "/questions/{question_id}",
        response_model=QuestionDetail,
        tags=["research"],
        operation_id="get_question",
    )
    def get_question(question_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_question(store, question_id)

    @app.get(
        "/questions/{question_id}/evidence",
        response_model=list[QuestionEvidenceItem],
        tags=["research"],
        operation_id="get_question_evidence",
    )
    def get_question_evidence(
        question_id: str,
        store: SqliteStore = Depends(get_store),
        limit: int = Query(10, ge=1, le=MAX_PAGE_SIZE),
    ):
        return queries.get_question_evidence(store, question_id, limit=limit)

    @app.get("/gaps", response_model=GapResponse, tags=["research"], operation_id="list_gaps")
    def list_gaps(
        store: SqliteStore = Depends(get_store),
        kinds: str | None = Query(None, description="Comma-separated gap kinds."),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    ):
        return queries.list_gaps(store, kinds=kinds, limit=limit)

    @app.get(
        "/changes", response_model=ChangeResponse, tags=["research"], operation_id="get_changes"
    )
    def get_changes(
        store: SqliteStore = Depends(get_store),
        days: int = Query(30, ge=1, le=3650),
    ):
        return queries.get_changes(store, days=days)

    @app.get(
        "/syntheses",
        response_model=list[SynthesisSummary],
        tags=["research"],
        operation_id="list_syntheses",
    )
    def list_syntheses(
        store: SqliteStore = Depends(get_store),
        stale: bool | None = Query(None),
    ):
        return queries.list_syntheses(store, stale=stale)

    @app.get(
        "/syntheses/{synthesis_id}",
        response_model=SynthesisSummary,
        tags=["research"],
        operation_id="get_synthesis",
    )
    def get_synthesis(synthesis_id: str, store: SqliteStore = Depends(get_store)):
        return queries.get_synthesis(store, synthesis_id)

    # -- the explorer ------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def explorer():
        page = STATIC_DIR / "explorer.html"
        if not page.is_file():  # pragma: no cover - only if the package is broken
            return JSONResponse({"detail": "explorer.html is missing"}, status_code=500)
        return FileResponse(page, media_type="text/html")

    return app
