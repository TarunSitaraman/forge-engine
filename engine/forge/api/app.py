"""FastAPI application: read-only views over the knowledge store.

**Every route is a GET, and that is enforced rather than observed.** A test
walks the OpenAPI schema and fails on any other method. Knowledge changes
through proposal and activation, which require a human decision; an HTTP write
path would be a second way in without that gate, and "we only wrote GETs" is
the kind of claim that stops being true six months later.

**Zero model calls.** The API is a view over what is already stored. The stats
route publishes the process call counter so a client can see that, and a test
asserts it stays at zero across every route.

The routes are thin on purpose. `KnowledgeGraph.explain_concept` and
`get_claim_evidence` already assemble the chains this phase is about; an API
that re-implemented them would drift from what `forge explain` prints.
"""

from __future__ import annotations

from pathlib import Path as FilePath
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from ..config import Settings
from ..domain import EntityType
from ..graph import KnowledgeGraph
from ..llm.base import CALLS
from ..storage import SqliteStore
from . import API_VERSION
from .models import (
    ClaimDetail,
    ClaimPage,
    ClaimSummary,
    ConceptDetail,
    ConceptPage,
    ConceptSummary,
    EvidenceItem,
    NeighborItem,
    OriginProposal,
    OriginSpan,
    PathResponse,
    Provenance,
    RevisionItem,
    SearchHit,
    SourcePage,
    SourceSummary,
    SpanDetail,
    StatsResponse,
)

STATIC_DIR = FilePath(__file__).resolve().parent / "static"

#: Ceiling on any listing, whatever `limit` asks for. The vault has 545
#: concepts and 8,133 spans; an unbounded list endpoint is a way to make the
#: server hold the whole store in memory because a client typed a big number.
MAX_PAGE_SIZE = 200


# --------------------------------------------------------------------------
# Adapters: domain objects -> wire models
# --------------------------------------------------------------------------


def _provenance(prov: Any) -> Provenance:
    return Provenance(
        tier=prov.tier.value,
        derivation=prov.derivation.value,
        confidence=getattr(prov, "confidence", None),
        model_id=getattr(prov, "model_id", None),
        agent=getattr(prov, "agent", None),
    )


def _concept_summary(concept: Any) -> ConceptSummary:
    return ConceptSummary(
        id=concept.id,
        canonical_name=concept.canonical_name,
        qualified_name=concept.qualified_name,
        namespace=concept.namespace,
        kind=concept.kind.value,
        aliases=list(concept.aliases),
        vault_path=concept.vault_path,
        provenance=_provenance(concept.provenance),
    )


def _claim_summary(claim: Any, evidence_count: int = 0) -> ClaimSummary:
    return ClaimSummary(
        id=claim.id,
        statement=claim.statement,
        subject_concept_id=claim.subject_concept_id,
        status=claim.status.value,
        provenance=_provenance(claim.provenance),
        evidence_count=evidence_count,
    )


def _span_detail(span: Any, store: SqliteStore) -> SpanDetail:
    document = store.get_document(span.document_id)
    source = store.get_source(document.source_id) if document else None
    return SpanDetail(
        id=span.id,
        document_id=span.document_id,
        ordinal=span.ordinal,
        locator=span.locator,
        citation=span.citation(),
        start_line=span.start_line,
        end_line=span.end_line,
        page=span.page,
        heading_path=list(span.heading_path),
        text=span.text,
        source_id=source.id if source else None,
        source_locator=source.locator if source else None,
    )


def _source_summary(source: Any) -> SourceSummary:
    return SourceSummary(
        id=source.id,
        locator=source.locator,
        kind=source.kind.value,
        trust_tier=source.trust_tier.value,
        title=source.title,
        byte_size=source.byte_size,
        line_count=source.line_count,
        content_hash=source.content_hash,
    )


def _changed_fields(before: dict | None, after: dict | None) -> list[str]:
    """Which top-level keys differ. Cheap, and enough for a timeline row."""
    if before is None or after is None:
        return sorted((after or before or {}).keys())
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


def _revision_item(revision: Any) -> RevisionItem:
    return RevisionItem(
        id=revision.id,
        entity_type=revision.entity_type.value,
        entity_id=revision.entity_id,
        op=revision.op.value,
        created_at=revision.created_at.isoformat(),
        cause=revision.cause,
        workflow_run_id=revision.workflow_run_id,
        note=revision.note,
        changed_fields=_changed_fields(revision.before, revision.after),
    )


def _neighbor(raw: dict[str, Any]) -> NeighborItem:
    return NeighborItem(
        concept_id=raw["entity_id"],
        label=raw.get("label"),
        link_type=raw["type"],
        direction=raw["direction"],
        score=raw.get("score"),
        rationale=raw.get("rationale"),
    )


def _page_bounds(limit: int, offset: int) -> tuple[int, int]:
    return max(0, min(limit, MAX_PAGE_SIZE)), max(0, offset)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


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

    def get_graph(store: SqliteStore = Depends(get_store)) -> KnowledgeGraph:
        return KnowledgeGraph(store)

    # -- meta --------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        return {"ok": True, "api_version": API_VERSION, "vault": str(settings.vault_path)}

    @app.get("/stats", response_model=StatsResponse, tags=["meta"])
    def stats(store: SqliteStore = Depends(get_store), graph: KnowledgeGraph = Depends(get_graph)):
        counts = store.counts()
        metrics = graph.metrics() if counts.get("concepts") else None
        return StatsResponse(
            api_version=API_VERSION,
            counts=counts,
            graph=metrics.to_dict() if metrics else None,
            llm_calls=CALLS.count,
        )

    # -- concepts ----------------------------------------------------------

    @app.get("/concepts", response_model=ConceptPage, tags=["concepts"])
    def list_concepts(
        store: SqliteStore = Depends(get_store),
        q: str | None = Query(None, description="Case-insensitive substring of the name."),
        kind: str | None = Query(None),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        limit, offset = _page_bounds(limit, offset)
        items = list(store.list_concepts())
        if q:
            needle = q.casefold()
            items = [c for c in items if needle in c.canonical_name.casefold()]
        if kind:
            items = [c for c in items if c.kind.value == kind]
        items.sort(key=lambda c: c.canonical_name.casefold())
        window = items[offset : offset + limit]
        return ConceptPage(
            total=len(items),
            limit=limit,
            offset=offset,
            returned=len(window),
            items=[_concept_summary(c) for c in window],
        )

    @app.get("/concepts/{concept_id}", response_model=ConceptDetail, tags=["concepts"])
    def get_concept(
        concept_id: str,
        store: SqliteStore = Depends(get_store),
        graph: KnowledgeGraph = Depends(get_graph),
    ):
        explained = graph.explain_concept(concept_id)
        if explained is None:
            raise HTTPException(status_code=404, detail=f"no concept {concept_id!r}")
        concept = store.get_concept(concept_id)
        proposal = explained.get("origin_proposal")
        return ConceptDetail(
            concept=_concept_summary(concept),
            origin_proposal=OriginProposal(**proposal) if proposal else None,
            origin_spans=[OriginSpan(**s) for s in explained["origin_spans"]],
            claims=[
                _claim_summary(c, len(store.evidence_for_claim(c.id)))
                for c in graph.get_concept_claims(concept_id)
            ],
            relationships=[_neighbor(n) for n in explained["relationships"]],
        )

    @app.get(
        "/concepts/{concept_id}/neighbors",
        response_model=list[NeighborItem],
        tags=["concepts"],
    )
    def concept_neighbors(
        concept_id: str,
        store: SqliteStore = Depends(get_store),
        graph: KnowledgeGraph = Depends(get_graph),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    ):
        if store.get_concept(concept_id) is None:
            raise HTTPException(status_code=404, detail=f"no concept {concept_id!r}")
        return [_neighbor(n.to_dict()) for n in graph.get_neighbors(concept_id)][:limit]

    # -- claims and their evidence ----------------------------------------

    @app.get("/claims", response_model=ClaimPage, tags=["claims"])
    def list_claims(
        store: SqliteStore = Depends(get_store),
        concept_id: str | None = Query(None),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        limit, offset = _page_bounds(limit, offset)
        items = list(store.list_claims())
        if concept_id:
            items = [c for c in items if c.subject_concept_id == concept_id]
        window = items[offset : offset + limit]
        return ClaimPage(
            total=len(items),
            limit=limit,
            offset=offset,
            returned=len(window),
            items=[_claim_summary(c, len(store.evidence_for_claim(c.id))) for c in window],
        )

    @app.get("/claims/{claim_id}", response_model=ClaimDetail, tags=["claims"])
    def get_claim(
        claim_id: str,
        store: SqliteStore = Depends(get_store),
        graph: KnowledgeGraph = Depends(get_graph),
    ):
        """Claim plus its full evidence chain.

        The phase gate is reaching the exact source span from a claim in **one
        interaction**, so the span text and its source locator are in this
        response rather than behind a second request.
        """
        claim = store.get_claim(claim_id)
        if claim is None:
            raise HTTPException(status_code=404, detail=f"no claim {claim_id!r}")
        chain = _evidence_chain(claim_id, store, graph)
        return ClaimDetail(
            **_claim_summary(claim, len(chain)).model_dump(),
            evidence=chain,
        )

    @app.get("/claims/{claim_id}/evidence", response_model=list[EvidenceItem], tags=["claims"])
    def claim_evidence(
        claim_id: str,
        store: SqliteStore = Depends(get_store),
        graph: KnowledgeGraph = Depends(get_graph),
    ):
        if store.get_claim(claim_id) is None:
            raise HTTPException(status_code=404, detail=f"no claim {claim_id!r}")
        return _evidence_chain(claim_id, store, graph)

    def _evidence_chain(
        claim_id: str, store: SqliteStore, graph: KnowledgeGraph
    ) -> list[EvidenceItem]:
        out: list[EvidenceItem] = []
        for raw in graph.get_claim_evidence(claim_id):
            source = store.get_source(raw["source_id"]) if raw.get("source_id") else None
            span = store.get_span(raw["span_id"])
            out.append(
                EvidenceItem(
                    relation=raw["relation"],
                    span_id=raw["span_id"],
                    citation=raw.get("citation"),
                    span_citation=span.citation() if span else None,
                    page=raw.get("page"),
                    heading_path=raw.get("heading_path") or [],
                    text=raw.get("text"),
                    document_id=raw.get("document_id"),
                    source_id=raw.get("source_id"),
                    source_locator=source.locator if source else None,
                    source_kind=raw.get("source_kind"),
                    trust_tier=raw.get("trust_tier"),
                )
            )
        return out

    # -- spans and sources -------------------------------------------------

    @app.get("/spans/{span_id}", response_model=SpanDetail, tags=["sources"])
    def get_span(span_id: str, store: SqliteStore = Depends(get_store)):
        span = store.get_span(span_id)
        if span is None:
            raise HTTPException(status_code=404, detail=f"no span {span_id!r}")
        return _span_detail(span, store)

    @app.get("/sources", response_model=SourcePage, tags=["sources"])
    def list_sources(
        store: SqliteStore = Depends(get_store),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ):
        limit, offset = _page_bounds(limit, offset)
        items = sorted(store.list_sources(), key=lambda s: s.locator)
        window = items[offset : offset + limit]
        return SourcePage(
            total=len(items),
            limit=limit,
            offset=offset,
            returned=len(window),
            items=[_source_summary(s) for s in window],
        )

    @app.get("/sources/{source_id}", response_model=SourceSummary, tags=["sources"])
    def get_source(source_id: str, store: SqliteStore = Depends(get_store)):
        source = store.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"no source {source_id!r}")
        return _source_summary(source)

    # -- history -----------------------------------------------------------

    @app.get(
        "/revisions/{entity_type}/{entity_id}",
        response_model=list[RevisionItem],
        tags=["history"],
    )
    def entity_revisions(
        entity_type: str, entity_id: str, store: SqliteStore = Depends(get_store)
    ):
        """The revision timeline for one entity, oldest first."""
        try:
            kind = EntityType(entity_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown entity type {entity_type!r}; expected one of "
                    f"{', '.join(e.value for e in EntityType)}"
                ),
            ) from None
        return [_revision_item(r) for r in store.revisions_for(kind, entity_id)]

    @app.get("/revisions", response_model=list[RevisionItem], tags=["history"])
    def recent_revisions(
        store: SqliteStore = Depends(get_store),
        limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    ):
        return [_revision_item(r) for r in store.recent_revisions(limit=limit)]

    # -- retrieval and traversal ------------------------------------------

    @app.get("/search", response_model=list[SearchHit], tags=["retrieval"])
    def search(
        q: str = Query(..., min_length=1),
        store: SqliteStore = Depends(get_store),
        limit: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    ):
        """Lexical span search. Deterministic, no model, no embeddings."""
        hits: list[SearchHit] = []
        for span, score in store.search_spans(q, limit=limit):
            document = store.get_document(span.document_id)
            source = store.get_source(document.source_id) if document else None
            hits.append(
                SearchHit(
                    span_id=span.id,
                    score=score,
                    citation=span.citation(),
                    text=span.text,
                    source_locator=source.locator if source else None,
                )
            )
        return hits

    @app.get("/path", response_model=PathResponse, tags=["retrieval"])
    def find_path(
        graph: KnowledgeGraph = Depends(get_graph),
        source: str = Query(..., description="Concept id to start from."),
        target: str = Query(..., description="Concept id to reach."),
        max_depth: int = Query(4, ge=1, le=6),
    ):
        """Shortest path within `max_depth`.

        `found: false` means "no path within max_depth", which is not the same
        as "no path": the search is bounded and cannot establish the latter.
        """
        path = graph.find_path(source, target, max_depth=max_depth)
        if path is None:
            return PathResponse(found=False)
        labels = graph.node_labels(path.nodes)
        return PathResponse(
            found=True,
            depth=path.depth,
            nodes=list(path.nodes),
            labels=[labels.get(n, n) for n in path.nodes],
            edges=[link.type.value for link in path.links],
        )

    # -- the explorer ------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def explorer():
        page = STATIC_DIR / "explorer.html"
        if not page.is_file():  # pragma: no cover - only if the package is broken
            return JSONResponse({"detail": "explorer.html is missing"}, status_code=500)
        return FileResponse(page, media_type="text/html")

    return app
