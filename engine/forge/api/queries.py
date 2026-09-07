"""The knowledge queries, once, for every interface that exposes them.

**Why this module exists.** Phase 8's gate is that no capability lives in only
one interface, and that the HTTP API and the MCP server have identical
semantics. Two hand-written implementations of "get a claim with its evidence"
would satisfy that on the day they were written and drift by the next change.
So the queries live here, as plain functions over a store, and both interfaces
are thin wrappers. Neither imports the other, and neither knows anything the
other does not.

**`CAPABILITIES` makes the gate checkable.** It names every query exactly once.
The HTTP layer stamps each route's `operation_id` with the capability name and
the MCP layer names each tool the same; a test compares the three sets and
fails if any interface is missing one, or invents one. Without that registry,
"identical semantics" is a claim in a docstring.

Nothing here writes. Nothing here calls a model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..domain import EntityType
from ..graph import KnowledgeGraph
from ..storage import SqliteStore
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
    PathEdge,
    PathResponse,
    Provenance,
    RevisionItem,
    SearchHit,
    SourcePage,
    SourceSummary,
    SpanDetail,
    StatsResponse,
)

#: Ceiling on any listing, whatever a caller asks for. The vault has 545
#: concepts and 8,133 spans; an unbounded list is a way to make the server hold
#: the whole store in memory because someone typed a big number.
MAX_PAGE_SIZE = 200


class NotFound(LookupError):
    """No such entity. Each interface maps this to its own idiom.

    A domain-level error rather than an `HTTPException`, so `queries` stays
    importable without a web framework and the MCP server does not have to
    catch something named after HTTP.
    """

    def __init__(self, kind: str, entity_id: str) -> None:
        super().__init__(f"no {kind} {entity_id!r}")
        self.kind = kind
        self.entity_id = entity_id


class BadRequest(ValueError):
    """The caller asked for something malformed, and can fix it from the message."""


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
        trust_tier=source.trust_tier.value if source else None,
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


def _neighbor(neighbor: Any) -> NeighborItem:
    """Carries the edge's own provenance, not just its type.

    An agent told that two concepts are related must be able to see *why* and
    on whose authority. `RELATED_TO` alone reads like a measurement; the
    rationale says it is a human-authored wikilink.
    """
    link = neighbor.link
    return NeighborItem(
        concept_id=neighbor.entity_id,
        label=neighbor.label,
        link_type=link.type.value,
        direction=neighbor.direction,
        score=link.score,
        rationale=link.rationale,
        provenance=_provenance(link.provenance),
    )


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(limit, MAX_PAGE_SIZE)), max(0, offset)


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


def get_stats(store: SqliteStore) -> StatsResponse:
    """Counts of what the store holds, and the shape of the graph over it."""
    from ..llm.base import CALLS
    from . import API_VERSION

    counts = store.counts()
    metrics = KnowledgeGraph(store).metrics() if counts.get("concepts") else None
    return StatsResponse(
        api_version=API_VERSION,
        counts=counts,
        graph=metrics.to_dict() if metrics else None,
        llm_calls=CALLS.count,
    )


def list_concepts(
    store: SqliteStore,
    *,
    q: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ConceptPage:
    """Concepts, optionally filtered by name substring or kind."""
    limit, offset = _bounds(limit, offset)
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


def get_concept(store: SqliteStore, concept_id: str) -> ConceptDetail:
    """Everything that justifies a concept: origin, claims, relationships."""
    graph = KnowledgeGraph(store)
    explained = graph.explain_concept(concept_id)
    if explained is None:
        raise NotFound("concept", concept_id)
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
        relationships=[_neighbor(n) for n in graph.get_neighbors(concept_id)],
    )


def list_concept_neighbors(
    store: SqliteStore, concept_id: str, *, limit: int = 50
) -> list[NeighborItem]:
    """Concepts one edge away, with each edge's provenance and rationale."""
    if store.get_concept(concept_id) is None:
        raise NotFound("concept", concept_id)
    limit, _ = _bounds(limit, 0)
    return [_neighbor(n) for n in KnowledgeGraph(store).get_neighbors(concept_id)][:limit]


def list_claims(
    store: SqliteStore,
    *,
    concept_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ClaimPage:
    """Claims, optionally only those about one concept."""
    limit, offset = _bounds(limit, offset)
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


def get_claim_evidence(store: SqliteStore, claim_id: str) -> list[EvidenceItem]:
    """The chain from a claim back to the exact source text."""
    if store.get_claim(claim_id) is None:
        raise NotFound("claim", claim_id)
    out: list[EvidenceItem] = []
    for raw in KnowledgeGraph(store).get_claim_evidence(claim_id):
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


def get_claim(store: SqliteStore, claim_id: str) -> ClaimDetail:
    """A claim together with its evidence, so the source is one call away."""
    claim = store.get_claim(claim_id)
    if claim is None:
        raise NotFound("claim", claim_id)
    chain = get_claim_evidence(store, claim_id)
    return ClaimDetail(**_claim_summary(claim, len(chain)).model_dump(), evidence=chain)


def get_span(store: SqliteStore, span_id: str) -> SpanDetail:
    """One span of a source document, verbatim, with where it came from."""
    span = store.get_span(span_id)
    if span is None:
        raise NotFound("span", span_id)
    return _span_detail(span, store)


def list_sources(store: SqliteStore, *, limit: int = 50, offset: int = 0) -> SourcePage:
    """Ingested sources, with the trust tier each was assigned."""
    limit, offset = _bounds(limit, offset)
    items = sorted(store.list_sources(), key=lambda s: s.locator)
    window = items[offset : offset + limit]
    return SourcePage(
        total=len(items),
        limit=limit,
        offset=offset,
        returned=len(window),
        items=[_source_summary(s) for s in window],
    )


def get_source(store: SqliteStore, source_id: str) -> SourceSummary:
    """One source by id."""
    source = store.get_source(source_id)
    if source is None:
        raise NotFound("source", source_id)
    return _source_summary(source)


def list_entity_revisions(
    store: SqliteStore, entity_type: str, entity_id: str
) -> list[RevisionItem]:
    """The revision timeline for one entity, oldest first."""
    try:
        kind = EntityType(entity_type)
    except ValueError:
        raise BadRequest(
            f"unknown entity type {entity_type!r}; expected one of "
            f"{', '.join(e.value for e in EntityType)}"
        ) from None
    return [_revision_item(r) for r in store.revisions_for(kind, entity_id)]


def list_recent_revisions(store: SqliteStore, *, limit: int = 50) -> list[RevisionItem]:
    """The most recent changes across every entity."""
    limit, _ = _bounds(limit, 0)
    return [_revision_item(r) for r in store.recent_revisions(limit=limit)]


def search_spans(store: SqliteStore, q: str, *, limit: int = 20) -> list[SearchHit]:
    """Lexical span search. Deterministic: no model, no embeddings."""
    if not q.strip():
        raise BadRequest("a search needs a non-empty query")
    limit, _ = _bounds(limit, 0)
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
                trust_tier=source.trust_tier.value if source else None,
            )
        )
    return hits


def find_path(
    store: SqliteStore, source: str, target: str, *, max_depth: int = 4
) -> PathResponse:
    """Shortest path between two concepts, within `max_depth`.

    `found: false` means "no path within max_depth", which is not the same as
    "no path": the search is bounded and cannot establish the latter. Each edge
    carries its own provenance, because a path is a chain of assertions and an
    agent must be able to see who asserted each one.
    """
    graph = KnowledgeGraph(store)
    path = graph.find_path(source, target, max_depth=max_depth)
    if path is None:
        return PathResponse(found=False)
    labels = graph.node_labels(path.nodes)
    return PathResponse(
        found=True,
        depth=path.depth,
        nodes=list(path.nodes),
        labels=[labels.get(n, n) for n in path.nodes],
        edges=[
            PathEdge(
                from_id=link.from_id,
                to_id=link.to_id,
                link_type=link.type.value,
                score=link.score,
                rationale=link.rationale,
                provenance=_provenance(link.provenance),
            )
            for link in path.links
        ],
    )


#: Every knowledge capability, named once. Both interfaces are checked against
#: this, so an addition that reaches only one of them fails the suite.
#:
#: Interface plumbing is deliberately absent: HTTP's `/health` and the explorer
#: page at `/` are how that interface is operated, not things an agent can ask
#: about the knowledge model. MCP has its own liveness semantics and needs
#: neither.
CAPABILITIES: dict[str, Callable[..., Any]] = {
    "get_stats": get_stats,
    "list_concepts": list_concepts,
    "get_concept": get_concept,
    "list_concept_neighbors": list_concept_neighbors,
    "list_claims": list_claims,
    "get_claim": get_claim,
    "get_claim_evidence": get_claim_evidence,
    "get_span": get_span,
    "list_sources": list_sources,
    "get_source": get_source,
    "list_entity_revisions": list_entity_revisions,
    "list_recent_revisions": list_recent_revisions,
    "search_spans": search_spans,
    "find_path": find_path,
}
