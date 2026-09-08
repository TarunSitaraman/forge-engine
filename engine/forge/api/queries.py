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

from ..domain import EntityType, GapKind, QuestionStatus
from ..graph import KnowledgeGraph
from ..storage import SqliteStore
from .models import (
    BeliefResponse,
    ChangeResponse,
    ClaimDetail,
    ClaimPage,
    ClaimSummary,
    ConceptDetail,
    ConceptPage,
    ConceptSummary,
    DissentItem,
    EvidenceItem,
    GapItem,
    GapResponse,
    NeighborItem,
    OriginProposal,
    OriginSpan,
    PathEdge,
    PathResponse,
    Provenance,
    QuestionDetail,
    QuestionEvidenceItem,
    QuestionSummary,
    RevisionItem,
    SaturatedKind,
    SearchHit,
    SourcePage,
    SourceSummary,
    SpanDetail,
    StatsResponse,
    SynthesisSummary,
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


# --------------------------------------------------------------------------
# Phase 9: research intelligence
# --------------------------------------------------------------------------


def _question_summary(question: Any, answer_count: int = 0) -> QuestionSummary:
    return QuestionSummary(
        id=question.id,
        text=question.text,
        status=question.status.value,
        tags=list(question.tags),
        note=question.note,
        concept_ids=list(question.concept_ids),
        created_at=question.created_at.isoformat(),
        resolved_at=question.resolved_at.isoformat() if question.resolved_at else None,
        provenance=_provenance(question.provenance),
        answer_count=answer_count,
    )


def _synthesis_summary(synthesis: Any) -> SynthesisSummary:
    return SynthesisSummary(
        id=synthesis.id,
        scope=synthesis.scope.value,
        scope_id=synthesis.scope_id,
        body=synthesis.body,
        source_claim_ids=list(synthesis.source_claim_ids),
        provenance=_provenance(synthesis.provenance),
        prompt_version=synthesis.prompt_version,
        generated_at=synthesis.generated_at.isoformat(),
        stale=synthesis.stale,
        stale_reason=synthesis.stale_reason,
        superseded_by=synthesis.superseded_by,
    )


def get_belief(store: SqliteStore, concept_id: str) -> BeliefResponse:
    """What is currently believed about one concept, with supports and dissent.

    Questions 1 and 2 of the vision's six, answered together because they are
    one question: a belief with no account of what disagrees with it is the
    failure mode the whole system exists to prevent.
    """
    from ..research import belief_for_concept

    belief = belief_for_concept(store, concept_id)
    if belief is None:
        raise NotFound("concept", concept_id)

    def summarise(claims):
        return [_claim_summary(c, len(store.evidence_for_claim(c.id))) for c in claims]

    return BeliefResponse(
        concept_id=belief.concept_id,
        concept_name=belief.concept_name,
        held=summarise(belief.held),
        disputed=summarise(belief.disputed),
        superseded=summarise(belief.superseded),
        supporting_sources=belief.supporting_sources,
        dissent=[DissentItem(**d.to_dict()) for d in belief.dissent],
        unevidenced=belief.unevidenced,
        is_settled=belief.is_settled,
    )


def list_questions(store: SqliteStore, *, status: str | None = None) -> list[QuestionSummary]:
    """Research questions, optionally filtered by status.

    Question 4 of the six is `status="open"`.
    """
    if status is not None:
        try:
            wanted = QuestionStatus(status)
        except ValueError:
            raise BadRequest(
                f"unknown question status {status!r}; expected one of "
                f"{', '.join(s.value for s in QuestionStatus)}"
            ) from None
    else:
        wanted = None
    return [
        _question_summary(q, len(store.answers_for_question(q.id)))
        for q in store.list_questions(wanted)
    ]


def get_question(store: SqliteStore, question_id: str) -> QuestionDetail:
    """One question with the claims that answer it."""
    question = store.get_question(question_id)
    if question is None:
        raise NotFound("question", question_id)
    answers = list(store.answers_for_question(question_id))
    return QuestionDetail(
        **_question_summary(question, len(answers)).model_dump(),
        answers=[_claim_summary(c, len(store.evidence_for_claim(c.id))) for c in answers],
    )


def get_question_evidence(
    store: SqliteStore, question_id: str, *, limit: int = 10
) -> list[QuestionEvidenceItem]:
    """Spans that bear on an open question and are not yet cited in an answer.

    Question 6 of the six. Retrieval scoped by the question rather than by a
    keyword, and deliberately excluding what has already been read into the
    answer: returning that would be a search box with extra steps.
    """
    from ..research import evidence_for_question

    if store.get_question(question_id) is None:
        raise NotFound("question", question_id)
    limit, _ = _bounds(limit, 0)

    out: list[QuestionEvidenceItem] = []
    for span_id, score, citation in evidence_for_question(store, question_id, limit=limit):
        span = store.get_span(span_id)
        document = store.get_document(span.document_id) if span else None
        source = store.get_source(document.source_id) if document else None
        out.append(
            QuestionEvidenceItem(
                span_id=span_id,
                score=score,
                citation=citation,
                text=span.text if span else "",
                source_locator=source.locator if source else None,
                trust_tier=source.trust_tier.value if source else None,
            )
        )
    return out


def list_gaps(
    store: SqliteStore, *, kinds: str | None = None, limit: int = 50
) -> GapResponse:
    """What the model does not hold, by deterministic graph query.

    Question 5 of the six. `kinds` is a comma-separated list; naming a kind
    explicitly enumerates it even when it saturates the corpus.
    """
    from ..research import gap_report

    wanted: list[GapKind] | None = None
    if kinds:
        wanted = []
        for raw in kinds.split(","):
            name = raw.strip()
            if not name:
                continue
            try:
                wanted.append(GapKind(name))
            except ValueError:
                raise BadRequest(
                    f"unknown gap kind {name!r}; expected one of "
                    f"{', '.join(k.value for k in GapKind)}"
                ) from None

    limit, _ = _bounds(limit, 0)
    report = gap_report(store, kinds=wanted or None, limit=limit)
    return GapResponse(
        total=report.total,
        returned=report.returned,
        by_kind=report.by_kind,
        saturated=[SaturatedKind(**s) for s in report.saturated],
        gaps=[
            GapItem(
                id=g.id,
                kind=g.kind.value,
                subject_id=g.subject_id,
                subject_type=g.subject_type,
                subject_label=g.subject_label,
                detail=g.detail,
                weight=round(g.weight, 3),
                evidence=list(g.evidence),
                detected_by=f"forge.research.gaps:{g.kind.value}",
            )
            for g in report.gaps
        ],
    )


def get_changes(store: SqliteStore, *, days: int = 30) -> ChangeResponse:
    """What changed in the model over the last `days` days.

    Question 3 of the six. Read from the revision log rather than by diffing
    snapshots, because a diff shows only the endpoints: a claim created,
    disputed and superseded inside the window is three facts about how
    understanding moved, and a diff would show one.
    """
    from ..research import changes_since

    if days < 1:
        raise BadRequest("a change window needs at least one day")
    report = changes_since(store, days=days)
    return ChangeResponse(**report.to_dict())


def list_syntheses(store: SqliteStore, *, stale: bool | None = None) -> list[SynthesisSummary]:
    """Generated aggregates, with whether each has gone stale.

    A stale synthesis is never hidden and never silently trusted: it is
    returned, marked, with the deterministic reason it went stale.
    """
    return [_synthesis_summary(s) for s in store.list_syntheses(stale=stale)]


def get_synthesis(store: SqliteStore, synthesis_id: str) -> SynthesisSummary:
    """One synthesis by id."""
    synthesis = store.get_synthesis(synthesis_id)
    if synthesis is None:
        raise NotFound("synthesis", synthesis_id)
    return _synthesis_summary(synthesis)


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
    # Phase 9, the six vision questions.
    "get_belief": get_belief,
    "list_questions": list_questions,
    "get_question": get_question,
    "get_question_evidence": get_question_evidence,
    "list_gaps": list_gaps,
    "get_changes": get_changes,
    "list_syntheses": list_syntheses,
    "get_synthesis": get_synthesis,
}
