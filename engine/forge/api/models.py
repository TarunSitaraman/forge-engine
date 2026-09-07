"""Response shapes for the read API.

**Every payload that can carry provenance, does.** Principle 10 requires
generated content to be visually distinguishable from source evidence, and a UI
can only do that if the tier travels with the data. Making it optional here
would push the decision to each route, and the one route that forgot would
render model output as though a human had written it.

These are deliberately *flat* view models rather than the domain entities.
The domain objects carry fields the API has no business publishing (content
hashes, internal derivation keys), and pinning the wire format separately means
a domain refactor cannot silently change what a client sees.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """How strongly warranted this object is, and what produced it.

    `tier` is the field a UI keys its styling on. `derivation` says whether a
    model was involved at all, which is the coarser question a reader usually
    asks first.
    """

    tier: str = Field(description="SOURCE_FACT | EXTRACTED_CLAIM | MODEL_INFERENCE | SYNTHESIS | USER_ASSERTION")
    derivation: str
    confidence: float | None = None
    model_id: str | None = None
    agent: str | None = None

    @property
    def is_model_generated(self) -> bool:
        return self.derivation == "MODEL"


class ConceptSummary(BaseModel):
    id: str
    canonical_name: str
    qualified_name: str
    namespace: str | None = None
    kind: str
    aliases: list[str] = Field(default_factory=list)
    vault_path: str | None = None
    provenance: Provenance


class ClaimSummary(BaseModel):
    id: str
    statement: str
    subject_concept_id: str | None = None
    status: str | None = None
    provenance: Provenance
    evidence_count: int = 0


class EvidenceItem(BaseModel):
    """One step of the claim -> span -> document -> source chain.

    `text` is the span verbatim. The gate for this phase is reaching the exact
    source span in one interaction, so the span's own words are part of the
    response rather than something to fetch afterwards.
    """

    relation: str
    span_id: str
    #: `source locator :: span citation`, one string for a client that wants one.
    citation: str | None = None
    #: The span's citation *without* the source prefix. A client that renders
    #: the locator on its own line needs this, or the path appears twice.
    span_citation: str | None = None
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    text: str | None = None
    document_id: str | None = None
    source_id: str | None = None
    source_locator: str | None = None
    source_kind: str | None = None
    trust_tier: str | None = None


class ClaimDetail(ClaimSummary):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class NeighborItem(BaseModel):
    """One concept one edge away, carrying the edge's own provenance."""

    concept_id: str
    label: str | None = None
    link_type: str
    direction: str
    score: float | None = None
    rationale: str | None = None
    provenance: Provenance | None = None


class OriginSpan(BaseModel):
    span_id: str
    citation: str
    text: str


class OriginProposal(BaseModel):
    id: str
    type: str
    status: str
    reason: str | None = None
    decided_by: str | None = None
    safety: str | None = None


class ConceptDetail(BaseModel):
    """Everything that justifies a concept's existence, in one response."""

    concept: ConceptSummary
    origin_proposal: OriginProposal | None = None
    origin_spans: list[OriginSpan] = Field(default_factory=list)
    claims: list[ClaimSummary] = Field(default_factory=list)
    relationships: list[NeighborItem] = Field(default_factory=list)


class SpanDetail(BaseModel):
    id: str
    document_id: str
    ordinal: int
    locator: str
    citation: str
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    text: str
    source_id: str | None = None
    source_locator: str | None = None
    trust_tier: str | None = None


class SourceSummary(BaseModel):
    id: str
    locator: str
    kind: str
    trust_tier: str
    title: str | None = None
    byte_size: int | None = None
    line_count: int | None = None
    content_hash: str | None = None


class RevisionItem(BaseModel):
    """One entry of an entity's history. The timeline the phase scope asks for.

    `cause` is the field that makes "why did my understanding change?"
    answerable, so it is published even though it is usually an opaque id: a
    client can follow it back to the source or claim that triggered the change.
    """

    id: str
    entity_type: str
    entity_id: str
    op: str
    created_at: str
    cause: str | None = None
    workflow_run_id: str | None = None
    note: str | None = None
    #: Which top-level fields differ between `before` and `after`. The states
    #: themselves are not published: they are whole serialized entities, and a
    #: timeline wants to show what moved, not re-transmit the object twice.
    changed_fields: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    span_id: str
    score: float
    citation: str | None = None
    text: str
    source_locator: str | None = None
    #: A hit is quoted source text, so its provenance is the source's trust
    #: tier rather than a provenance tier. Published for the same reason: an
    #: agent must never receive a result it cannot attribute.
    trust_tier: str | None = None


class PathEdge(BaseModel):
    """One hop, with who asserted it and why.

    A path is a chain of assertions. Publishing only the link *type* would tell
    an agent that two concepts are `RELATED_TO` without saying whether that
    came from a human-authored wikilink or a model's guess, which is precisely
    the distinction this system exists to keep.
    """

    from_id: str
    to_id: str
    link_type: str
    score: float | None = None
    rationale: str | None = None
    provenance: Provenance


class PathResponse(BaseModel):
    found: bool
    depth: int | None = None
    nodes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    edges: list[PathEdge] = Field(default_factory=list)


class Page(BaseModel):
    """A slice of a listing, with enough to ask for the next one."""

    total: int
    limit: int
    offset: int
    returned: int


class ConceptPage(Page):
    items: list[ConceptSummary] = Field(default_factory=list)


class ClaimPage(Page):
    items: list[ClaimSummary] = Field(default_factory=list)


class SourcePage(Page):
    items: list[SourceSummary] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """What the store holds, plus the shape of the graph over it."""

    api_version: str
    counts: dict[str, int]
    graph: dict[str, object] | None = None
    #: Asserted, not assumed. The whole API is deterministic; a non-zero value
    #: here means a route reached a model, which this phase forbids.
    llm_calls: int = 0
