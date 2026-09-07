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

    tier: str = Field(
        description="SOURCE_FACT | EXTRACTED_CLAIM | MODEL_INFERENCE | SYNTHESIS | USER_ASSERTION"
    )
    derivation: str
    model_id: str | None = None
    agent: str | None = None

    # **No confidence field, deliberately.** The first version of this model
    # published one, read through `getattr(prov, "confidence", None)` against a
    # `Provenance` that has no such attribute, so it was `null` in every
    # response ever served. Worse than useless: a nullable confidence in a
    # published schema suggests the system calibrates and merely declined to
    # this time. It does not. `forge.evolution.impact` states the reason: a
    # number a model emits about its own certainty is not a measurement, and
    # attaching one makes output look calibrated when it is not. Outcomes are
    # categorical and provenance carries the rest.

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


# --------------------------------------------------------------------------
# Phase 9: research intelligence
# --------------------------------------------------------------------------


class QuestionSummary(BaseModel):
    """A research question. Always `USER_ASSERTION`: a human asked it."""

    id: str
    text: str
    status: str
    tags: list[str] = Field(default_factory=list)
    note: str | None = None
    concept_ids: list[str] = Field(default_factory=list)
    created_at: str
    resolved_at: str | None = None
    provenance: Provenance
    answer_count: int = 0


class QuestionDetail(QuestionSummary):
    answers: list[ClaimSummary] = Field(default_factory=list)


class DissentItem(BaseModel):
    """One reason to doubt a belief, and what raised it.

    Forge has no `CONTRADICTS` edge on purpose, so `kind` is never a model's
    verdict: `disputed_claim` and `superseded_claim` are states a human created,
    and `open_conflict` is a proposal awaiting one.
    """

    kind: str
    detail: str
    claim_id: str | None = None
    proposal_id: str | None = None
    raised_by: str | None = None


class BeliefResponse(BaseModel):
    """What is currently held about one concept, with its supports and dissent.

    There is no confidence number here and there will not be: outcomes are
    categorical and provenance carries the rest. What a reader gets instead is
    every held claim with how it was derived, which is strictly more information
    than a single score.
    """

    concept_id: str
    concept_name: str
    held: list[ClaimSummary] = Field(default_factory=list)
    disputed: list[ClaimSummary] = Field(default_factory=list)
    superseded: list[ClaimSummary] = Field(default_factory=list)
    supporting_sources: list[str] = Field(default_factory=list)
    dissent: list[DissentItem] = Field(default_factory=list)
    #: Held claims resting on no evidence. Only USER_ASSERTION may legitimately
    #: do that; anything else listed here is a defect worth seeing.
    unevidenced: list[str] = Field(default_factory=list)
    is_settled: bool = False


class GapItem(BaseModel):
    """A structural observation about what the model does not hold.

    Never a judgement: "this concept has no claims" is a fact about the graph,
    and whether it matters is the reader's call. `weight` orders a report and
    is not a probability.
    """

    id: str
    kind: str
    subject_id: str
    subject_type: str
    subject_label: str
    detail: str
    weight: float
    evidence: list[str] = Field(default_factory=list)
    #: The rule that produced this finding. A derived result still has to say
    #: where it came from: an agent handed a gap should be able to report that
    #: Forge computed it by a named deterministic rule, rather than presenting
    #: it as an opinion someone formed.
    detected_by: str


class SaturatedKind(BaseModel):
    """A gap kind that describes the corpus rather than any one subject."""

    kind: str
    count: int
    population: int
    detail: str


class GapResponse(BaseModel):
    total: int
    returned: int
    #: How the whole report was produced. Same reason as `GapItem.detected_by`:
    #: the provenance of a derived report is the procedure that derived it.
    derived_by: str = "forge.research.gaps.gap_report (deterministic graph queries)"
    by_kind: dict[str, int] = Field(default_factory=dict)
    saturated: list[SaturatedKind] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)


class ChangeResponse(BaseModel):
    """What moved in the model over a window, read from the revision log."""

    derived_by: str = "forge.research.changes.changes_since (the revision log)"
    since: str
    until: str
    total: int
    #: True when the scan hit its ceiling, so the counts are a floor. Reported
    #: rather than silently truncated.
    truncated: bool = False
    by_entity: dict[str, dict[str, int]] = Field(default_factory=dict)
    claims_created: list[str] = Field(default_factory=list)
    claims_superseded: list[str] = Field(default_factory=list)
    claims_disputed: list[str] = Field(default_factory=list)
    concepts_created: list[str] = Field(default_factory=list)
    syntheses_staled: list[str] = Field(default_factory=list)
    causes: list[str] = Field(default_factory=list)


class SynthesisSummary(BaseModel):
    """A generated aggregate over claims, and the object most likely to be
    mistaken for evidence, which is why staleness is published beside it."""

    id: str
    scope: str
    scope_id: str | None = None
    body: str
    source_claim_ids: list[str] = Field(default_factory=list)
    provenance: Provenance
    prompt_version: str | None = None
    generated_at: str
    stale: bool
    #: What made it stale, in the words of the deterministic check. A bare
    #: `stale = true` tells a reader nothing about what moved underneath it.
    stale_reason: str | None = None
    superseded_by: str | None = None


class QuestionEvidenceItem(BaseModel):
    """A span that bears on an open question and is not yet cited in an answer."""

    span_id: str
    score: float
    citation: str
    text: str
    source_locator: str | None = None
    trust_tier: str | None = None
