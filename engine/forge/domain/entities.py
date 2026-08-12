"""The Phase 1 canonical entities.

Nine types, per the approved canonical model:

    Source, Document, Span, Concept, Claim, EvidenceLink, ClaimLink,
    Provenance (in :mod:`.provenance`), Revision (in :mod:`.revision`)

Deliberately **not** implemented in Phase 1: Contradiction, Synthesis,
Question, KnowledgeGap.

These are pure domain models. They know nothing about SQLite, Neo4j, Qdrant,
or Ollama — persistence lives behind the protocols in :mod:`forge.storage`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..ids import deterministic_id
from .enums import (
    DETERMINISTIC_LINK_TYPES,
    ClaimStatus,
    ConceptKind,
    Derivation,
    EvidenceRelation,
    LinkType,
    ProvenanceTier,
    SourceKind,
    TrustTier,
)
from .provenance import Provenance, ProvenanceViolation, utc_now


class Entity(BaseModel):
    """Base: stable identity plus creation timestamp."""

    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------
# Source / Document / Span — the evidence chain
# --------------------------------------------------------------------------


class Source(Entity):
    """The origin of information. Identity is independent of content.

    ``content_hash`` is the change-detection key: re-ingesting a source whose
    hash is unchanged must be a no-op costing zero LLM calls.
    """

    kind: SourceKind
    #: Vault-relative POSIX path, URL, or repo ref. Canonicalized.
    locator: str
    content_hash: str
    trust_tier: TrustTier = TrustTier.UNVERIFIED

    title: str | None = None
    authors: tuple[str, ...] = ()
    published_at: datetime | None = None

    byte_size: int = 0
    line_count: int = 0
    ingested_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)

    @staticmethod
    def make_id(locator: str) -> str:
        return deterministic_id("source", locator)

    @classmethod
    def for_path(cls, rel_path: str, **kwargs: object) -> Source:
        locator = PurePosixPath(rel_path).as_posix()
        return cls(id=cls.make_id(locator), locator=locator, **kwargs)  # type: ignore[arg-type]


class Document(Entity):
    """A parsed rendering of a Source at a point in time.

    Separate from ``Source`` so a source can be re-parsed with a better
    extractor without losing its identity or its prior parse.
    """

    source_id: str
    version: int = 1
    parser: str
    parser_version: str
    content_hash: str
    #: Ordered heading tree as (level, text, line_number) triples.
    headings: tuple[tuple[int, str, int], ...] = ()
    frontmatter_present: bool = False
    frontmatter_valid: bool = False
    parsed_at: datetime = Field(default_factory=utc_now)

    @staticmethod
    def make_id(source_id: str, content_hash: str) -> str:
        return deterministic_id("document", source_id, content_hash)


class Span(Entity):
    """A located region of a Document — the atom of provenance.

    Without spans, "traceable to evidence" degrades to document-level
    attribution, which is not traceability.

    A span must carry enough location to answer "where exactly did this come
    from?" in a form a human can check against the original:

        Document: "Attention Is All You Need"
        Evidence: page 3, section "Model Architecture", lines 40-58

    Phase 2 added ``page``, ``char_start``/``char_end``, and ``page_span`` as
    **optional** fields. Phase 1 Markdown spans set none of them and are
    unaffected; ``make_id`` is unchanged, so existing span identities are
    stable.
    """

    document_id: str
    ordinal: int
    #: Human-readable location, e.g. "L12-L48" or "p.3 L1-L12".
    locator: str
    #: Heading path from document root, e.g. ("Architecture", "Storage").
    heading_path: tuple[str, ...] = ()
    start_line: int
    end_line: int
    text: str
    content_hash: str
    chunk_strategy: str = "heading"

    # -- Phase 2 additions, all optional -----------------------------------

    #: 1-based page number for paginated sources. ``None`` for Markdown, which
    #: has no pages — recorded as absent rather than faked as page 1.
    page: int | None = None
    #: ``(first_page, last_page)`` when a span crosses a page boundary.
    page_span: tuple[int, int] | None = None
    #: Character offsets into the document's extracted text. Practical and
    #: useful; deliberately not token offsets, which would bind provenance to a
    #: tokenizer version.
    char_start: int | None = None
    char_end: int | None = None

    @staticmethod
    def make_id(document_id: str, ordinal: int, locator: str) -> str:
        return deterministic_id("span", document_id, str(ordinal), locator)

    @model_validator(mode="after")
    def _line_order(self) -> Span:
        if self.end_line < self.start_line:
            raise ValueError(f"span end_line {self.end_line} precedes start_line {self.start_line}")
        if self.char_start is not None and self.char_end is not None:
            if self.char_end < self.char_start:
                raise ValueError(
                    f"span char_end {self.char_end} precedes char_start {self.char_start}"
                )
        if self.page is not None and self.page < 1:
            raise ValueError(f"page numbers are 1-based; got {self.page}")
        if self.page_span is not None and self.page_span[1] < self.page_span[0]:
            raise ValueError(f"page_span end precedes start: {self.page_span}")
        return self

    def citation(self) -> str:
        """Human-readable provenance string, e.g. 'p.3 > Model Architecture'."""
        parts: list[str] = []
        if self.page_span is not None and self.page_span[0] != self.page_span[1]:
            parts.append(f"pp.{self.page_span[0]}-{self.page_span[1]}")
        elif self.page is not None:
            parts.append(f"p.{self.page}")
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        if not parts:
            parts.append(self.locator)
        return " | ".join(parts)


# --------------------------------------------------------------------------
# Concept / Claim — the understanding layer
# --------------------------------------------------------------------------


class Concept(Entity):
    """A durable, named idea. Directly descends from the vault's
    "one canonical home per concept" rule."""

    canonical_name: str
    kind: ConceptKind = ConceptKind.CONCEPT
    aliases: tuple[str, ...] = ()
    #: Pointer to the claim that currently defines this concept, if any.
    #: A definition is itself an assertion and must carry provenance, so it is
    #: not stored as a bare string.
    definition_claim_id: str | None = None
    #: Canonical Markdown home in the vault, when one exists. READ-ONLY in
    #: Phase 1 (ADR-001 D2: segregated write-back, no in-place enrichment).
    vault_path: str | None = None
    #: Namespace that disambiguates a shared bare name, e.g. "data-structure".
    #: Set from the user's identity decision; ``None`` when unambiguous.
    namespace: str | None = None
    #: Proposal that created this concept. Phase 3: no canonical entity may be
    #: an unexplained orphan — every one traces back to the decision that
    #: created it, and from there to the source spans that justified it.
    origin_proposal_id: str | None = None
    #: Spans that evidenced this concept's creation.
    origin_span_ids: tuple[str, ...] = ()
    provenance: Provenance
    updated_at: datetime = Field(default_factory=utc_now)

    @staticmethod
    def make_id(canonical_name: str, namespace: str | None = None) -> str:
        """Deterministic identity, namespaced when the user has decided one.

        The namespace participates so that `pattern/Heap` and
        `data-structure/Heap` are genuinely different concepts rather than one
        overwriting the other — which is the whole point of resolving the
        collision instead of picking a winner.
        """
        key = canonical_name.strip().casefold()
        return deterministic_id("concept", f"{namespace}/{key}" if namespace else key)

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}/{self.canonical_name}" if self.namespace else self.canonical_name


class Claim(Entity):
    """An assertable statement — the unit of understanding.

    Invariant enforced by :func:`forge.domain.validation.validate_claim`:
    a claim whose tier requires evidence must have at least one EvidenceLink.
    """

    statement: str
    subject_concept_id: str | None = None
    provenance: Provenance
    status: ClaimStatus = ClaimStatus.ACTIVE

    valid_from: datetime = Field(default_factory=utc_now)
    valid_to: datetime | None = None
    superseded_by: str | None = None
    #: Proposal that created this claim (Phase 3).
    origin_proposal_id: str | None = None

    @staticmethod
    def make_id(statement: str, source_ref: str) -> str:
        return deterministic_id("claim", statement.strip(), source_ref)

    @property
    def tier(self) -> ProvenanceTier:
        return self.provenance.tier

    @model_validator(mode="after")
    def _statement_nonempty(self) -> Claim:
        if not self.statement.strip():
            raise ValueError("claim statement must not be empty")
        if self.status is ClaimStatus.SUPERSEDED and self.superseded_by is None:
            raise ValueError("superseded claim must record superseded_by")
        return self


# --------------------------------------------------------------------------
# Reified links
# --------------------------------------------------------------------------


class EvidenceLink(Entity):
    """Claim -> Span. Why a claim is believed."""

    claim_id: str
    span_id: str
    relation: EvidenceRelation
    provenance: Provenance

    @staticmethod
    def make_id(claim_id: str, span_id: str, relation: EvidenceRelation) -> str:
        return deterministic_id("evidence", claim_id, span_id, relation.value)

    @model_validator(mode="after")
    def _quotes_are_deterministic(self) -> EvidenceLink:
        # A "quote" asserts the span contains this text verbatim. That is a
        # deterministic check, so a model may not be the one claiming it.
        if (
            self.relation is EvidenceRelation.QUOTES
            and self.provenance.derivation is Derivation.MODEL
        ):
            raise ProvenanceViolation(
                "EvidenceRelation.QUOTES must be established deterministically; "
                "a model asserting a verbatim quote cannot be trusted to be verbatim"
            )
        return self


class ClaimLink(Entity):
    """Claim -> Claim, or Concept -> Concept. Carries its own provenance.

    This is the reified ``Relationship`` entity from the canonical model.
    """

    from_id: str
    to_id: str
    type: LinkType
    provenance: Provenance
    rationale: str | None = None
    #: Similarity score for RELATED_TO edges. Required for RELATED_TO, which is
    #: otherwise the edge that silently fills a knowledge graph with noise.
    score: float | None = None
    active: bool = True

    @staticmethod
    def make_id(from_id: str, to_id: str, link_type: LinkType) -> str:
        return deterministic_id("claimlink", from_id, to_id, link_type.value)

    @model_validator(mode="after")
    def _check(self) -> ClaimLink:
        if self.from_id == self.to_id:
            raise ValueError(f"self-link is not meaningful ({self.type.value} on {self.from_id})")

        if self.type is LinkType.RELATED_TO and self.score is None:
            raise ValueError(
                "RELATED_TO requires an explicit score; an unscored RELATED_TO is "
                "the edge that turns a knowledge graph into an untyped mesh"
            )

        # Principle 7, made checkable: semantic edge types cannot be claimed by
        # deterministic code, because deciding them requires judgement.
        if (
            self.provenance.derivation is Derivation.DETERMINISTIC
            and self.type not in DETERMINISTIC_LINK_TYPES
        ):
            raise ProvenanceViolation(
                f"link type {self.type.value} is semantic and cannot be asserted "
                f"deterministically; deterministic types are "
                f"{sorted(t.value for t in DETERMINISTIC_LINK_TYPES)}"
            )
        return self
