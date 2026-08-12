"""Storage protocols.

Phase 1 deliberately commits to **no** database technology. These protocols are
the seam that keeps that promise honest: the domain and the indexer depend on
these interfaces, never on SQLite, Neo4j, Qdrant, or Postgres.

When a graph store is eventually justified by measurement (see
docs/architecture/technology-decisions.md §5.3), it implements
:class:`KnowledgeStore` and nothing upstream changes.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain import (
    Claim,
    ClaimLink,
    Concept,
    Document,
    EvidenceLink,
    EntityType,
    Revision,
    Source,
    Span,
)


@runtime_checkable
class SourceStore(Protocol):
    """Sources, documents, and spans — the evidence chain."""

    def put_source(self, source: Source) -> None: ...
    def get_source(self, source_id: str) -> Source | None: ...
    def get_source_by_locator(self, locator: str) -> Source | None: ...
    def list_sources(self) -> Sequence[Source]: ...
    def delete_source(self, source_id: str) -> None: ...

    def put_document(self, document: Document) -> None: ...
    def get_document(self, document_id: str) -> Document | None: ...
    def documents_for_source(self, source_id: str) -> Sequence[Document]: ...

    def put_spans(self, spans: Sequence[Span]) -> None: ...
    def get_span(self, span_id: str) -> Span | None: ...
    def spans_for_document(self, document_id: str) -> Sequence[Span]: ...


@runtime_checkable
class KnowledgeStore(Protocol):
    """Concepts, claims, and the reified links between them."""

    def put_concept(self, concept: Concept) -> None: ...
    def get_concept(self, concept_id: str) -> Concept | None: ...
    def get_concept_by_name(self, canonical_name: str) -> Concept | None: ...
    def list_concepts(self) -> Sequence[Concept]: ...

    def put_claim(self, claim: Claim, evidence: Sequence[EvidenceLink] = ()) -> None:
        """Persist a claim with its evidence.

        Implementations MUST call :func:`forge.domain.validate_claim` so that a
        claim requiring evidence cannot be stored without it, regardless of
        caller. This is the enforcement point for provenance at the storage
        boundary.
        """
        ...

    def get_claim(self, claim_id: str) -> Claim | None: ...
    def list_claims(self) -> Sequence[Claim]: ...
    def evidence_for_claim(self, claim_id: str) -> Sequence[EvidenceLink]: ...

    def supersede_claim(self, old_id: str, new_claim: Claim, *, cause: str | None = None) -> None:
        """Replace a claim non-destructively.

        The old claim MUST be retained with ``status=SUPERSEDED``, and a
        ``SUPERSEDE`` revision MUST be written recording both states.
        """
        ...

    def put_link(self, link: ClaimLink) -> None: ...
    def links_from(self, entity_id: str) -> Sequence[ClaimLink]: ...
    def links_to(self, entity_id: str) -> Sequence[ClaimLink]: ...


@runtime_checkable
class RevisionStore(Protocol):
    """Append-only history of derived state."""

    def append_revision(self, revision: Revision) -> None: ...
    def revisions_for(self, entity_type: EntityType, entity_id: str) -> Sequence[Revision]: ...
    def recent_revisions(self, limit: int = 50) -> Sequence[Revision]: ...
    def count_revisions(self) -> int: ...


@runtime_checkable
class Store(SourceStore, KnowledgeStore, RevisionStore, Protocol):
    """The full storage surface Phase 1 needs."""

    def initialize(self) -> None:
        """Create schema if absent. Must be idempotent."""
        ...

    def reset(self) -> None:
        """Drop all derived state.

        Safe by construction: everything here rebuilds from the Markdown vault.
        """
        ...

    def close(self) -> None: ...
