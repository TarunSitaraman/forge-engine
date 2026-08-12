"""Relationship activation — evidence-gated, deliberately stingy.

The failure mode this module exists to prevent is **relationship spam**: a
graph where every concept is `RELATED_TO` every other concept, which carries
no information and makes traversal useless. The rule is simple and absolute:

    If the evidence is insufficient, the relationship is not created.

Concretely, a relationship requires all of:

1. **Both endpoints are canonical concepts.** No edges to names that were
   merely mentioned.
2. **Co-occurrence in at least one span**, or an explicit user assertion.
   Two concepts that never appear together are not related just because a
   model said so.
3. **A supported type** from a small vocabulary. Five types, no more.
4. **Provenance and evidence spans.**
5. **Deterministic identity**, so re-running converges instead of duplicating.

``RELATED_TO`` additionally requires a co-occurrence *count* above a floor and
carries its score, because it is the type most prone to becoming noise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..domain import (
    ClaimLink,
    Concept,
    Derivation,
    EntityType,
    LinkType,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    Span,
)
from ..logging import get_logger
from ..parsing.links import normalize
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

RELATIONSHIP_VERSION = "relationships/0.3.0"

#: The whole Phase 3 vocabulary. Small on purpose: five types that a reviewer
#: can hold in their head, rather than thirteen that blur together.
SUPPORTED_TYPES: frozenset[LinkType] = frozenset(
    {
        LinkType.RELATED_TO,
        LinkType.PART_OF,
        LinkType.DEPENDS_ON,
        LinkType.IMPLEMENTS,
        LinkType.EXPLAINS,
    }
)

#: Minimum spans two concepts must share before `RELATED_TO` is justified.
#: One shared span is a coincidence; two is a pattern. Raising this is the
#: first lever to pull if the graph ever gets noisy.
MIN_COOCCURRENCE = 2


@dataclass
class RelationshipCandidate:
    """A relationship proposed for activation, with its evidence."""

    from_concept_id: str
    to_concept_id: str
    type: LinkType
    span_ids: tuple[str, ...]
    score: float | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_concept_id,
            "to": self.to_concept_id,
            "type": self.type.value,
            "spans": list(self.span_ids),
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass
class RelationshipReport:
    created: int = 0
    already_present: int = 0
    rejected: list[dict[str, str]] = field(default_factory=list)
    candidates_considered: int = 0

    def reject(self, candidate: RelationshipCandidate, reason: str) -> None:
        self.rejected.append(
            {
                "from": candidate.from_concept_id,
                "to": candidate.to_concept_id,
                "type": candidate.type.value,
                "reason": reason,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": self.candidates_considered,
            "created": self.created,
            "already_present": self.already_present,
            "rejected": len(self.rejected),
            "rejections": self.rejected[:20],
        }


class RelationshipActivator:
    """Creates concept relationships only where evidence supports them."""

    version = RELATIONSHIP_VERSION

    def __init__(self, store: SqliteStore, *, min_cooccurrence: int = MIN_COOCCURRENCE) -> None:
        self.store = store
        self.min_cooccurrence = min_cooccurrence

    # -- discovery ---------------------------------------------------------

    def discover_cooccurrence(self, *, limit: int | None = None) -> list[RelationshipCandidate]:
        """Find `RELATED_TO` candidates from concepts sharing spans.

        Deterministic: concept names (and their aliases) are matched against
        span text by substring, and pairs are emitted in sorted order. No LLM
        is involved — this is co-occurrence counting, which is arithmetic.
        """
        concepts = list(self.store.list_concepts())
        if len(concepts) < 2:
            return []

        spans_by_concept = self._concept_spans(concepts)
        pairs: dict[tuple[str, str], set[str]] = defaultdict(set)

        for i, first in enumerate(concepts):
            for second in concepts[i + 1 :]:
                shared = spans_by_concept[first.id] & spans_by_concept[second.id]
                if shared:
                    key = tuple(sorted((first.id, second.id)))  # type: ignore[assignment]
                    pairs[key] |= shared

        candidates = [
            RelationshipCandidate(
                from_concept_id=a,
                to_concept_id=b,
                type=LinkType.RELATED_TO,
                span_ids=tuple(sorted(spans)),
                score=round(min(1.0, len(spans) / 5.0), 4),
                rationale=f"co-occurs in {len(spans)} span(s)",
            )
            for (a, b), spans in sorted(pairs.items())
        ]
        return candidates[:limit] if limit else candidates

    def _concept_spans(self, concepts: Sequence[Concept]) -> dict[str, set[str]]:
        """Map concept id -> span ids whose text mentions it.

        Matching is done on *normalized* text (case, punctuation and spacing
        removed) so that "Retrieval-Augmented Generation" and "Retrieval
        Augmented Generation" count as the same mention. Plain substring
        matching silently misses the hyphenated form, which is the form real
        documents most often use.
        """
        spans = self._all_spans()
        out: dict[str, set[str]] = {c.id: set() for c in concepts}
        normalized_spans = [(span.id, normalize(span.text)) for span in spans]

        for concept in concepts:
            needles = {
                normalize(name)
                for name in (concept.canonical_name, *concept.aliases)
            }
            # Very short names produce runaway false positives once punctuation
            # is stripped ("AI" matches "chAIn"), so they are skipped rather
            # than allowed to manufacture relationships.
            needles = {n for n in needles if len(n) >= 6}
            if not needles:
                continue
            for span_id, text in normalized_spans:
                if any(needle in text for needle in needles):
                    out[concept.id].add(span_id)
        return out

    def _all_spans(self) -> list[Span]:
        spans: list[Span] = []
        for source in self.store.list_sources():
            for document in self.store.documents_for_source(source.id):
                spans.extend(self.store.spans_for_document(document.id))
        return spans

    # -- activation --------------------------------------------------------

    def activate(
        self, candidates: Iterable[RelationshipCandidate], *, model_id: str | None = None
    ) -> RelationshipReport:
        """Create relationships that clear every gate. Reject the rest, loudly."""
        report = RelationshipReport()

        for candidate in candidates:
            report.candidates_considered += 1

            if (reason := self._reject_reason(candidate)) is not None:
                report.reject(candidate, reason)
                continue

            link = self._build(candidate, model_id)
            if self.store.get_link(link.id) is not None:
                report.already_present += 1
                continue

            self.store.put_link(link)
            report.created += 1

        log.info("relationships_activated", **{k: v for k, v in report.to_dict().items() if k != "rejections"})
        return report

    def _reject_reason(self, candidate: RelationshipCandidate) -> str | None:
        if candidate.type not in SUPPORTED_TYPES:
            return (
                f"type {candidate.type.value} is outside the Phase 3 vocabulary "
                f"({sorted(t.value for t in SUPPORTED_TYPES)})"
            )
        if candidate.from_concept_id == candidate.to_concept_id:
            return "self-relationship carries no information"

        for concept_id in (candidate.from_concept_id, candidate.to_concept_id):
            if self.store.get_concept(concept_id) is None:
                return f"endpoint {concept_id} is not a canonical concept"

        if not candidate.span_ids:
            return "no evidence spans: a relationship without evidence is a guess"

        resolvable = [s for s in candidate.span_ids if self.store.get_span(s) is not None]
        if not resolvable:
            return f"none of the cited spans resolve ({list(candidate.span_ids)})"

        # RELATED_TO is the noise-prone type, so it carries the extra gate.
        if candidate.type is LinkType.RELATED_TO and len(resolvable) < self.min_cooccurrence:
            return (
                f"only {len(resolvable)} shared span(s); RELATED_TO requires at least "
                f"{self.min_cooccurrence} to distinguish a relationship from a coincidence"
            )
        return None

    def _build(self, candidate: RelationshipCandidate, model_id: str | None) -> ClaimLink:
        """Construct the edge with provenance matching how it was derived."""
        # Co-occurrence counting is arithmetic, so these edges are
        # deterministic — which is also what the domain layer requires, since
        # RELATED_TO is in DETERMINISTIC_LINK_TYPES.
        deterministic = candidate.type is LinkType.RELATED_TO and model_id is None
        provenance = Provenance(
            tier=ProvenanceTier.MODEL_INFERENCE
            if not deterministic
            else ProvenanceTier.EXTRACTED_CLAIM,
            derivation=Derivation.DETERMINISTIC if deterministic else Derivation.MODEL,
            agent="RelationshipActivator",
            agent_version=RELATIONSHIP_VERSION,
            model_id=None if deterministic else model_id,
            inputs=tuple(
                ProvenanceInput(
                    entity_type=EntityType.SPAN, entity_id=span_id, tier=ProvenanceTier.SOURCE_FACT
                )
                for span_id in candidate.span_ids
            ),
        )
        return ClaimLink(
            id=ClaimLink.make_id(
                candidate.from_concept_id, candidate.to_concept_id, candidate.type
            ),
            from_id=candidate.from_concept_id,
            to_id=candidate.to_concept_id,
            type=candidate.type,
            score=candidate.score,
            rationale=candidate.rationale,
            provenance=provenance,
        )
