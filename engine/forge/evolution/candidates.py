"""Deterministic candidate narrowing — what might this evidence affect?

**The rule this module exists to enforce:** the LLM is never handed the corpus
and asked what is relevant. It is handed a small, already-justified candidate
set. That is not only a cost decision — though it is that too, since scanning
600 documents per model call is unaffordable — it is a groundedness decision.
A model asked "what in my knowledge base does this affect?" will answer
fluently and unverifiably. A model asked "does this evidence bear on *this*
claim?" can be checked.

Every candidate carries the **selector** that found it and a human-readable
detail. A candidate set with no justification is indistinguishable from a
guess, and this one has to survive review.

Selectors, in the order they are applied (cheapest and most certain first):

===================  ==========================================================
``exact_name``       The concept's canonical name appears in the evidence text.
``alias``            A registered alias appears in the evidence text.
``heading``          The name appears in a heading of the new document.
``identity``         A user-decided collision identity matched.
``lexical``          FTS5/BM25 over spans surfaced the concept's own material.
``graph_neighbour``  Bounded traversal from an already-selected concept.
===================  ==========================================================

Embeddings are consulted only when a provider is genuinely available, and are
not used at all by default — Phase 3 measured them as a retrieval regression
(``docs/research/retrieval-baseline.md``), so switching them on here without
new evidence would contradict a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..domain import CandidateRecord, Concept, Span
from ..graph import KnowledgeGraph
from ..identity import IdentityService
from ..logging import get_logger
from ..parsing.links import normalize
from ..retrieval.search import SearchQuery, SearchService
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

NARROWING_VERSION = "candidates/0.1.0"

#: A name shorter than this matches inside unrelated words often enough to be
#: noise ("RAG" inside "storage"). Same threshold and same reasoning as the
#: Phase 3 relationship activator.
MIN_NAME_CHARS = 4

#: How many concepts one run may consider. The cap is a cost ceiling: every
#: candidate becomes claims, and every claim becomes part of a model prompt.
DEFAULT_MAX_CANDIDATES = 12

#: Traversal depth for graph expansion. One hop: a neighbour of a neighbour is
#: too weak a reason to spend a semantic assessment on.
GRAPH_DEPTH = 1


@dataclass
class NarrowingResult:
    """Candidates plus the accounting needed to explain them."""

    candidates: list[CandidateRecord] = field(default_factory=list)
    considered: int = 0
    by_selector: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def concept_ids(self) -> list[str]:
        return [c.concept_id for c in self.candidates]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "considered": self.considered,
            "by_selector": dict(sorted(self.by_selector.items())),
            "truncated": self.truncated,
        }


class CandidateNarrower:
    """Finds which canonical concepts new evidence might affect.

    Entirely deterministic. Makes **zero** LLM calls — asserted in tests, since
    "deterministic first" is a claim that decays silently if nothing checks it.
    """

    version = NARROWING_VERSION

    def __init__(
        self,
        store: SqliteStore,
        *,
        identity: IdentityService | None = None,
        search: SearchService | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self.store = store
        self.identity = identity or IdentityService()
        self.search = search or SearchService(store)
        self.graph = KnowledgeGraph(store)
        self.max_candidates = max_candidates

    def narrow(self, spans: Sequence[Span]) -> NarrowingResult:
        """Select concepts the given evidence spans might bear on."""
        result = NarrowingResult()
        if not spans:
            return result

        concepts = list(self.store.list_concepts())
        result.considered = len(concepts)
        if not concepts:
            return result

        haystack = normalize(" ".join(s.text for s in spans))
        headings = normalize(
            " ".join(part for s in spans for part in s.heading_path)
        )
        # Preserve first-wins order: a concept found by exact name should keep
        # that stronger justification even if lexical search also surfaces it.
        chosen: dict[str, CandidateRecord] = {}

        def take(concept: Concept, selector: str, detail: str, score: float | None = None) -> None:
            if concept.id in chosen:
                return
            chosen[concept.id] = CandidateRecord(
                concept_id=concept.id,
                concept_name=concept.qualified_name,
                selector=selector,
                detail=detail,
                score=score,
            )

        # 1-2. Name and alias matching over the evidence text.
        for concept in concepts:
            name = normalize(concept.canonical_name)
            if len(name) >= MIN_NAME_CHARS and name in haystack:
                take(concept, "exact_name", f"{concept.canonical_name!r} appears in the evidence")
                continue
            for alias in concept.aliases:
                normalized_alias = normalize(alias)
                if len(normalized_alias) >= MIN_NAME_CHARS and normalized_alias in haystack:
                    take(concept, "alias", f"alias {alias!r} appears in the evidence")
                    break

        # 3. Headings carry more weight than body text, but are checked after
        # body matches so the stronger `exact_name` justification wins.
        for concept in concepts:
            name = normalize(concept.canonical_name)
            if len(name) >= MIN_NAME_CHARS and name in headings:
                take(concept, "heading", f"{concept.canonical_name!r} appears in a heading")

        # 4. User-decided identities. If the evidence mentions a colliding bare
        # name the user has resolved, the resolved concept is a candidate —
        # this is the identity config doing work rather than sitting inert.
        for name, qualified in self.identity.resolved_names().items():
            if len(name) >= MIN_NAME_CHARS and normalize(name) in haystack:
                for concept in concepts:
                    if concept.qualified_name == qualified:
                        take(
                            concept,
                            "identity",
                            f"bare name {name!r} resolves to {qualified!r} by user decision",
                        )

        # 5. Lexical retrieval. Uses the evidence's own headings as the query,
        # which is what a person would type — the section titles are the
        # densest available summary of what the new material is about.
        for concept in self._lexical(spans, concepts):
            take(*concept)

        # 6. Graph expansion from what we already have, bounded.
        for concept, origin in self._neighbours(list(chosen)):
            take(concept, "graph_neighbour", f"related to {origin!r} in the knowledge graph")

        ordered = sorted(chosen.values(), key=lambda c: (_RANK[c.selector], c.concept_name))
        result.truncated = len(ordered) > self.max_candidates
        result.candidates = ordered[: self.max_candidates]
        for candidate in result.candidates:
            result.by_selector[candidate.selector] = (
                result.by_selector.get(candidate.selector, 0) + 1
            )

        log.info(
            "candidates_narrowed",
            considered=result.considered,
            selected=len(result.candidates),
            truncated=result.truncated,
            **result.by_selector,
        )
        return result

    # -- selectors ---------------------------------------------------------

    def _lexical(
        self, spans: Sequence[Span], concepts: Sequence[Concept]
    ) -> list[tuple[Concept, str, str, float]]:
        """Concepts whose own name matches the evidence's strongest phrases.

        Deliberately narrow: FTS is used to rank *phrases from the evidence*,
        and a concept is taken only when the phrase actually names it. Using
        BM25 alone to decide conceptual relevance would reintroduce exactly
        the ungrounded relevance-guessing this module avoids.
        """
        queries = [p for s in spans for p in s.heading_path if len(p) >= MIN_NAME_CHARS]
        if not queries:
            return []

        by_name = {normalize(c.canonical_name): c for c in concepts}
        found: list[tuple[Concept, str, str, float]] = []
        seen: set[str] = set()
        for query in queries[:8]:
            hits = self.search.search(SearchQuery(text=query, limit=5))
            if not hits:
                continue
            normalized_query = normalize(query)
            for name, concept in by_name.items():
                if concept.id in seen or len(name) < MIN_NAME_CHARS:
                    continue
                if name in normalized_query or normalized_query in name:
                    seen.add(concept.id)
                    found.append(
                        (
                            concept,
                            "lexical",
                            f"section {query!r} matches this concept ({len(hits)} span hits)",
                            round(hits[0].score, 3),
                        )
                    )
        return found

    def _neighbours(self, concept_ids: Sequence[str]) -> list[tuple[Concept, str]]:
        """One bounded hop out from the concepts already selected."""
        out: list[tuple[Concept, str]] = []
        for concept_id in list(concept_ids)[: self.max_candidates]:
            origin = self.store.get_concept(concept_id)
            if origin is None:
                continue
            for concept, _distance in self.graph.get_related_concepts(
                concept_id, max_depth=GRAPH_DEPTH
            ):
                out.append((concept, origin.qualified_name))
        return out


#: Selector precedence for ordering. Lower is stronger evidence of relevance.
_RANK: dict[str, int] = {
    "exact_name": 0,
    "alias": 1,
    "identity": 2,
    "heading": 3,
    "lexical": 4,
    "graph_neighbour": 5,
}
