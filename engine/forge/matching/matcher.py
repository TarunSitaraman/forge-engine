"""Concept candidate matching.

This is **not** graph construction and **not** entity resolution. It answers one
question about an extracted concept name:

    Have I seen this before, might I have seen it before, or is it new?

producing exactly one of ``NEW_CONCEPT``, ``MATCH_CANDIDATE``, ``AMBIGUOUS``.

There is deliberately no ``MERGED`` outcome. Deciding that two concepts are the
same is a human judgement, and getting it wrong silently rewrites the meaning of
someone's knowledge base. The matcher's strongest possible output is a proposal.

The corpus makes this concrete. `Heap`, `Binary Search`, and `Trie` each exist
twice in the vault — once as a pattern, once as an algorithm or data structure —
accounting for 180 of its 282 unresolved links. Any matcher that picks one of
those two homes is wrong half the time and never says so. So a name with more
than one canonical home in the vault is ``AMBIGUOUS`` by construction, before
any similarity scoring runs at all.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence

from ..domain import Concept, MatchKind
from ..logging import get_logger
from ..parsing.links import normalize

log = get_logger(__name__)

MATCHER_VERSION = "matcher/0.2.0"

#: Lexical ratio at/above which a name is offered as a candidate.
LEXICAL_THRESHOLD = 0.86
#: Cosine similarity at/above which an embedding neighbour is offered.
EMBEDDING_THRESHOLD = 0.88
#: If the top two candidates are within this margin, the result is ambiguous
#: rather than a pick. Closeness is a reason to ask, not a reason to guess.
AMBIGUITY_MARGIN = 0.05


@dataclass
class MatchCandidate:
    """One possible existing concept for an extracted name."""

    concept_id: str | None
    canonical_name: str
    #: Which signal produced it: exact | alias | normalized | lexical | embedding | vault_path
    signal: str
    score: float
    vault_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "canonical_name": self.canonical_name,
            "signal": self.signal,
            "score": round(self.score, 4),
            "vault_path": self.vault_path,
        }


@dataclass
class MatchResult:
    """Outcome of matching one extracted concept name."""

    name: str
    kind: MatchKind
    candidates: list[MatchCandidate] = field(default_factory=list)
    reason: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return self.kind is MatchKind.AMBIGUOUS

    @property
    def best(self) -> MatchCandidate | None:
        """Top candidate — meaningful only for MATCH_CANDIDATE.

        Returns ``None`` when ambiguous. Callers cannot accidentally treat an
        unresolved collision as a resolved match.
        """
        if self.kind is not MatchKind.MATCH_CANDIDATE:
            return None
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


#: Filenames that name a *document's role*, not a concept. A vault has many
#: files called `_index` or `README` or `01-overview`; that is organization,
#: not a concept collision, and treating it as one would flood the ambiguity
#: index with noise. Anything genuinely conceptual is still included.
_STRUCTURAL_STEM = re.compile(r"^(_.*|readme|index|roadmap|\d+[-_].*)$", re.IGNORECASE)


def build_ambiguity_index(
    paths: Iterable[str], *, include_structural: bool = False
) -> dict[str, list[str]]:
    """Map normalized file stem -> vault paths, for stems with more than one home.

    Built from vault paths rather than from stored concepts, because the
    collision exists in the corpus whether or not concepts have been created
    for it yet.

    Structural filenames are excluded by default (see ``_STRUCTURAL_STEM``).
    Pass ``include_structural=True`` to see the raw picture.
    """
    by_stem: dict[str, list[str]] = {}
    for path in paths:
        stem = PurePosixPath(path).stem
        if not include_structural and _STRUCTURAL_STEM.match(stem.strip()):
            continue
        by_stem.setdefault(normalize(stem), []).append(path)
    return {stem: sorted(hits) for stem, hits in by_stem.items() if len(hits) > 1}


class ConceptMatcher:
    """Finds candidate matches for extracted concept names."""

    version = MATCHER_VERSION

    def __init__(
        self,
        concepts: Sequence[Concept] = (),
        *,
        ambiguity_index: dict[str, list[str]] | None = None,
        proposed_concepts: Sequence[tuple[str, str]] = (),
        embeddings: dict[str, list[float]] | None = None,
        lexical_threshold: float = LEXICAL_THRESHOLD,
        embedding_threshold: float = EMBEDDING_THRESHOLD,
    ) -> None:
        self.concepts = list(concepts)
        self.ambiguity_index = ambiguity_index or {}
        #: ``(name, proposal_id)`` for concepts already *proposed* but not yet
        #: accepted. In Phase 2 nothing creates a Concept — extraction produces
        #: proposals — so without this, every source would rediscover the same
        #: concept as brand new and the second document could never recognise
        #: the first. Matching against pending proposals is what makes
        #: "existing concepts detected" true before any approval workflow runs.
        self.proposed_concepts = list(proposed_concepts)
        self._by_proposed: dict[str, list[tuple[str, str]]] = {}
        for name, proposal_id in self.proposed_concepts:
            self._by_proposed.setdefault(normalize(name), []).append((name, proposal_id))
        #: concept_id -> vector. Absent means embedding matching is skipped and
        #: lexical matching carries the load. This is the documented degradation
        #: mode, not an error.
        self.embeddings = embeddings or {}
        self.lexical_threshold = lexical_threshold
        self.embedding_threshold = embedding_threshold

        self._by_exact = {c.canonical_name.casefold(): c for c in self.concepts}
        self._by_normalized: dict[str, list[Concept]] = {}
        self._by_alias: dict[str, list[Concept]] = {}
        for concept in self.concepts:
            self._by_normalized.setdefault(normalize(concept.canonical_name), []).append(concept)
            for alias in concept.aliases:
                self._by_alias.setdefault(alias.casefold(), []).append(concept)

    @property
    def embeddings_available(self) -> bool:
        return bool(self.embeddings)

    def match(self, name: str, *, query_vector: Sequence[float] | None = None) -> MatchResult:
        """Classify one extracted concept name."""
        cleaned = name.strip()
        if not cleaned:
            return MatchResult(name=name, kind=MatchKind.NEW_CONCEPT, reason="empty name")

        normalized = normalize(cleaned)

        # 1. Vault collisions dominate everything. If the corpus itself holds
        #    two canonical homes for this name, no similarity score can resolve
        #    which one was meant.
        if colliding := self.ambiguity_index.get(normalized):
            return MatchResult(
                name=cleaned,
                kind=MatchKind.AMBIGUOUS,
                candidates=[
                    MatchCandidate(
                        concept_id=self._id_for_path(path),
                        canonical_name=PurePosixPath(path).stem,
                        signal="vault_path",
                        score=1.0,
                        vault_path=path,
                    )
                    for path in colliding
                ],
                reason=(
                    f"{len(colliding)} canonical homes exist for this name in the vault; "
                    f"the intended one cannot be determined automatically"
                ),
            )

        # 2. Exact canonical name.
        if (exact := self._by_exact.get(cleaned.casefold())) is not None:
            return MatchResult(
                name=cleaned,
                kind=MatchKind.MATCH_CANDIDATE,
                candidates=[_candidate(exact, "exact", 1.0)],
                reason="exact canonical name match",
            )

        # 3. Already proposed from another source. Not a stored concept yet,
        #    but definitely not new either — saying "new" would duplicate it.
        if proposed := self._by_proposed.get(normalized):
            if len(proposed) == 1:
                name, proposal_id = proposed[0]
                return MatchResult(
                    name=cleaned,
                    kind=MatchKind.MATCH_CANDIDATE,
                    candidates=[
                        MatchCandidate(
                            concept_id=None,
                            canonical_name=name,
                            signal="proposed",
                            score=0.99,
                        )
                    ],
                    reason=f"already proposed from another source (proposal {proposal_id[:10]})",
                )
            return MatchResult(
                name=cleaned,
                kind=MatchKind.AMBIGUOUS,
                candidates=[
                    MatchCandidate(concept_id=None, canonical_name=n, signal="proposed", score=0.99)
                    for n, _ in proposed
                ],
                reason=f"{len(proposed)} distinct proposals already exist for this name",
            )

        # 4. Alias.
        if aliased := self._by_alias.get(cleaned.casefold()):
            if len(aliased) == 1:
                return MatchResult(
                    name=cleaned,
                    kind=MatchKind.MATCH_CANDIDATE,
                    candidates=[_candidate(aliased[0], "alias", 0.98)],
                    reason="matched a registered alias",
                )
            return MatchResult(
                name=cleaned,
                kind=MatchKind.AMBIGUOUS,
                candidates=[_candidate(c, "alias", 0.98) for c in aliased],
                reason=f"alias is registered on {len(aliased)} concepts",
            )

        # 4. Normalized (punctuation/spacing-insensitive).
        if normed := self._by_normalized.get(normalized):
            if len(normed) == 1:
                return MatchResult(
                    name=cleaned,
                    kind=MatchKind.MATCH_CANDIDATE,
                    candidates=[_candidate(normed[0], "normalized", 0.95)],
                    reason="matches after normalizing punctuation and spacing",
                )
            return MatchResult(
                name=cleaned,
                kind=MatchKind.AMBIGUOUS,
                candidates=[_candidate(c, "normalized", 0.95) for c in normed],
                reason=f"{len(normed)} concepts normalize to the same form",
            )

        # 5. Fuzzy signals: lexical always, embeddings when available.
        scored = self._lexical(cleaned)
        if query_vector is not None and self.embeddings:
            scored.extend(self._semantic(query_vector))

        if not scored:
            return MatchResult(
                name=cleaned,
                kind=MatchKind.NEW_CONCEPT,
                reason="no existing concept matched on any signal",
            )

        best_per_concept: dict[str, MatchCandidate] = {}
        for candidate in scored:
            key = candidate.concept_id or candidate.canonical_name
            if key not in best_per_concept or candidate.score > best_per_concept[key].score:
                best_per_concept[key] = candidate
        ranked = sorted(best_per_concept.values(), key=lambda c: -c.score)

        # Near-ties are ambiguous. A 0.87-vs-0.86 "win" is noise, not a decision.
        if len(ranked) > 1 and (ranked[0].score - ranked[1].score) < AMBIGUITY_MARGIN:
            return MatchResult(
                name=cleaned,
                kind=MatchKind.AMBIGUOUS,
                candidates=ranked[:5],
                reason=(
                    f"top candidates score within {AMBIGUITY_MARGIN} of each other "
                    f"({ranked[0].score:.2f} vs {ranked[1].score:.2f})"
                ),
            )

        return MatchResult(
            name=cleaned,
            kind=MatchKind.MATCH_CANDIDATE,
            candidates=ranked[:5],
            reason=f"{ranked[0].signal} similarity {ranked[0].score:.2f}",
        )

    def match_all(self, names: Iterable[str]) -> list[MatchResult]:
        return [self.match(n) for n in names]

    # -- signals -----------------------------------------------------------

    def _lexical(self, name: str) -> list[MatchCandidate]:
        out: list[MatchCandidate] = []
        for concept in self.concepts:
            ratio = difflib.SequenceMatcher(
                None, name.casefold(), concept.canonical_name.casefold()
            ).ratio()
            if ratio >= self.lexical_threshold:
                out.append(_candidate(concept, "lexical", ratio))
        return out

    def _semantic(self, vector: Sequence[float]) -> list[MatchCandidate]:
        out: list[MatchCandidate] = []
        by_id = {c.id: c for c in self.concepts}
        for concept_id, other in self.embeddings.items():
            concept = by_id.get(concept_id)
            if concept is None:
                continue
            score = cosine(vector, other)
            if score >= self.embedding_threshold:
                out.append(_candidate(concept, "embedding", score))
        return out

    def _id_for_path(self, path: str) -> str | None:
        for concept in self.concepts:
            if concept.vault_path == path:
                return concept.id
        return None


# -- helpers ---------------------------------------------------------------


def _candidate(concept: Concept, signal: str, score: float) -> MatchCandidate:
    return MatchCandidate(
        concept_id=concept.id,
        canonical_name=concept.canonical_name,
        signal=signal,
        score=score,
        vault_path=concept.vault_path,
    )


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 for zero or mismatched vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
