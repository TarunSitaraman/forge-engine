"""Concept candidate matching. Produces candidates; never merges."""

from .matcher import (
    MATCHER_VERSION,
    ConceptMatcher,
    MatchCandidate,
    MatchResult,
    build_ambiguity_index,
    cosine,
)

__all__ = [
    "ConceptMatcher",
    "MatchResult",
    "MatchCandidate",
    "build_ambiguity_index",
    "cosine",
    "MATCHER_VERSION",
]
