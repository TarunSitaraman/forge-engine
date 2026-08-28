"""Deterministic hashing embeddings — a zero-dependency, zero-download vector.

**What this is, precisely.** A hashed bag-of-features vector over word tokens
and character 4-grams, with sublinear term frequency and L2 normalization. It
is a *lexical-statistical* representation.

**What this is NOT.** It is not a neural sentence embedding, and it does not
capture meaning. Two passages that share no vocabulary will score near zero no
matter how synonymous they are. Anyone reading a "semantic retrieval" number
produced by this provider should read it as "character-and-token overlap in
vector form", nothing more.

**Why it exists.** Phase 3 requires that embeddings be *measured* rather than
assumed to help. No neural model could be obtained in this environment (the
sandbox network policy blocks both ollama.com and huggingface.co), so without
this provider the entire embedding pathway — storage, cache invalidation,
fusion, evaluation — would ship untested and unmeasured. With it, the pathway
is exercised end to end and produces real numbers, while
:class:`OllamaEmbeddingProvider` remains the intended production provider.

Its measured results are reported honestly, including when they are worse than
lexical search.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

MODEL_ID = "hashing-v1"
DEFAULT_DIMENSIONS = 256
CHAR_NGRAM = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Small stopword list. Without IDF weighting these terms would otherwise
#: dominate every vector and make all documents look alike.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in into is it its of on or
    that the this to was were what when where which who why will with without
    you your can could should would may might must not no nor but if then than
    there here about after before over under between each such only own same so
    very too much many more most other some any all both few own""".split()
)


class HashingEmbeddingProvider:
    """Deterministic lexical-statistical embeddings. Always available."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, *, char_ngrams: bool = True) -> None:
        self._dimensions = dimensions
        self.char_ngrams = char_ngrams

    @property
    def model_id(self) -> str:
        # Dimensionality and feature configuration are part of the identity, so
        # changing either invalidates cached vectors rather than silently
        # mixing incompatible ones.
        suffix = "c" if self.char_ngrams else "w"
        return f"{MODEL_ID}-{self._dimensions}{suffix}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def available(self) -> bool:
        """Always true: nothing to download, nothing to reach."""
        return True

    def embed(self, texts: Sequence[str], *, task: str = "document") -> list[list[float]]:
        return [self._vector(text) for text in texts]

    # -- internals ---------------------------------------------------------

    def _vector(self, text: str) -> list[float]:
        counts = Counter(self._features(text))
        if not counts:
            return [0.0] * self._dimensions

        vector = [0.0] * self._dimensions
        for feature, count in counts.items():
            index = _bucket(feature, self._dimensions)
            # Sublinear TF: a term appearing ten times is not ten times as
            # important as one appearing once.
            weight = 1.0 + math.log(count)
            # Signed hashing cancels some collision bias instead of letting
            # every collision add constructively.
            vector[index] += weight * _sign(feature)

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = [t for t in _TOKEN_RE.findall(lowered) if t not in _STOPWORDS and len(t) > 1]

        features = list(tokens)
        if self.char_ngrams:
            # Character n-grams give partial credit for morphological variants
            # ("chunking" / "chunked") that exact token matching misses.
            joined = " ".join(tokens)
            features.extend(
                joined[i : i + CHAR_NGRAM] for i in range(max(0, len(joined) - CHAR_NGRAM + 1))
            )
        return features


def _bucket(feature: str, dimensions: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _sign(feature: str) -> float:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=1, person=b"sign").digest()
    return 1.0 if digest[0] & 1 else -1.0
