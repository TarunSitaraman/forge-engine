"""Optional local embedding provider abstraction.

**Embeddings are never required.** Ingestion, retrieval, and concept matching
all work without them. This is the documented degradation mode:

===========================  ==========================================
With embeddings              Without embeddings
===========================  ==========================================
Hybrid lexical + semantic    Lexical (FTS5/BM25) only
Semantic concept candidates  Exact / alias / normalized / lexical only
===========================  ==========================================

Nothing silently changes behaviour: :meth:`EmbeddingProvider.available`
reports the state, and callers surface it rather than pretending the semantic
signal was consulted.

No Qdrant. Vectors live in SQLite, which at this corpus size is not a
compromise — brute-force cosine over a few thousand vectors is milliseconds.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Optional throughout."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def available(self) -> bool:
        """Whether this provider can currently serve requests. Never raises."""
        ...

    def embed(self, texts: Sequence[str], *, task: str = "document") -> list[list[float]]:
        """`task` is "document" or "query"; providers that do not care ignore it."""
        ...


class NullEmbeddingProvider:
    """The always-absent provider — the default.

    Exists so callers never branch on ``provider is None``: they ask
    ``available`` and get a consistent answer either way.
    """

    @property
    def model_id(self) -> str:
        return "none"

    @property
    def dimensions(self) -> int:
        return 0

    @property
    def available(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, task: str = "document") -> list[list[float]]:
        raise RuntimeError(
            "no embedding provider configured; lexical retrieval and deterministic "
            "matching remain available"
        )
