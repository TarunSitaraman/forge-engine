"""Optional local embeddings. Never required."""

from .base import EmbeddingProvider, NullEmbeddingProvider
from .hashing import DEFAULT_DIMENSIONS, MODEL_ID, HashingEmbeddingProvider
from .ollama_embeddings import DEFAULT_EMBED_MODEL, OllamaEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "HashingEmbeddingProvider",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_DIMENSIONS",
    "MODEL_ID",
]
