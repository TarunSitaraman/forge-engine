"""Optional local embeddings. Never required."""

from .base import EmbeddingProvider, NullEmbeddingProvider
from .ollama_embeddings import DEFAULT_EMBED_MODEL, OllamaEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "DEFAULT_EMBED_MODEL",
]
