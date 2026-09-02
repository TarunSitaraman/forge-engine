"""Optional local embeddings. Never required."""

from .base import EmbeddingProvider, NullEmbeddingProvider
from .hashing import DEFAULT_DIMENSIONS, MODEL_ID, HashingEmbeddingProvider
from .ollama_embeddings import DEFAULT_EMBED_MODEL, OllamaEmbeddingProvider
from .spacy_vectors import DEFAULT_SPACY_MODEL, SpacyEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "HashingEmbeddingProvider",
    "SpacyEmbeddingProvider",
    "DEFAULT_SPACY_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_DIMENSIONS",
    "MODEL_ID",
]
