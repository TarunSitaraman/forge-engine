"""Static word-vector embeddings via spaCy.

**What this is.** Mean-pooled 300-dimensional GloVe-style word vectors from a
spaCy pipeline (`en_core_web_md` by default). Unlike
:class:`~forge.embeddings.hashing.HashingEmbeddingProvider`, these vectors are
*learned from a corpus*, so two passages that share no vocabulary can still
score as similar — which is the entire property the hashing provider lacks and
the reason this exists.

**What this is NOT.** It is not a modern sentence embedding. Mean-pooling
static word vectors discards word order and cannot represent a phrase whose
meaning differs from its parts' average, and any token outside the model's
20k-vector vocabulary contributes nothing. A transformer sentence encoder
(`nomic-embed-text`, `bge-*`) is strictly better and remains the intended
production provider; see :class:`OllamaEmbeddingProvider`.

**Why it exists.** The same reason the hashing provider does, one rung up. The
build environment blocks `huggingface.co` and `ollama.com`, so no transformer
encoder can be obtained — but spaCy's models ship as wheels from GitHub
releases, which is reachable. That makes it possible to answer the question the
hashing provider could not: *does semantic matching, as opposed to vocabulary
overlap, retrieve documents lexical search never finds?*

It is a floor for that question, not a ceiling. A gain here is evidence the
approach works and understates what a real encoder would do; no gain here is
weaker evidence of the opposite, because the instrument is weak.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..logging import get_logger

log = get_logger(__name__)

DEFAULT_SPACY_MODEL = "en_core_web_md"


class SpacyEmbeddingProvider:
    """Mean-pooled static word vectors from a spaCy pipeline."""

    def __init__(self, model: str = DEFAULT_SPACY_MODEL) -> None:
        self._model_name = model
        self._nlp: Any | None = None
        self._dimensions = 0
        self._load_failed = False

    @property
    def model_id(self) -> str:
        return f"spacy-{self._model_name}"

    @property
    def dimensions(self) -> int:
        self._ensure_loaded()
        return self._dimensions

    @property
    def available(self) -> bool:
        """Never raises: a missing model is an absent provider, not a crash."""
        self._ensure_loaded()
        return self._nlp is not None

    def _ensure_loaded(self) -> None:
        if self._nlp is not None or self._load_failed:
            return
        try:
            import spacy

            # Only the tokenizer and vectors are needed. Disabling the rest is
            # not a micro-optimization: the tagger and parser dominate runtime
            # and contribute nothing to a mean-pooled vector.
            nlp = spacy.load(self._model_name, exclude=["tagger", "parser", "ner", "lemmatizer"])
        except Exception as exc:
            log.info("spacy_unavailable", model=self._model_name, error=str(exc))
            self._load_failed = True
            return

        vectors = nlp.vocab.vectors
        if vectors.shape[0] == 0:
            # `en_core_web_sm` loads happily and has no vectors at all; its
            # `.vector` is a context-sensitive tensor, not an embedding, and
            # every similarity would be meaningless rather than wrong-looking.
            log.info("spacy_model_has_no_vectors", model=self._model_name)
            self._load_failed = True
            return

        self._nlp = nlp
        self._dimensions = int(vectors.shape[1])

    def embed(self, texts: Sequence[str], *, task: str = "document") -> list[list[float]]:
        """`task` is ignored: static vectors have no query/document asymmetry."""
        self._ensure_loaded()
        if self._nlp is None:
            return [[0.0] * max(self._dimensions, 1) for _ in texts]

        out: list[list[float]] = []
        # `pipe` batches; per-text `nlp()` calls are several times slower over
        # a corpus this size.
        for doc in self._nlp.pipe(texts, batch_size=64):
            vector = doc.vector
            if vector is None or not vector.any():
                out.append([0.0] * self._dimensions)
                continue
            norm = float((vector**2).sum()) ** 0.5
            # L2-normalize so cosine is a dot product, matching every other
            # provider's contract.
            out.append([float(v) / norm for v in vector] if norm else [0.0] * self._dimensions)
        return out
