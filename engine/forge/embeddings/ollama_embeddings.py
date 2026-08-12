"""Local embeddings via Ollama.

Uses Ollama's ``/api/embed`` endpoint. No key, no account, no paid service —
the same local-first guarantee as the generation provider.

Dimensionality is discovered from the first successful call rather than
hardcoded, because it is a property of the model, and a wrong constant would
silently corrupt every stored vector.
"""

from __future__ import annotations

from typing import Sequence

import httpx

from ..logging import get_logger

log = get_logger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaEmbeddingProvider:
    """Embeddings from a local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_EMBED_MODEL,
        *,
        timeout: float = 60.0,
        batch_size: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.batch_size = batch_size
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._dimensions = 0
        self._available: bool | None = None

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        """Discovered on first use; 0 until then."""
        return self._dimensions

    @property
    def available(self) -> bool:
        """Whether Ollama is reachable *and* has this embedding model pulled.

        Cached after the first check so a run does not probe repeatedly. Never
        raises — an unavailable provider is a normal state, not an error.
        """
        if self._available is not None:
            return self._available
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = {m.get("name", "") for m in response.json().get("models", [])}
            self._available = any(
                name == self._model or name.startswith(f"{self._model}:") for name in models
            )
            if not self._available:
                log.info(
                    "embedding_model_absent",
                    model=self._model,
                    hint=f"run: ollama pull {self._model}",
                )
        except Exception as exc:
            log.info("embedding_provider_unavailable", error=str(exc)[:120])
            self._available = False
        return self._available

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches. Raises only on a genuine transport failure."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.post(
                "/api/embed", json={"model": self._model, "input": batch}
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings", [])
            if len(vectors) != len(batch):  # pragma: no cover - provider contract
                raise RuntimeError(
                    f"embedding count mismatch: sent {len(batch)}, got {len(vectors)}"
                )
            out.extend([[float(x) for x in v] for v in vectors])

        if out and not self._dimensions:
            self._dimensions = len(out[0])
        return out

    def close(self) -> None:
        self._client.close()
