"""Derivation keys — the cost-control primitive.

    SOURCE HASH + PROCESSOR VERSION + MODEL ID + PROMPT/SCHEMA VERSION
    = DERIVATION KEY

If any component changes, the key changes and the derived result is recomputed.
If none changed, the cached result is reused and the expensive work — LLM calls
above all — is skipped entirely.

This is deliberately **not** a generic caching framework. It is one function
that builds a key and a small typed record, because that is all Phase 2 needs
and a framework would be speculative machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ids import deterministic_id


@dataclass(frozen=True)
class DerivationKey:
    """Everything that can invalidate a derived result."""

    kind: str
    content_hash: str
    processor_version: str
    model_id: str = "none"
    prompt_version: str = "none"
    schema_version: str = "none"

    def value(self) -> str:
        return deterministic_id(
            "derivation",
            self.kind,
            self.content_hash,
            self.processor_version,
            self.model_id,
            self.prompt_version,
            self.schema_version,
        )

    def describe(self) -> dict[str, str]:
        """Components, for logs and for explaining a cache miss."""
        return {
            "kind": self.kind,
            "content_hash": self.content_hash[:12],
            "processor_version": self.processor_version,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass
class CacheStats:
    """Per-run cache accounting, surfaced in the ingestion report."""

    hits: int = 0
    misses: int = 0
    writes: int = 0

    def hit(self) -> None:
        self.hits += 1

    def miss(self) -> None:
        self.misses += 1

    def write(self) -> None:
        self.writes += 1

    def to_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


def extraction_key(
    content_hash: str,
    processor_version: str,
    model_id: str,
    prompt_version: str,
    schema_version: str,
) -> DerivationKey:
    return DerivationKey(
        kind="extraction",
        content_hash=content_hash,
        processor_version=processor_version,
        model_id=model_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )


def embedding_key(content_hash: str, model_id: str, dimensions: int) -> DerivationKey:
    return DerivationKey(
        kind="embedding",
        content_hash=content_hash,
        processor_version=f"dim{dimensions}",
        model_id=model_id,
    )


def parse_key(content_hash: str, processor_version: str) -> DerivationKey:
    return DerivationKey(
        kind="parse", content_hash=content_hash, processor_version=processor_version
    )


def as_payload(data: Any) -> dict[str, Any]:
    """Normalize a cached value into a JSON-storable envelope."""
    return data if isinstance(data, dict) else {"value": data}
