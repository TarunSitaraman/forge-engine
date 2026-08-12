"""Retrieval — Phase 2 scope only.

Implemented: lexical search, metadata filtering, source filtering, and concept /
document / span lookup. Optional semantic re-ranking when embeddings exist.

**Not** implemented, deliberately: a chatbot, a conversational interface, or
generation over retrieved text. Retrieval here returns *evidence with
provenance*, not prose.

The API shape is the point. Callers ask :meth:`SearchService.search` with a
:class:`SearchQuery` and receive :class:`SearchHit` values. Adding vector or
graph retrieval later means adding a scorer inside this service — not changing
any caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..domain import Concept, Document, ProvenanceTier, Source, Span, TrustTier
from ..embeddings.base import EmbeddingProvider, NullEmbeddingProvider
from ..logging import get_logger
from ..matching.matcher import cosine
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)


@dataclass
class SearchQuery:
    """A retrieval request. All filters are optional and combine with AND."""

    text: str = ""
    limit: int = 20
    #: Restrict to these source locators (substring match).
    source_contains: str | None = None
    #: Restrict to these source kinds, e.g. {"pdf"}.
    source_kinds: set[str] = field(default_factory=set)
    trust_tiers: set[TrustTier] = field(default_factory=set)
    #: Restrict to spans on this page (paginated sources only).
    page: int | None = None
    #: Restrict to spans whose heading path contains this text.
    heading_contains: str | None = None
    #: Use embeddings to re-rank when available.
    semantic: bool = False


@dataclass
class SearchHit:
    """One retrieved span, with the full chain back to its origin.

    The chain is not decoration: a hit that cannot be resolved to a document
    and source is not evidence, and Phase 2 exists to make that impossible.
    """

    span: Span
    document: Document | None
    source: Source | None
    score: float
    #: "lexical" | "lexical+semantic"
    signal: str = "lexical"

    @property
    def citation(self) -> str:
        origin = self.source.locator if self.source else "unknown source"
        return f"{origin} :: {self.span.citation()}"

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "span_id": self.span.id,
            "score": round(self.score, 4),
            "signal": self.signal,
            "citation": self.citation,
            "source": self.source.locator if self.source else None,
            "source_kind": self.source.kind.value if self.source else None,
            "trust_tier": self.source.trust_tier.value if self.source else None,
            "document_id": self.span.document_id,
            "page": self.span.page,
            "heading_path": list(self.span.heading_path),
            "lines": f"{self.span.start_line}-{self.span.end_line}",
            "chars": (
                f"{self.span.char_start}-{self.span.char_end}"
                if self.span.char_start is not None
                else None
            ),
            "text": self.span.text if include_text else None,
        }


class SearchService:
    """Lexical retrieval with metadata filtering, plus optional semantic re-rank."""

    def __init__(
        self, store: SqliteStore, *, embeddings: EmbeddingProvider | None = None
    ) -> None:
        self.store = store
        self.embeddings = embeddings or NullEmbeddingProvider()

    @property
    def semantic_available(self) -> bool:
        return self.embeddings.available and self.store.count_embeddings() > 0

    def degradation_note(self) -> str | None:
        """Human-readable note when semantic retrieval is unavailable."""
        if self.semantic_available:
            return None
        if not self.embeddings.available:
            return "no embedding provider available; lexical retrieval only"
        return "no embeddings stored yet; lexical retrieval only"

    # -- search ------------------------------------------------------------

    def search(self, query: SearchQuery) -> list[SearchHit]:
        """Lexical search, filtered, optionally re-ranked semantically."""
        if not query.text.strip():
            return []

        # Over-fetch before filtering, so filters do not silently starve results.
        raw = self.store.search_spans(_fts_query(query.text), limit=max(query.limit * 5, 50))

        hits: list[SearchHit] = []
        for span, rank in raw:
            document = self.store.get_document(span.document_id)
            source = self.store.get_source(document.source_id) if document else None
            hit = SearchHit(span=span, document=document, source=source, score=-rank)
            if self._passes(hit, query):
                hits.append(hit)

        if query.semantic and self.semantic_available:
            hits = self._rerank(query.text, hits)

        hits.sort(key=lambda h: -h.score)
        return hits[: query.limit]

    def _passes(self, hit: SearchHit, query: SearchQuery) -> bool:
        source, span = hit.source, hit.span
        if query.source_contains and (
            source is None or query.source_contains.lower() not in source.locator.lower()
        ):
            return False
        if query.source_kinds and (source is None or source.kind.value not in query.source_kinds):
            return False
        if query.trust_tiers and (source is None or source.trust_tier not in query.trust_tiers):
            return False
        if query.page is not None and span.page != query.page:
            return False
        if query.heading_contains:
            joined = " > ".join(span.heading_path).lower()
            if query.heading_contains.lower() not in joined:
                return False
        return True

    def _rerank(self, text: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Blend lexical rank with embedding similarity.

        Failures degrade to the lexical ordering rather than raising: retrieval
        must not break because an optional component misbehaved.
        """
        try:
            query_vector = self.embeddings.embed([text])[0]
        except Exception as exc:
            log.info("semantic_rerank_unavailable", error=str(exc)[:120])
            return hits

        stored = dict(self.store.get_embeddings("span", self.embeddings.model_id))
        if not stored:
            return hits

        lexical_max = max((h.score for h in hits), default=1.0) or 1.0
        for hit in hits:
            vector = stored.get(hit.span.id)
            if vector is None:
                continue
            similarity = cosine(query_vector, vector)
            # Equal weighting. Not tuned — there is nothing to tune against
            # until there is a labelled retrieval set (Phase 3).
            hit.score = 0.5 * (hit.score / lexical_max) + 0.5 * similarity
            hit.signal = "lexical+semantic"
        return hits

    # -- lookups -----------------------------------------------------------

    def concepts(self, query: str = "", *, limit: int = 50) -> list[Concept]:
        needle = query.strip().lower()
        found = [
            c
            for c in self.store.list_concepts()
            if not needle
            or needle in c.canonical_name.lower()
            or any(needle in a.lower() for a in c.aliases)
        ]
        return found[:limit]

    def documents(self, query: str = "", *, limit: int = 50) -> list[tuple[Document, Source | None]]:
        needle = query.strip().lower()
        out: list[tuple[Document, Source | None]] = []
        for source in self.store.list_sources():
            if needle and needle not in source.locator.lower() and needle not in (
                (source.title or "").lower()
            ):
                continue
            for document in self.store.documents_for_source(source.id):
                out.append((document, source))
        return out[:limit]

    def span(self, span_id: str) -> SearchHit | None:
        """Look up one span with its full provenance chain."""
        span = self.store.get_span(span_id)
        if span is None:
            return None
        document = self.store.get_document(span.document_id)
        source = self.store.get_source(document.source_id) if document else None
        return SearchHit(span=span, document=document, source=source, score=1.0, signal="lookup")

    def spans_for_source(self, locator: str) -> list[Span]:
        source = self.store.get_source_by_locator(locator)
        if source is None:
            return []
        spans: list[Span] = []
        for document in self.store.documents_for_source(source.id):
            spans.extend(self.store.spans_for_document(document.id))
        return spans

    # -- embedding maintenance --------------------------------------------

    def index_embeddings(self, spans: Sequence[Span]) -> int:
        """Embed and store span vectors. No-op when unavailable."""
        if not self.embeddings.available or not spans:
            return 0
        try:
            vectors = self.embeddings.embed([s.text for s in spans])
        except Exception as exc:
            log.warning("embedding_failed", error=str(exc)[:160])
            return 0
        for span, vector in zip(spans, vectors):
            self.store.put_embedding("span", span.id, self.embeddings.model_id, vector)
        return len(vectors)


def _fts_query(text: str) -> str:
    """Turn user text into a safe FTS5 MATCH expression.

    FTS5 treats several characters as operators, so each token is wrapped in
    double quotes. Any double quote *inside* a token must then be doubled —
    otherwise a query like ``quote"inside`` terminates the string early and
    SQLite raises a syntax error the user has no way to diagnose. This is the
    same shape as SQL injection, in a query language most callers never see.
    """
    tokens = [t for t in (w.strip() for w in text.split()) if t]
    if not tokens:
        return '""'
    escaped = [t.replace('"', '""') for t in tokens]
    return " OR ".join(f'"{t}"' for t in escaped)
