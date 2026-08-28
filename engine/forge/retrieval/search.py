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

import re

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..domain import Concept, Document, ProvenanceTier, Source, Span, TrustTier
from ..embeddings.base import EmbeddingProvider, NullEmbeddingProvider
from ..logging import get_logger
from ..matching.matcher import cosine
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)


_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SearchQuery:
    """A retrieval request. All filters are optional and combine with AND."""

    text: str = ""
    limit: int = 20
    #: Restrict to these source locators (substring match).
    source_contains: str | None = None
    #: Drop spans whose source locator starts with any of these path prefixes.
    #:
    #: Needed because the vault and the engine's own documentation share one
    #: index: `docs/` describes how Forge works, and answering "what is RAG?"
    #: from `docs/cli.md` retrieves the manual instead of the knowledge.
    #: Measured before this existed, `docs/` took five of the top eight spans.
    #:
    #: **Prefix, not substring** — that distinction cost an hour. A substring
    #: test for `"docs/"` also matches `Technologies/Docs/rag.md`, so excluding
    #: the engine manual silently deleted the entire canonical technology
    #: reference folder, and the best possible answer to "what is retrieval
    #: augmented generation?" stopped appearing anywhere in the top 40.
    exclude_sources: tuple[str, ...] = ()
    #: Restrict to these source kinds, e.g. {"pdf"}.
    source_kinds: set[str] = field(default_factory=set)
    trust_tiers: set[TrustTier] = field(default_factory=set)
    #: Restrict to spans on this page (paginated sources only).
    page: int | None = None
    #: Restrict to spans whose heading path contains this text.
    heading_contains: str | None = None
    #: Use embeddings to re-rank when available.
    semantic: bool = False
    #: Multiply a hit's score when the query's terms appear in the span's
    #: heading path or its source's filename. 1.0 disables it.
    #:
    #: BM25 scores a span by its own text, so a page *named* for the topic
    #: loses to a page that merely mentions it often. Asking "what is retrieval
    #: augmented generation?" ranked four Prompt-Library spans above
    #: `Technologies/Docs/rag.md`, the canonical reference, which did not make
    #: the top eight at all. In a vault whose organising rule is one canonical
    #: home per concept, the filename is a strong relevance signal.
    title_boost: float = 1.0
    #: Share of the fused score given to embedding similarity when
    #: ``semantic`` is on. The remainder goes to the normalized lexical score.
    semantic_weight: float = 0.5


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

        if query.title_boost != 1.0:
            hits = [self._boosted(h, query) for h in hits]

        if query.semantic and self.semantic_available:
            hits = self._fuse(query, hits)

        hits.sort(key=lambda h: -h.score)
        return hits[: query.limit]

    def _boosted(self, hit: SearchHit, query: SearchQuery) -> SearchHit:
        """Raise a hit whose heading path or filename matches the query.

        Boost is applied once, not per matching term: a filename that matches
        the topic is one signal, and compounding it would let a long query
        overwhelm the text score entirely.
        """
        terms = {t for t in _WORD_RE.findall(query.text.lower()) if len(t) > 2}
        if not terms:
            return hit

        haystacks = [" ".join(hit.span.heading_path).lower()]
        if hit.source is not None:
            stem = hit.source.locator.rsplit("/", 1)[-1]
            haystacks.append(stem.removesuffix(".md").replace("-", " ").lower())

        matched = any(t in hay for hay in haystacks for t in terms)
        if not matched:
            return hit
        return replace(hit, score=hit.score * query.title_boost)

    def _passes(self, hit: SearchHit, query: SearchQuery) -> bool:
        source, span = hit.source, hit.span
        if query.source_contains and (
            source is None or query.source_contains.lower() not in source.locator.lower()
        ):
            return False
        if query.exclude_sources and source is not None:
            locator = source.locator.lower().lstrip("./")
            if any(locator.startswith(bad.lower()) for bad in query.exclude_sources):
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

    def _fuse(self, query: SearchQuery, hits: list[SearchHit]) -> list[SearchHit]:
        """Blend lexical results with a *first-class* semantic search.

        Semantic retrieval used to be a re-rank of the lexical candidates,
        which quietly capped it: a span BM25 never retrieved could not be
        recovered no matter how well it matched in embedding space. That is
        precisely the failing case — the labelled set's `fuzzy_concept`
        queries score R@10 0.100 because a question like "stopping a language
        model from making things up by giving it real passages" shares almost
        no vocabulary with `rag.md`, so it was never a candidate at all.

        It also meant the evaluation and the product measured different
        systems: `RetrievalEvaluator` scores every stored vector, while this
        path scored only what lexical found. Searching the vectors directly and
        then fusing makes the two agree.

        Failures degrade to the lexical ordering rather than raising: retrieval
        must not break because an optional component misbehaved.
        """
        text = query.text
        try:
            # The query side must declare itself: a model trained with
            # asymmetric prefixes puts documents and questions in different
            # regions, and embedding a question as a document quietly loses
            # most of the benefit.
            query_vector = self.embeddings.embed([text], task="query")[0]
        except TypeError as exc:
            # A provider whose signature does not match is a programming error,
            # not an absent optional component. Logging it at info alongside
            # genuine unavailability made a broken integration look exactly
            # like "semantic is switched off".
            log.warning("semantic_rerank_provider_incompatible", error=str(exc)[:160])
            return hits
        except Exception as exc:
            log.info("semantic_rerank_unavailable", error=str(exc)[:120])
            return hits

        stored = dict(self.store.get_embeddings("span", self.embeddings.model_id))
        if not stored:
            return hits

        similarities = {
            span_id: cosine(query_vector, vector) for span_id, vector in stored.items()
        }

        # Pull in the strongest semantic matches that lexical never found, so
        # the candidate set is the union of both retrievers rather than
        # whatever BM25 happened to return.
        known = {h.span.id for h in hits}
        extra = sorted(
            (sid for sid in similarities if sid not in known),
            key=lambda sid: -similarities[sid],
        )[: max(query.limit, 10)]
        for span_id in extra:
            span = self.store.get_span(span_id)
            if span is None:
                continue
            document = self.store.get_document(span.document_id)
            source = self.store.get_source(document.source_id) if document else None
            candidate = SearchHit(span=span, document=document, source=source, score=0.0)
            if self._passes(candidate, query):
                hits.append(candidate)

        # Normalize lexical to [0,1] so the two signals are commensurable.
        # Scoring only the embedded spans and leaving the rest on a raw BM25
        # scale would let any span without a vector outrank every span with one.
        lexical_max = max((h.score for h in hits), default=1.0) or 1.0
        for hit in hits:
            lexical = hit.score / lexical_max
            similarity = similarities.get(hit.span.id)
            if similarity is None:
                hit.score = query.semantic_weight * 0.0 + (1 - query.semantic_weight) * lexical
                continue
            hit.score = (
                query.semantic_weight * similarity
                + (1 - query.semantic_weight) * lexical
            )
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
