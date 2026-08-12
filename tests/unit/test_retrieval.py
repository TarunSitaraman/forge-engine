"""Retrieval service, including the optional embedding path.

Embeddings are exercised through a deterministic fake provider so the semantic
path is genuinely tested without requiring a model.
"""

from __future__ import annotations

import pytest

from forge.domain import (
    Concept,
    ConceptKind,
    Document,
    ProvenanceTier,
    Source,
    SourceKind,
    Span,
    TrustTier,
    deterministic_provenance,
)
from forge.retrieval import SearchQuery, SearchService
from forge.retrieval.search import _fts_query


class FakeEmbeddings:
    """Deterministic stand-in: vectors derived from character counts.

    Enough structure that similar texts score similarly, with no model.
    """

    def __init__(self, *, available: bool = True, explode: bool = False) -> None:
        self._available = available
        self.explode = explode
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return 4

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, texts):
        self.calls += 1
        if self.explode:
            raise RuntimeError("embedding backend fell over")
        out = []
        for text in texts:
            lowered = text.lower()
            out.append(
                [
                    float(lowered.count("retrieval")),
                    float(lowered.count("heap")),
                    float(lowered.count("chunk")),
                    float(len(lowered) % 7),
                ]
            )
        return out


@pytest.fixture
def populated(store):
    """A store with two documents' worth of spans."""
    source = Source.for_path(
        "docs/paper.pdf",
        kind=SourceKind.PDF,
        content_hash="h1",
        trust_tier=TrustTier.UNVERIFIED,
        title="A Paper",
    )
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "h1"),
        source_id=source.id,
        parser="forge.pdf",
        parser_version="1",
        content_hash="h1",
    )
    store.put_document(document)

    spans = [
        Span(
            id=Span.make_id(document.id, i, f"p.{i + 1}"),
            document_id=document.id,
            ordinal=i,
            locator=f"p.{i + 1} L1-L3",
            heading_path=("Paper", heading),
            start_line=1,
            end_line=3,
            text=text,
            content_hash=f"h{i}",
            page=i + 1,
        )
        for i, (heading, text) in enumerate(
            [
                ("Retrieval", "Retrieval augmented generation grounds output in retrieval passages"),
                ("Chunking", "Chunk size affects retrieval quality; chunk boundaries matter"),
                ("Heaps", "A heap keeps the smallest element at its root for fast access"),
            ]
        )
    ]
    store.put_spans(spans)
    return store, source, document, spans


class TestLexicalSearch:
    def test_finds_matching_spans(self, populated):
        store, *_ = populated
        hits = SearchService(store).search(SearchQuery(text="retrieval"))
        assert hits and all("retrieval" in h.span.text.lower() for h in hits)

    def test_empty_query_returns_nothing(self, populated):
        store, *_ = populated
        assert SearchService(store).search(SearchQuery(text="   ")) == []

    def test_limit_is_respected(self, populated):
        store, *_ = populated
        assert len(SearchService(store).search(SearchQuery(text="retrieval", limit=1))) == 1

    def test_hits_carry_full_chain(self, populated):
        store, source, document, _ = populated
        hit = SearchService(store).search(SearchQuery(text="heap"))[0]
        assert hit.source.id == source.id
        assert hit.document.id == document.id
        assert hit.citation.startswith("docs/paper.pdf")
        assert "p.3" in hit.citation

    def test_to_dict_is_serializable(self, populated):
        import json

        store, *_ = populated
        hit = SearchService(store).search(SearchQuery(text="heap"))[0]
        assert json.loads(json.dumps(hit.to_dict()))["page"] == 3

    @pytest.mark.parametrize(
        "query", ["RAG (retrieval)", 'quote"inside', "AND OR NOT", "*", "-", "a:b", "'"]
    )
    def test_operator_characters_are_neutralized(self, populated, query):
        """FTS5 syntax must never leak from user input into a SQL error."""
        store, *_ = populated
        SearchService(store).search(SearchQuery(text=query))  # must not raise

    def test_embedded_quotes_are_doubled(self):
        assert _fts_query('quote"inside') == '"quote""inside"'


class TestFilters:
    def test_page_filter(self, populated):
        store, *_ = populated
        assert SearchService(store).search(SearchQuery(text="retrieval", page=1))
        assert not SearchService(store).search(SearchQuery(text="heap", page=1))

    def test_heading_filter(self, populated):
        store, *_ = populated
        service = SearchService(store)
        assert service.search(SearchQuery(text="chunk", heading_contains="Chunking"))
        assert not service.search(SearchQuery(text="chunk", heading_contains="Nonexistent"))

    def test_source_and_kind_filters(self, populated):
        store, *_ = populated
        service = SearchService(store)
        assert service.search(SearchQuery(text="heap", source_contains="paper"))
        assert not service.search(SearchQuery(text="heap", source_contains="other"))
        assert service.search(SearchQuery(text="heap", source_kinds={"pdf"}))
        assert not service.search(SearchQuery(text="heap", source_kinds={"markdown"}))

    def test_trust_tier_filter(self, populated):
        store, *_ = populated
        service = SearchService(store)
        assert service.search(SearchQuery(text="heap", trust_tiers={TrustTier.UNVERIFIED}))
        assert not service.search(SearchQuery(text="heap", trust_tiers={TrustTier.PEER_REVIEWED}))


class TestSemanticPath:
    def test_unavailable_by_default(self, populated):
        store, *_ = populated
        service = SearchService(store)
        assert service.semantic_available is False
        assert "no embedding provider" in service.degradation_note()

    def test_available_only_once_vectors_exist(self, populated):
        store, _, _, spans = populated
        service = SearchService(store, embeddings=FakeEmbeddings())
        assert service.semantic_available is False
        assert "no embeddings stored" in service.degradation_note()

        assert service.index_embeddings(spans) == 3
        assert service.semantic_available is True
        assert service.degradation_note() is None

    def test_rerank_changes_signal(self, populated):
        store, _, _, spans = populated
        service = SearchService(store, embeddings=FakeEmbeddings())
        service.index_embeddings(spans)
        hits = service.search(SearchQuery(text="retrieval", semantic=True))
        assert hits and any(h.signal == "lexical+semantic" for h in hits)

    def test_lexical_still_works_when_semantic_requested_but_absent(self, populated):
        """The documented degradation mode."""
        store, *_ = populated
        hits = SearchService(store).search(SearchQuery(text="retrieval", semantic=True))
        assert hits and all(h.signal == "lexical" for h in hits)

    def test_embedding_failure_degrades_to_lexical(self, populated):
        """An optional component misbehaving must not break retrieval."""
        store, _, _, spans = populated
        working = SearchService(store, embeddings=FakeEmbeddings())
        working.index_embeddings(spans)

        broken = SearchService(store, embeddings=FakeEmbeddings(explode=True))
        hits = broken.search(SearchQuery(text="retrieval", semantic=True))
        assert hits, "search must still return results"

    def test_index_embeddings_noop_when_unavailable(self, populated):
        store, _, _, spans = populated
        service = SearchService(store, embeddings=FakeEmbeddings(available=False))
        assert service.index_embeddings(spans) == 0

    def test_index_embeddings_survives_provider_error(self, populated):
        store, _, _, spans = populated
        service = SearchService(store, embeddings=FakeEmbeddings(explode=True))
        assert service.index_embeddings(spans) == 0


class TestLookups:
    def test_span_lookup(self, populated):
        store, _, _, spans = populated
        hit = SearchService(store).span(spans[0].id)
        assert hit is not None and hit.signal == "lookup"

    def test_missing_span_lookup(self, populated):
        store, *_ = populated
        assert SearchService(store).span("nope") is None

    def test_spans_for_source(self, populated):
        store, source, _, _ = populated
        assert len(SearchService(store).spans_for_source(source.locator)) == 3
        assert SearchService(store).spans_for_source("nope.md") == []

    def test_document_lookup(self, populated):
        store, *_ = populated
        service = SearchService(store)
        assert len(service.documents()) == 1
        assert len(service.documents("paper")) == 1
        assert service.documents("nonexistent") == []

    def test_concept_lookup(self, store):
        concept = Concept(
            id=Concept.make_id("RAG"),
            canonical_name="RAG",
            kind=ConceptKind.TECHNOLOGY,
            aliases=("Retrieval Augmented Generation",),
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
        )
        store.put_concept(concept)
        service = SearchService(store)
        assert len(service.concepts()) == 1
        assert len(service.concepts("rag")) == 1
        assert len(service.concepts("retrieval augmented")) == 1, "aliases are searched"
        assert service.concepts("nothing") == []
