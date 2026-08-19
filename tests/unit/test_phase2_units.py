"""Markdown adapter, chunking, derivation keys, embeddings, and extraction.

Grouped into one module because each area is small; splitting them would spread
~15 tests across five files with no navigational benefit.
"""

from __future__ import annotations

import json

import pytest

from forge.domain import ExtractionStatus, IngestionStatus, ProvenanceTier, Span, SourceKind
from forge.embeddings import NullEmbeddingProvider, OllamaEmbeddingProvider
from forge.extraction import CandidateExtractor, extraction_provenance
from forge.extraction.extractor import _grounded
from forge.ingestion import build_spans, extraction_key, split_sentences
from forge.ingestion.chunking import MAX_CHUNK_CHARS
from forge.llm import MockProvider
from forge.sources import AdapterRegistry, MarkdownAdapter, TextBlock


# --------------------------------------------------------------------------
# Markdown adapter
# --------------------------------------------------------------------------


class TestMarkdownAdapter:
    def test_reuses_phase1_parser_behaviour(self, fixture_vault):
        """Code fences must still be masked — the Phase 1 hazard, via the adapter."""
        result = MarkdownAdapter().acquire(fixture_vault / "DSA/01_Patterns/Graph Traversal.md")
        assert result.status is IngestionStatus.INGESTED
        # The fixture contains `grid = [[0, 1], [1, 0]]` inside a code fence.
        assert not any(link.strip("[]").isdigit() for link in result.metadata["wikilinks"])
        assert "DFS" in result.metadata["wikilinks"]

    def test_preserves_all_required_metadata(self, fixture_vault):
        result = MarkdownAdapter().acquire(fixture_vault / "DSA/01_Patterns/Graph Traversal.md")
        meta = result.metadata
        assert result.title == "Graph Traversal"
        assert meta["frontmatter_present"] is True
        assert meta["headings"]
        assert meta["wikilinks"] and meta["related"]
        assert result.content_hash and result.line_count > 0

    def test_recovers_related_from_malformed_frontmatter(self, fixture_vault):
        """The nested-list form parses as YAML but is useless; text extraction works."""
        result = MarkdownAdapter().acquire(fixture_vault / "DSA/01_Patterns/Graph Traversal.md")
        assert result.metadata["related"] == ["DFS", "BFS", "Union Find"]

    def test_hash_matches_phase1_indexer(self, fixture_vault, indexer):
        """A file ingested here and indexed there must be the same source."""
        rel = "DSA/01_Patterns/DFS.md"
        acquired = MarkdownAdapter().acquire(fixture_vault / rel)
        indexed = indexer.build_index().by_path()[rel]
        assert acquired.content_hash == indexed.content_hash

    def test_markdown_has_no_pages(self, fixture_vault):
        result = MarkdownAdapter().acquire(fixture_vault / "Notes/plain-note.md")
        assert result.page_count is None
        assert all(b.page is None for b in result.blocks)

    def test_offsets_index_into_text(self, fixture_vault):
        result = MarkdownAdapter().acquire(fixture_vault / "DSA/01_Patterns/Graph Traversal.md")
        for block in result.blocks:
            assert result.text[block.char_start : block.char_end].strip() == block.text

    def test_missing_file(self, tmp_path):
        assert MarkdownAdapter().acquire(tmp_path / "no.md").status is IngestionStatus.NOT_FOUND


class TestRegistry:
    def test_routes_by_extension(self, tmp_path):
        registry = AdapterRegistry()
        assert registry.for_path(tmp_path / "a.md").kind is SourceKind.MARKDOWN
        assert registry.for_path(tmp_path / "a.pdf").kind is SourceKind.PDF
        assert registry.for_path(tmp_path / "a.txt") is None

    def test_unsupported_names_what_is_supported(self, tmp_path):
        result = AdapterRegistry().acquire(tmp_path / "a.txt")
        assert result.status is IngestionStatus.UNSUPPORTED
        assert ".pdf" in (result.detail or "")

    def test_discover_is_sorted_and_filtered(self, fixture_vault):
        found = AdapterRegistry().discover(fixture_vault)
        assert found == sorted(found)
        assert all(p.suffix in (".md", ".markdown", ".pdf") for p in found)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def block(text: str, ordinal: int, **kw) -> TextBlock:
    defaults = dict(
        page=None,
        heading_path=(),
        is_heading=False,
        start_line=ordinal + 1,
        end_line=ordinal + 1,
        char_start=ordinal * 100,
        char_end=ordinal * 100 + len(text),
    )
    defaults.update(kw)
    return TextBlock(text=text, ordinal=ordinal, **defaults)


class TestChunking:
    def test_identical_input_gives_identical_span_ids(self):
        blocks = [block("Alpha content here", 0), block("Beta content here", 1)]
        a = build_spans("doc1", blocks)
        b = build_spans("doc1", blocks)
        assert [s.id for s in a] == [s.id for s in b]

    def test_different_documents_give_different_span_ids(self):
        blocks = [block("Same text entirely", 0)]
        assert build_spans("doc1", blocks)[0].id != build_spans("doc2", blocks)[0].id

    def test_heading_starts_a_new_span(self):
        blocks = [
            block("Body under first heading, long enough to stand alone here", 0, heading_path=("A",)),
            block("Second Heading", 1, is_heading=True, heading_path=("B",)),
            block("Body under second heading, also long enough to matter", 2, heading_path=("B",)),
        ]
        spans = build_spans("d", blocks)
        assert len(spans) == 2
        assert spans[1].heading_path == ("B",)

    def test_page_change_starts_a_new_span(self):
        blocks = [
            block("Content on the first page of this document", 0, page=1),
            block("Content on the second page of this document", 1, page=2),
        ]
        spans = build_spans("d", blocks)
        assert len(spans) == 2
        assert [s.page for s in spans] == [1, 2]

    def test_lone_heading_is_merged_forward(self):
        """A bare heading span carries a location but no content."""
        blocks = [
            block("Tiny", 0, is_heading=True, heading_path=("Tiny",)),
            block("Substantial body text that follows the heading directly", 1, heading_path=("Tiny",)),
        ]
        spans = build_spans("d", blocks)
        assert len(spans) == 1
        assert "Tiny" in spans[0].text and "Substantial" in spans[0].text

    def test_spans_carry_full_location(self):
        blocks = [block("Some located content here", 0, page=3, heading_path=("H",))]
        span = build_spans("d", blocks)[0]
        assert span.page == 3
        assert span.char_start is not None and span.char_end is not None
        assert span.locator.startswith("p.3")
        assert "H" in span.citation()

    def test_ordinals_are_contiguous(self):
        blocks = [block(f"Block number {i} with enough text to survive", i) for i in range(6)]
        spans = build_spans("d", blocks)
        assert [s.ordinal for s in spans] == list(range(len(spans)))

    def test_oversized_group_is_split(self):
        big = "word " * 800  # ~4000 chars
        blocks = [block(big, i, heading_path=("H",)) for i in range(3)]
        spans = build_spans("d", blocks)
        assert len(spans) > 1

    def test_empty_blocks_produce_no_spans(self):
        assert build_spans("d", []) == []
        assert build_spans("d", [block("   ", 0)]) == []

    def test_sentence_split_never_cuts_mid_sentence(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        pieces = split_sentences(text, limit=30)
        assert all(p.strip().endswith(".") for p in pieces)

    def test_sentence_split_returns_whole_when_unsplittable(self):
        """An oversized span beats a truncated one."""
        text = "x" * (MAX_CHUNK_CHARS + 100)
        assert split_sentences(text) == [text]


# --------------------------------------------------------------------------
# Derivation keys
# --------------------------------------------------------------------------


class TestDerivationKeys:
    def _key(self, **kw):
        base = dict(
            content_hash="h1",
            processor_version="p1",
            model_id="m1",
            prompt_version="pr1",
            schema_version="s1",
        )
        base.update(kw)
        return extraction_key(**base).value()

    def test_same_inputs_same_key(self):
        assert self._key() == self._key()

    @pytest.mark.parametrize(
        "field", ["content_hash", "processor_version", "model_id", "prompt_version", "schema_version"]
    )
    def test_every_component_invalidates(self, field):
        """If any component changes, the derived result must be recomputed."""
        assert self._key() != self._key(**{field: "changed"})

    def test_describe_explains_a_miss(self):
        described = extraction_key("h", "p", "m", "pr", "s").describe()
        assert set(described) == {
            "kind",
            "content_hash",
            "processor_version",
            "model_id",
            "prompt_version",
            "schema_version",
        }


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def make_span(text: str, span_id: str = "sp1") -> Span:
    return Span(
        id=span_id,
        document_id="d1",
        ordinal=0,
        locator="p.1",
        start_line=1,
        end_line=3,
        text=text,
        content_hash="h",
    )


class TestExtraction:
    SPAN_TEXT = (
        "Retrieval Augmented Generation grounds generation in retrieved passages, "
        "which reduces hallucination on open-domain questions considerably."
    )

    def test_no_provider_skips_without_failing(self):
        result = CandidateExtractor(None).extract([make_span(self.SPAN_TEXT)])
        assert result.status is ExtractionStatus.SKIPPED_NO_PROVIDER
        assert result.llm_calls == 0

    def test_unavailable_provider_skips(self):
        class Dead(MockProvider):
            def health(self):
                return False, "nothing listening"

        result = CandidateExtractor(Dead()).extract([make_span(self.SPAN_TEXT)])
        assert result.status is ExtractionStatus.SKIPPED_NO_PROVIDER
        assert result.failures[0]["kind"] == "provider_unavailable"

    def test_malformed_output_fails_rather_than_fabricates(self):
        result = CandidateExtractor(MockProvider(default_response="I am prose")).extract(
            [make_span(self.SPAN_TEXT)]
        )
        assert result.status is ExtractionStatus.FAILED
        assert result.concepts == [] and result.claims == []
        assert all(f["kind"] == "schema_violation" for f in result.failures)

    def test_empty_response_is_a_schema_violation(self):
        """`{}` must not validate — it would mean 'extracted nothing' scores as success."""
        result = CandidateExtractor(MockProvider(default_response="{}")).extract(
            [make_span(self.SPAN_TEXT)]
        )
        assert result.status is ExtractionStatus.FAILED

    def test_ungrounded_quote_is_dropped_and_reported(self, scripted_extractor):
        extractor = scripted_extractor(
            claims=[
                {
                    "statement": "An invented claim about something",
                    "evidence_quote": "text that appears nowhere in the span at all",
                    "concept": "",
                }
            ]
        )
        result = extractor.extract([make_span(self.SPAN_TEXT)])
        assert result.claims == []
        assert any(f["kind"] == "ungrounded_quote" for f in result.failures)

    def test_grounded_quote_is_kept(self, scripted_extractor):
        extractor = scripted_extractor(
            claims=[
                {
                    "statement": "RAG reduces hallucination",
                    "evidence_quote": "reduces hallucination on open-domain questions",
                    "concept": "RAG",
                }
            ]
        )
        result = extractor.extract([make_span(self.SPAN_TEXT)])
        assert len(result.claims) == 1
        assert result.claims[0].span_id == "sp1"

    @pytest.mark.parametrize(
        "quote,expected",
        [
            ("grounds generation in retrieved passages", True),
            ("Grounds  Generation   in retrieved passages", True),  # whitespace/case
            ("completely fabricated statement never written", False),
            ("", False),
            ("   ", False),
        ],
    )
    def test_grounding_check(self, quote, expected):
        assert _grounded(quote, self.SPAN_TEXT) is expected

    @pytest.mark.parametrize(
        "quote",
        [
            "Retrieval-Augmented Generation grounds generation",  # hyphenation
            "\u201cgrounds generation in retrieved passages\u201d",  # curly quotes
            "grounds generation ... on open-domain questions",  # elided with ellipsis
            "grounds generation in passages, which reduces hallucination",  # dropped word
        ],
    )
    def test_formatting_noise_is_still_grounded(self, quote):
        """The tolerance that motivated a fuzzy check must survive tightening it."""
        assert _grounded(quote, self.SPAN_TEXT) is True

    @pytest.mark.parametrize(
        "quote",
        [
            # Every word below appears in SPAN_TEXT. Only the order is wrong,
            # which is exactly what a bag-of-words check cannot see.
            "retrieved passages reduces generation grounds hallucination",
            "open-domain generation grounds retrieved questions considerably",
            # Meaning inverted using only the span's own vocabulary.
            "Retrieval Augmented Generation reduces retrieved passages considerably",
        ],
    )
    def test_quote_reassembled_from_span_vocabulary_is_not_grounded(self, quote):
        """Order is the load-bearing part of the grounding check.

        Until 2026-08-19 `_grounded` scored bag-of-words overlap, so any quote
        built from the span's own words scored 1.0 and was stored as evidence —
        including one that inverted the span's meaning. That defeats the rule
        the function exists to enforce, and no test caught it because the only
        negative case used vocabulary foreign to the span, which is the one
        kind of fabrication a word-set check does detect.
        """
        assert _grounded(quote, self.SPAN_TEXT) is False

    def test_ungrounded_reordering_is_dropped_end_to_end(self, scripted_extractor):
        """The unit rule has to actually reject the claim in the pipeline."""
        extractor = scripted_extractor(
            claims=[
                {
                    "statement": "Retrieval reduces retrieved passages",
                    "evidence_quote": "retrieved passages reduces generation grounds hallucination",
                    "concept": "RAG",
                }
            ]
        )
        result = extractor.extract([make_span(self.SPAN_TEXT)])
        assert result.claims == []
        assert any(f["kind"] == "ungrounded_quote" for f in result.failures)

    def test_short_spans_are_still_extracted(self):
        """A 55-char definition is meaningful; an earlier 120-char floor lost it."""
        extractor = CandidateExtractor(MockProvider(default_response="{}"))
        selected = extractor._select([make_span("Heap\nA heap keeps the smallest element at its root.")])
        assert len(selected) == 1

    def test_bare_headings_are_not_sent_to_the_model(self):
        extractor = CandidateExtractor(MockProvider(default_response="{}"))
        assert extractor._select([make_span("Heap")]) == []

    def test_max_spans_caps_cost(self):
        extractor = CandidateExtractor(MockProvider(default_response="{}"), max_spans=2)
        spans = [make_span(self.SPAN_TEXT, f"sp{i}") for i in range(10)]
        assert len(extractor._select(spans)) == 2

    def test_selection_is_deterministic(self):
        extractor = CandidateExtractor(MockProvider(default_response="{}"), max_spans=3)
        spans = [make_span(self.SPAN_TEXT + str(i), f"sp{i}") for i in range(8)]
        assert [s.id for s in extractor._select(spans)] == [s.id for s in extractor._select(spans)]


class TestExtractionProvenance:
    @pytest.mark.parametrize("tier", [ProvenanceTier.SOURCE_FACT, ProvenanceTier.USER_ASSERTION])
    def test_model_cannot_claim_forbidden_tiers(self, tier):
        with pytest.raises(ValueError, match="cannot produce"):
            extraction_provenance("m", make_span("text"), tier=tier)

    def test_records_model_and_span(self):
        provenance = extraction_provenance("llama3.1:8b", make_span("text"))
        assert provenance.model_id == "llama3.1:8b"
        assert provenance.derivation.value == "model"
        assert provenance.inputs[0].entity_id == "sp1"
        assert provenance.prompt_version


# --------------------------------------------------------------------------
# Embeddings (optional)
# --------------------------------------------------------------------------


class TestEmbeddings:
    def test_null_provider_is_unavailable_and_says_so(self):
        provider = NullEmbeddingProvider()
        assert provider.available is False
        assert provider.dimensions == 0
        with pytest.raises(RuntimeError, match="lexical retrieval"):
            provider.embed(["x"])

    def test_ollama_embeddings_unavailable_without_a_server(self):
        provider = OllamaEmbeddingProvider("http://127.0.0.1:1", timeout=1.0)
        assert provider.available is False  # must never raise

    def test_availability_is_cached(self):
        provider = OllamaEmbeddingProvider("http://127.0.0.1:1", timeout=1.0)
        assert provider.available is provider.available

    def test_model_id_is_reported(self):
        assert OllamaEmbeddingProvider(model="bge-small").model_id == "bge-small"
