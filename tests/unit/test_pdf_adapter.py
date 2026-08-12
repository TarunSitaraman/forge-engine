"""PDF acquisition against real PDF fixtures.

Every fixture is a genuine PDF built from raw PDF syntax
(scripts/make_pdf_fixtures.py), not a mock — a PDF parser tested against mocks
tests nothing.
"""

from __future__ import annotations

import pytest

from forge.domain import IngestionStatus, SourceKind
from forge.sources import PdfAdapter, SourceAdapter


@pytest.fixture
def adapter() -> PdfAdapter:
    return PdfAdapter()


class TestBasics:
    def test_satisfies_protocol(self, adapter):
        assert isinstance(adapter, SourceAdapter)
        assert adapter.kind is SourceKind.PDF

    @pytest.mark.parametrize("name,supported", [("a.pdf", True), ("A.PDF", True), ("a.md", False)])
    def test_supports(self, adapter, tmp_path, name, supported):
        assert adapter.supports(tmp_path / name) is supported

    def test_processor_version_is_reported(self, adapter):
        assert adapter.processor_version.startswith("pdf/")


class TestNormalExtraction:
    def test_single_page(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "simple.pdf")
        assert result.status is IngestionStatus.INGESTED
        assert result.page_count == 1
        assert "Self-attention" in result.text
        assert result.blocks

    def test_title_from_metadata(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "simple.pdf")
        assert result.title == "Attention Mechanisms"

    def test_metadata_extracted(self, adapter, pdf_dir):
        meta = adapter.acquire(pdf_dir / "simple.pdf").metadata
        assert meta["filename"] == "simple.pdf"
        assert meta.get("author") == "Forge Test Suite"

    def test_multipage_preserves_page_boundaries(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        assert result.page_count == 3
        assert sorted({b.page for b in result.blocks}) == [1, 2, 3]

    def test_content_hash_is_stable(self, adapter, pdf_dir):
        a = adapter.acquire(pdf_dir / "multipage.pdf")
        b = adapter.acquire(pdf_dir / "multipage.pdf")
        assert a.content_hash == b.content_hash and a.content_hash

    def test_different_pdfs_hash_differently(self, adapter, pdf_dir):
        a = adapter.acquire(pdf_dir / "simple.pdf")
        b = adapter.acquire(pdf_dir / "multipage.pdf")
        assert a.content_hash != b.content_hash


class TestHeadingDetection:
    def test_headings_detected_by_font_size(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        headings = [b.text for b in result.blocks if b.is_heading]
        assert "Retrieval Augmented Generation" in headings
        assert "Chunking Strategy" in headings

    def test_body_text_is_not_a_heading(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        body = [b for b in result.blocks if not b.is_heading]
        assert body
        assert all("RAG grounds" not in b.text or not b.is_heading for b in result.blocks)

    def test_heading_path_nests_by_size(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        chunking = next(b for b in result.blocks if b.text == "Chunking Strategy")
        assert chunking.heading_path == ("Retrieval Augmented Generation", "Chunking Strategy")

    def test_heuristic_is_declared_not_implied(self, adapter, pdf_dir):
        """PDF has no heading concept; the metadata must say so."""
        result = adapter.acquire(pdf_dir / "simple.pdf")
        assert result.metadata["heading_detection"] == "font-size heuristic"


class TestLocationAccuracy:
    def test_char_offsets_index_into_the_extracted_text(self, adapter, pdf_dir):
        """Offsets that don't resolve are worse than no offsets."""
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        for block in result.blocks:
            assert result.text[block.char_start : block.char_end].strip() == block.text

    def test_line_numbers_index_into_the_extracted_text(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "empty-page.pdf")
        lines = result.text.split("\n")
        for block in result.blocks:
            assert lines[block.start_line - 1].strip() == block.text

    def test_blocks_are_in_reading_order(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "multipage.pdf")
        assert [b.ordinal for b in result.blocks] == list(range(len(result.blocks)))
        assert [b.page for b in result.blocks] == sorted(b.page for b in result.blocks)


class TestEdgeCases:
    def test_empty_page_is_warned_not_failed(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "empty-page.pdf")
        assert result.status is IngestionStatus.INGESTED
        assert any("page 2" in w for w in result.warnings)

    def test_empty_page_produces_no_blocks_for_that_page(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "empty-page.pdf")
        assert 2 not in {b.page for b in result.blocks}
        assert {b.page for b in result.blocks} == {1, 3}

    def test_image_only_pdf_reports_ocr_required(self, adapter, pdf_dir):
        """A valid PDF with no text layer must not read as a successful ingest."""
        result = adapter.acquire(pdf_dir / "image-only.pdf")
        assert result.status is IngestionStatus.OCR_REQUIRED
        assert result.blocks == []
        assert "image-only or scanned" in (result.detail or "")
        assert "OCR is not" in (result.detail or "")

    def test_malformed_pdf_fails_gracefully(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "malformed.pdf")
        assert result.status is IngestionStatus.PARSE_FAILED
        assert result.detail and "could not open" in result.detail

    def test_non_pdf_content_fails_gracefully(self, adapter, pdf_dir):
        result = adapter.acquire(pdf_dir / "not-a-pdf.pdf")
        assert result.status is IngestionStatus.PARSE_FAILED

    def test_missing_file(self, adapter, tmp_path):
        result = adapter.acquire(tmp_path / "nope.pdf")
        assert result.status is IngestionStatus.NOT_FOUND

    def test_failures_are_values_not_exceptions(self, adapter, pdf_dir):
        """A directory ingest must survive one corrupt file."""
        for name in ("malformed.pdf", "not-a-pdf.pdf", "image-only.pdf"):
            result = adapter.acquire(pdf_dir / name)  # must not raise
            assert result.ok is False


class TestDeterminism:
    def test_repeated_acquisition_is_identical(self, adapter, pdf_dir):
        a = adapter.acquire(pdf_dir / "multipage.pdf")
        b = adapter.acquire(pdf_dir / "multipage.pdf")
        assert a.text == b.text
        assert [(x.text, x.page, x.char_start) for x in a.blocks] == [
            (y.text, y.page, y.char_start) for y in b.blocks
        ]
