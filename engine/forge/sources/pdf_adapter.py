"""Deterministic PDF acquisition via pypdfium2.

No LLM. No OCR. Same bytes in, same blocks out.

``pypdfium2`` was chosen in Phase 0 over PyMuPDF specifically to avoid the
AGPL/commercial licensing question (see technology-decisions §6). It is
Apache/BSD-licensed and ships prebuilt wheels, so PDF support needs no system
packages.

**What this adapter promises, and what it does not.** Page numbers are exact —
they come from the document structure. Character offsets are exact within the
text this parser extracted. *Headings are a heuristic*: PDF has no heading
concept, so they are inferred from font size relative to the page's body text.
That is stated as a heuristic in the metadata rather than presented as
structure the format actually carries.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterator

from ..domain import IngestionStatus, SourceKind
from ..ids import text_hash
from ..logging import get_logger
from .base import AcquisitionResult, TextBlock, failed, normalize_text

log = get_logger(__name__)

PROCESSOR_VERSION = "pdf/0.2.0"

#: A line whose font size exceeds the page's body size by this factor is
#: treated as a heading. Tuned to be conservative: missing a heading costs
#: less than inventing document structure that is not there.
HEADING_SIZE_RATIO = 1.15

#: Below this many extractable characters per page, a PDF is treated as
#: image-only. Not zero: real scanned PDFs often carry a few stray characters
#: from watermarks or embedded metadata.
MIN_CHARS_PER_PAGE = 8


class PdfAdapter:
    """Acquires text and location metadata from a local PDF."""

    processor_version = PROCESSOR_VERSION

    @property
    def kind(self) -> SourceKind:
        return SourceKind.PDF

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def acquire(self, path: Path) -> AcquisitionResult:
        locator = path.as_posix()

        if not path.is_file():
            return failed(IngestionStatus.NOT_FOUND, self.kind, locator, f"no such file: {path}")

        try:
            import pypdfium2 as pdfium
        except ImportError:  # pragma: no cover - dependency is declared
            return failed(
                IngestionStatus.UNSUPPORTED,
                self.kind,
                locator,
                "pypdfium2 is not installed; PDF ingestion unavailable",
            )

        raw = path.read_bytes()

        try:
            document = pdfium.PdfDocument(str(path))
            page_count = len(document)
        except Exception as exc:
            # Malformed PDFs are a reported outcome, not a crash. A directory
            # ingest must survive one corrupt file.
            log.warning("pdf_open_failed", path=locator, error=str(exc))
            return failed(
                IngestionStatus.PARSE_FAILED,
                self.kind,
                locator,
                f"could not open PDF: {_clean(exc)}",
            )

        try:
            pages = list(self._read_pages(document, pdfium))
        except Exception as exc:
            log.warning("pdf_read_failed", path=locator, error=str(exc))
            return failed(
                IngestionStatus.PARSE_FAILED,
                self.kind,
                locator,
                f"could not read PDF pages: {_clean(exc)}",
            )
        finally:
            _close(document)

        metadata = self._metadata(path, pdfium)
        total_chars = sum(len(text) for text, _ in pages)

        # Image-only detection. Reported honestly rather than returning a
        # successful ingest with zero content.
        if page_count and total_chars < MIN_CHARS_PER_PAGE * page_count:
            return AcquisitionResult(
                status=IngestionStatus.OCR_REQUIRED,
                kind=self.kind,
                locator=locator,
                content_hash=text_hash(raw.decode("latin-1", errors="replace")),
                title=metadata.get("title"),
                metadata=metadata,
                byte_size=len(raw),
                page_count=page_count,
                detail=(
                    f"only {total_chars} extractable characters across {page_count} page(s); "
                    f"this appears to be an image-only or scanned PDF. OCR is not "
                    f"implemented, so no text was extracted."
                ),
            )

        text, blocks, warnings = self._build_blocks(pages)

        return AcquisitionResult(
            status=IngestionStatus.INGESTED,
            kind=self.kind,
            locator=locator,
            text=text,
            # Hash the *extracted text*, not the raw bytes. Two PDFs with
            # identical content but different producer metadata or timestamps
            # must not read as different sources — and re-saving a PDF changes
            # its bytes constantly.
            content_hash=text_hash(text),
            blocks=blocks,
            title=metadata.get("title") or _title_from_blocks(blocks),
            metadata={**metadata, "heading_detection": "font-size heuristic"},
            byte_size=len(raw),
            line_count=text.count("\n") + 1 if text else 0,
            page_count=page_count,
            warnings=warnings,
        )

    # -- internals ---------------------------------------------------------

    def _read_pages(self, document: Any, pdfium: Any) -> Iterator[tuple[str, list[tuple[str, float]]]]:
        """Yield ``(page_text, [(line_text, font_size), ...])`` per page."""
        raw_api = pdfium.raw
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            try:
                page_text = normalize_text(textpage.get_text_range() or "")
                lines = self._lines_with_sizes(textpage, raw_api, page_text)
            finally:
                _close(textpage)
                _close(page)
            yield page_text, lines

    def _lines_with_sizes(self, textpage: Any, raw_api: Any, page_text: str) -> list[tuple[str, float]]:
        """Pair each visual line with its dominant font size.

        Font size is the only signal a text-layer PDF offers for heading
        detection. Reading it per character and taking the median per line is
        robust to stray glyphs at a different size.
        """
        try:
            count = raw_api.FPDFText_CountChars(textpage)
        except Exception:  # pragma: no cover - defensive
            return [(line, 0.0) for line in page_text.split("\n")]

        sizes: list[float] = []
        chars: list[str] = []
        for i in range(count):
            try:
                code = raw_api.FPDFText_GetUnicode(textpage, i)
                size = raw_api.FPDFText_GetFontSize(textpage, i)
            except Exception:  # pragma: no cover - defensive
                break
            chars.append(chr(code) if code else "")
            sizes.append(float(size))

        lines: list[tuple[str, float]] = []
        current: list[str] = []
        current_sizes: list[float] = []
        for ch, size in zip(chars, sizes):
            if ch in ("\r", "\n"):
                if current:
                    lines.append(("".join(current), _median(current_sizes)))
                    current, current_sizes = [], []
                continue
            current.append(ch)
            if ch.strip():
                current_sizes.append(size)
        if current:
            lines.append(("".join(current), _median(current_sizes)))
        return lines

    def _build_blocks(
        self, pages: list[tuple[str, list[tuple[str, float]]]]
    ) -> tuple[str, list[TextBlock], list[str]]:
        """Assemble blocks with exact offsets into a single document text.

        The concatenated text is the reference frame: every offset, line
        number, and page recorded on a block indexes into it, so a span can
        always be resolved back to the exact characters it came from.
        """
        body_size = _body_font_size(pages)
        parts: list[str] = []
        blocks: list[TextBlock] = []
        warnings: list[str] = []

        char_cursor = 0
        line_cursor = 1
        ordinal = 0
        heading_stack: list[tuple[float, str]] = []

        for page_index, (page_text, lines) in enumerate(pages, start=1):
            if not page_text.strip():
                warnings.append(f"page {page_index} contains no extractable text")
                continue

            for line_text, size in lines:
                stripped = line_text.strip()
                if not stripped:
                    # Skipped lines are not appended to `parts`, so the line
                    # cursor must NOT advance — line numbers index into the
                    # assembled text, and drifting them would make every
                    # downstream citation point at the wrong line.
                    continue

                is_heading = bool(body_size) and size >= body_size * HEADING_SIZE_RATIO
                if is_heading:
                    # Larger heading closes smaller open ones.
                    while heading_stack and heading_stack[-1][0] <= size:
                        heading_stack.pop()
                    heading_stack.append((size, stripped))
                    heading_path = tuple(h for _, h in heading_stack)
                else:
                    heading_path = tuple(h for _, h in heading_stack)

                parts.append(line_text)
                start_char = char_cursor
                char_cursor += len(line_text) + 1  # +1 for the joining newline

                blocks.append(
                    TextBlock(
                        text=stripped,
                        ordinal=ordinal,
                        page=page_index,
                        heading_path=heading_path,
                        is_heading=is_heading,
                        heading_level=len(heading_stack) if is_heading else None,
                        start_line=line_cursor,
                        end_line=line_cursor,
                        char_start=start_char,
                        char_end=char_cursor - 1,
                    )
                )
                ordinal += 1
                line_cursor += 1

        return "\n".join(parts), blocks, warnings

    def _metadata(self, path: Path, pdfium: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {"filename": path.name}
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception:  # pragma: no cover - already validated above
            return meta
        try:
            for key in ("Title", "Author", "Subject", "Creator", "Producer", "CreationDate"):
                try:
                    value = document.get_metadata_value(key)
                except Exception:
                    continue
                if value:
                    meta[key.lower()] = value
        finally:
            _close(document)
        return meta


# -- helpers ---------------------------------------------------------------


def _body_font_size(pages: list[tuple[str, list[tuple[str, float]]]]) -> float:
    """Median font size across the document — the baseline headings exceed."""
    sizes = [size for _, lines in pages for text, size in lines if text.strip() and size > 0]
    return statistics.median(sizes) if sizes else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _title_from_blocks(blocks: list[TextBlock]) -> str | None:
    """Fall back to the first detected heading when metadata carries no title."""
    for block in blocks:
        if block.is_heading:
            return block.text
    return blocks[0].text if blocks else None


def _close(obj: Any) -> None:
    for attr in ("close", "__exit__"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                fn() if attr == "close" else fn(None, None, None)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            return


def _clean(exc: Exception) -> str:
    return " ".join(str(exc).split())[:160]
