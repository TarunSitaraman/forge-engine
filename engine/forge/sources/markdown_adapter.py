"""Markdown acquisition.

**Reuses the Phase 1 parser wholesale.** There is exactly one Markdown parser
in Forge (:mod:`forge.parsing.markdown`), and this adapter wraps it rather than
reimplementing heading, wikilink, tag, or frontmatter handling. That parser
already carries the hard-won behaviour — code-fence masking so Python list
literals are not read as wikilinks, frontmatter exclusion that preserves line
numbers, CRLF normalization — and duplicating any of it would guarantee the two
copies drift.

The adapter's own contribution is small and specific: turn the parsed document
into located :class:`~forge.sources.base.TextBlock` values under the same
contract the PDF adapter satisfies, so the ingestion pipeline does not care
which format it is handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain import IngestionStatus, SourceKind
from ..ids import text_hash
from ..parsing.frontmatter import extract_wikilink_values, parse_frontmatter
from ..parsing.markdown import parse_markdown
from .base import AcquisitionResult, TextBlock, failed, normalize_text

PROCESSOR_VERSION = "markdown/0.2.0"


class MarkdownAdapter:
    """Acquires text, structure, and metadata from a local Markdown file."""

    processor_version = PROCESSOR_VERSION

    @property
    def kind(self) -> SourceKind:
        return SourceKind.MARKDOWN

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".md", ".markdown")

    def acquire(self, path: Path) -> AcquisitionResult:
        locator = path.as_posix()

        if not path.is_file():
            return failed(IngestionStatus.NOT_FOUND, self.kind, locator, f"no such file: {path}")

        raw_bytes = path.read_bytes()
        text = normalize_text(raw_bytes.decode("utf-8", errors="replace"))

        parsed = parse_markdown(text)
        frontmatter = parse_frontmatter(parsed.frontmatter_raw)
        related = (
            tuple(extract_wikilink_values(parsed.frontmatter_raw, "related"))
            if parsed.frontmatter_raw
            else ()
        )

        blocks = self._blocks(text, parsed)

        metadata: dict[str, Any] = {
            "filename": path.name,
            "frontmatter_present": frontmatter.present,
            "frontmatter_valid": frontmatter.valid,
            "frontmatter_keys": sorted(frontmatter.data.keys()),
            "frontmatter": {k: v for k, v in frontmatter.data.items() if _scalarish(v)},
            "tags": list(parsed.tags),
            "wikilinks": [w.target for w in parsed.wikilinks],
            "markdown_links": [m.target for m in parsed.markdown_links],
            "related": list(related),
            "headings": [(h.level, h.text, h.line) for h in parsed.headings],
            "code_blocks": parsed.code_block_count,
            "diagnostics": [d.to_dict() for d in frontmatter.diagnostics],
        }

        warnings = [
            f"{d.code.value}: {d.message}"
            for d in frontmatter.diagnostics
            if d.severity.value in ("error", "warning")
        ]

        return AcquisitionResult(
            status=IngestionStatus.INGESTED,
            kind=self.kind,
            locator=locator,
            text=text,
            # Identical hashing to the Phase 1 indexer, so a file ingested here
            # and indexed there agree on whether it changed.
            content_hash=text_hash(text),
            blocks=blocks,
            title=parsed.title,
            metadata=metadata,
            byte_size=len(raw_bytes),
            line_count=parsed.line_count,
            page_count=None,  # Markdown has no pages; recorded as absent, not faked
            warnings=warnings,
        )

    def _blocks(self, text: str, parsed: Any) -> list[TextBlock]:
        """Split into heading-delimited blocks with exact offsets.

        Mirrors the Phase 1 indexer's span boundaries — content before the
        first heading becomes a preamble block, so no part of the document is
        unattributable.
        """
        lines = text.split("\n")
        total = len(lines)
        # Precompute the character offset at which each 1-based line starts.
        line_offsets: list[int] = []
        cursor = 0
        for line in lines:
            line_offsets.append(cursor)
            cursor += len(line) + 1

        headings = [(h.level, h.text, h.line) for h in parsed.headings]
        blocks: list[TextBlock] = []
        ordinal = 0

        def add(start: int, end: int, path: tuple[str, ...], is_heading: bool, level: int | None) -> None:
            nonlocal ordinal
            body = "\n".join(lines[start - 1 : end]).strip()
            if not body:
                return
            char_start = line_offsets[start - 1]
            char_end = line_offsets[end - 1] + len(lines[end - 1])
            blocks.append(
                TextBlock(
                    text=body,
                    ordinal=ordinal,
                    page=None,
                    heading_path=path,
                    is_heading=is_heading,
                    heading_level=level,
                    start_line=start,
                    end_line=end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            ordinal += 1

        if not headings:
            add(1, total, (), False, None)
            return blocks

        first_line = headings[0][2]
        if first_line > 1:
            add(1, first_line - 1, (), False, None)

        stack: list[tuple[int, str]] = []
        for i, (level, title, line) in enumerate(headings):
            end = headings[i + 1][2] - 1 if i + 1 < len(headings) else total
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            add(line, end, tuple(t for _, t in stack), True, level)
        return blocks


def _scalarish(value: Any) -> bool:
    """Keep frontmatter values that survive JSON round-tripping cleanly."""
    return isinstance(value, (str, int, float, bool, list)) or value is None
