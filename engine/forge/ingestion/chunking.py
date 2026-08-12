"""Deterministic chunking: adapter blocks -> canonical Spans.

Chunking policy lives here, in one place, so both source formats produce spans
under the same rules and neither adapter has to know about them.

Three rules, in priority order:

1. **Never cross a structural boundary.** Headings and pages end a chunk. A
   span that spans two sections cannot be cited precisely, which defeats the
   point of having spans.
2. **Never split a sentence** when a sentence boundary is available. An
   oversized paragraph is split at sentence ends, not at a character count.
3. **Deterministic identity.** Content-identical documents produce
   byte-identical spans with identical ids.

This is explicitly *not* tuned for a vector database. No overlap windows, no
token targets — those are retrieval-quality decisions that belong to a later
phase, once there is something to measure them against.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..domain import Span
from ..ids import text_hash
from ..sources.base import TextBlock

CHUNK_STRATEGY = "structural/0.2.0"

#: Soft ceiling in characters. A chunk over this is split at sentence
#: boundaries; a chunk under it is left whole even if small, because
#: structural coherence beats uniform size.
MAX_CHUNK_CHARS = 2400

#: Blocks shorter than this are merged forward into the next block within the
#: same section, so a lone heading line does not become its own span.
MIN_CHUNK_CHARS = 80

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def build_spans(document_id: str, blocks: Sequence[TextBlock]) -> list[Span]:
    """Group blocks into spans, preserving structure and exact location."""
    groups = _group(blocks)
    spans: list[Span] = []
    ordinal = 0

    for group in groups:
        for piece in _split_oversized(group):
            span = _make_span(document_id, ordinal, piece)
            if span is not None:
                spans.append(span)
                ordinal += 1
    return spans


# -- grouping --------------------------------------------------------------


def _group(blocks: Sequence[TextBlock]) -> list[list[TextBlock]]:
    """Group consecutive blocks that share a section and page.

    A new group starts at a heading, at a page change, or when the current
    group is already large enough.
    """
    groups: list[list[TextBlock]] = []
    current: list[TextBlock] = []
    current_size = 0

    for block in blocks:
        starts_new = (
            not current
            or block.is_heading
            or block.heading_path != current[-1].heading_path
            or block.page != current[-1].page
            or current_size >= MAX_CHUNK_CHARS
        )
        # A heading immediately following a heading joins it, so a section
        # title and its subtitle stay together rather than becoming two spans.
        if starts_new and current and block.is_heading and current[-1].is_heading:
            starts_new = False

        if starts_new and current:
            groups.append(current)
            current, current_size = [], 0

        current.append(block)
        current_size += len(block.text) + 1

    if current:
        groups.append(current)

    return _merge_tiny(groups)


def _merge_tiny(groups: list[list[TextBlock]]) -> list[list[TextBlock]]:
    """Fold undersized groups into the next one when they share a page.

    Prevents a bare heading from becoming a span of its own — such a span
    carries a location but almost no content, which is noise in retrieval.
    """
    if len(groups) < 2:
        return groups

    out: list[list[TextBlock]] = []
    pending: list[TextBlock] = []

    for group in groups:
        size = sum(len(b.text) + 1 for b in group)
        merged = pending + group
        pending = []
        if size < MIN_CHUNK_CHARS and all(b.is_heading for b in group):
            pending = merged
            continue
        out.append(merged)

    if pending:
        if out:
            out[-1].extend(pending)
        else:
            out.append(pending)
    return out


# -- splitting -------------------------------------------------------------


def _split_oversized(group: list[TextBlock]) -> list[list[TextBlock]]:
    """Split a group that exceeds the soft ceiling, at block boundaries."""
    total = sum(len(b.text) + 1 for b in group)
    if total <= MAX_CHUNK_CHARS or len(group) == 1:
        return [group]

    pieces: list[list[TextBlock]] = []
    current: list[TextBlock] = []
    size = 0
    for block in group:
        block_size = len(block.text) + 1
        if current and size + block_size > MAX_CHUNK_CHARS:
            pieces.append(current)
            current, size = [], 0
        current.append(block)
        size += block_size
    if current:
        pieces.append(current)
    return pieces


def split_sentences(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split overlong text at sentence boundaries.

    Exposed for the case where a single block exceeds the ceiling on its own —
    a PDF paragraph with no internal structure. Falls back to returning the
    text whole rather than cutting mid-sentence: an oversized span is a
    retrieval inefficiency, a truncated one is a provenance error.
    """
    if len(text) <= limit:
        return [text]

    sentences = _SENTENCE_END.split(text)
    if len(sentences) == 1:
        return [text]

    out: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) > limit:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        out.append(" ".join(current))
    return out


# -- span construction -----------------------------------------------------


def _make_span(document_id: str, ordinal: int, blocks: list[TextBlock]) -> Span | None:
    text = "\n".join(b.text for b in blocks).strip()
    if not text:
        return None

    first, last = blocks[0], blocks[-1]
    pages = sorted({b.page for b in blocks if b.page is not None})
    page = pages[0] if pages else None
    page_span = (pages[0], pages[-1]) if len(pages) > 1 else None

    locator = _locator(first, last, pages)

    return Span(
        id=Span.make_id(document_id, ordinal, locator),
        document_id=document_id,
        ordinal=ordinal,
        locator=locator,
        heading_path=first.heading_path,
        start_line=first.start_line,
        end_line=max(last.end_line, first.start_line),
        text=text,
        content_hash=text_hash(text),
        chunk_strategy=CHUNK_STRATEGY,
        page=page,
        page_span=page_span,
        char_start=first.char_start,
        char_end=last.char_end,
    )


def _locator(first: TextBlock, last: TextBlock, pages: list[int]) -> str:
    """Stable, human-readable location string.

    Part of the span id, so its format is deliberately stable: changing it
    would change every span identity in the store.
    """
    line_part = (
        f"L{first.start_line}"
        if first.start_line == last.end_line
        else f"L{first.start_line}-L{last.end_line}"
    )
    if not pages:
        return line_part
    if len(pages) > 1:
        return f"pp.{pages[0]}-{pages[-1]} {line_part}"
    return f"p.{pages[0]} {line_part}"
