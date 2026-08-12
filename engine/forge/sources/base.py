"""Source adapter abstraction.

An adapter has exactly one job: **acquire source material and turn it into
text with locations.** It does not extract concepts, discover relationships,
synthesize, reason over a graph, or orchestrate an LLM. Those live in
:mod:`forge.extraction` and above.

Keeping acquisition this narrow is what makes new source types cheap: a future
adapter for web pages or repositories implements :class:`SourceAdapter` and
nothing downstream changes.

Adapters are **deterministic**. Same bytes in, same blocks out, every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from ..domain import IngestionStatus, SourceKind


@dataclass(frozen=True)
class TextBlock:
    """A located run of text as the adapter found it.

    This is the adapter's output unit and is deliberately *not* a
    :class:`~forge.domain.Span`. Blocks describe the document's own structure
    (a paragraph on page 3, a section under a heading); spans are what the
    chunker produces from them. Separating the two keeps chunking policy out of
    the parsers.
    """

    text: str
    #: Ordinal within the document, in reading order.
    ordinal: int
    #: 1-based page for paginated formats; ``None`` when the format has no pages.
    page: int | None = None
    #: Heading path in effect at this block, outermost first.
    heading_path: tuple[str, ...] = ()
    #: True when this block is itself a heading.
    is_heading: bool = False
    heading_level: int | None = None
    #: 1-based line range within the document's extracted text.
    start_line: int = 1
    end_line: int = 1
    #: Character offsets into the document's extracted text.
    char_start: int = 0
    char_end: int = 0


@dataclass
class AcquisitionResult:
    """Everything an adapter produces for one source."""

    status: IngestionStatus
    kind: SourceKind
    locator: str
    #: Full extracted text, normalized to LF. The offsets in ``blocks`` index
    #: into this string, so it is the single reference frame for provenance.
    text: str = ""
    content_hash: str = ""
    blocks: list[TextBlock] = field(default_factory=list)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    byte_size: int = 0
    line_count: int = 0
    page_count: int | None = None
    #: Human-readable reason when status is not INGESTED.
    detail: str | None = None
    #: Non-fatal problems worth surfacing (e.g. pages that yielded no text).
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is IngestionStatus.INGESTED

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "kind": self.kind.value,
            "locator": self.locator,
            "blocks": len(self.blocks),
            "pages": self.page_count,
            "chars": len(self.text),
            "detail": self.detail,
            "warnings": self.warnings,
        }


@runtime_checkable
class SourceAdapter(Protocol):
    """Acquires source material. Nothing more."""

    #: Bumped whenever extraction behaviour changes. Participates in the
    #: derivation key, so a parser improvement invalidates cached results.
    processor_version: str

    @property
    def kind(self) -> SourceKind: ...

    def supports(self, path: Path) -> bool: ...

    def acquire(self, path: Path) -> AcquisitionResult: ...


def failed(
    status: IngestionStatus,
    kind: SourceKind,
    locator: str,
    detail: str,
) -> AcquisitionResult:
    """Build a non-successful result.

    Failures are returned as values rather than raised, so ingesting a
    directory of mixed files reports per-file outcomes instead of aborting on
    the first bad one.
    """
    return AcquisitionResult(status=status, kind=kind, locator=locator, detail=detail)


def normalize_text(text: str) -> str:
    """LF-normalize. Matches Phase 1 hashing so line endings never read as edits."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def blocks_to_lines(blocks: Sequence[TextBlock]) -> int:
    return max((b.end_line for b in blocks), default=0)
