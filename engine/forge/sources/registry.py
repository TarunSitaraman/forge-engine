"""Adapter registry — selects the adapter for a path.

Phase 2 registers exactly two adapters. Speculative connectors (web, GitHub,
YouTube, Notion, …) are deliberately absent: the extension point exists, and
building connectors before the evidence foundation is solid would be surface
area instead of capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..domain import IngestionStatus, SourceKind
from .base import AcquisitionResult, SourceAdapter, failed
from .markdown_adapter import MarkdownAdapter
from .pdf_adapter import PdfAdapter


class AdapterRegistry:
    """Maps a path to the adapter that can acquire it."""

    def __init__(self, adapters: Sequence[SourceAdapter] | None = None) -> None:
        self._adapters: list[SourceAdapter] = list(
            adapters if adapters is not None else (MarkdownAdapter(), PdfAdapter())
        )

    def register(self, adapter: SourceAdapter) -> None:
        self._adapters.append(adapter)

    def for_path(self, path: Path) -> SourceAdapter | None:
        for adapter in self._adapters:
            if adapter.supports(path):
                return adapter
        return None

    def acquire(self, path: Path) -> AcquisitionResult:
        """Acquire, or return an UNSUPPORTED result naming what is supported."""
        adapter = self.for_path(path)
        if adapter is None:
            return failed(
                IngestionStatus.UNSUPPORTED,
                SourceKind.MANUAL,
                path.as_posix(),
                f"no adapter for {path.suffix or 'extensionless file'}; "
                f"Phase 2 supports: {', '.join(self.supported_extensions())}",
            )
        return adapter.acquire(path)

    def supported_extensions(self) -> list[str]:
        known = [".md", ".markdown", ".pdf"]
        return [ext for ext in known if self.for_path(Path("x" + ext)) is not None]

    def processor_version(self, path: Path) -> str:
        adapter = self.for_path(path)
        return adapter.processor_version if adapter else "none/0"

    def discover(self, root: Path, *, recursive: bool = True) -> list[Path]:
        """Find ingestible files under a directory, sorted for determinism."""
        if root.is_file():
            return [root] if self.for_path(root) else []
        pattern = "**/*" if recursive else "*"
        found = [
            p
            for p in root.glob(pattern)
            if p.is_file() and not p.is_symlink() and self.for_path(p) is not None
        ]
        return sorted(found)


DEFAULT_REGISTRY = AdapterRegistry
