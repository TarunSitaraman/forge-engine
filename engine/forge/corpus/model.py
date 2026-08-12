"""Data structures produced by the corpus indexer.

These are the indexer's *output* records — deliberately separate from the
canonical domain entities. The indexer describes what is in the vault; the
domain model describes what Forge understands. Conflating them would make the
index a knowledge claim, which it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import ChangeStatus
from ..parsing.frontmatter import Diagnostic, RepairProposal
from ..parsing.links import ResolvedLink


@dataclass
class IndexedFile:
    """Everything deterministically known about one Markdown file."""

    path: str  # vault-relative, POSIX
    content_hash: str
    byte_size: int
    line_count: int
    title: str | None

    frontmatter_present: bool
    frontmatter_valid: bool
    frontmatter_keys: tuple[str, ...] = ()
    canonical: bool | None = None
    doc_type: str | None = None
    status: str | None = None
    tags: tuple[str, ...] = ()

    heading_count: int = 0
    headings: tuple[tuple[int, str, int], ...] = ()
    code_block_count: int = 0
    code_languages: tuple[str, ...] = ()

    wikilink_count: int = 0
    markdown_link_count: int = 0
    #: Link names recovered from the frontmatter `related:` field by text
    #: extraction — recoverable even though that field is unparseable as YAML.
    related: tuple[str, ...] = ()

    diagnostics: list[Diagnostic] = field(default_factory=list)
    repairs: list[RepairProposal] = field(default_factory=list)
    links: list[ResolvedLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "byte_size": self.byte_size,
            "line_count": self.line_count,
            "title": self.title,
            "frontmatter_present": self.frontmatter_present,
            "frontmatter_valid": self.frontmatter_valid,
            "frontmatter_keys": list(self.frontmatter_keys),
            "canonical": self.canonical,
            "doc_type": self.doc_type,
            "status": self.status,
            "tags": list(self.tags),
            "heading_count": self.heading_count,
            "code_block_count": self.code_block_count,
            "code_languages": list(self.code_languages),
            "wikilink_count": self.wikilink_count,
            "markdown_link_count": self.markdown_link_count,
            "related": list(self.related),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "repairs": [r.to_dict() for r in self.repairs],
        }


@dataclass
class CorpusIndex:
    """The complete deterministic index of a vault."""

    vault_path: str
    files: list[IndexedFile] = field(default_factory=list)
    #: content_hash -> paths sharing it (only entries with >1 path).
    duplicate_hashes: dict[str, list[str]] = field(default_factory=dict)
    #: Wall-clock seconds; excluded from equality/determinism comparisons.
    duration_seconds: float = 0.0

    @property
    def file_count(self) -> int:
        return len(self.files)

    def by_path(self) -> dict[str, IndexedFile]:
        return {f.path: f for f in self.files}

    def fingerprint(self) -> str:
        """Stable hash of the index's content.

        Two runs over identical inputs must produce the same fingerprint. This
        is how determinism is asserted in tests, rather than by eyeballing.
        """
        from ..ids import text_hash

        payload = "\n".join(f"{f.path}:{f.content_hash}" for f in sorted(self.files, key=lambda x: x.path))
        return text_hash(payload)


@dataclass(frozen=True)
class SourceChange:
    """One source's change status between a previous index and the current vault."""

    path: str
    status: ChangeStatus
    old_hash: str | None = None
    new_hash: str | None = None


@dataclass
class ChangeSet:
    changes: list[SourceChange] = field(default_factory=list)

    def of(self, status: ChangeStatus) -> list[SourceChange]:
        return [c for c in self.changes if c.status is status]

    @property
    def new(self) -> list[SourceChange]:
        return self.of(ChangeStatus.NEW)

    @property
    def modified(self) -> list[SourceChange]:
        return self.of(ChangeStatus.MODIFIED)

    @property
    def unchanged(self) -> list[SourceChange]:
        return self.of(ChangeStatus.UNCHANGED)

    @property
    def deleted(self) -> list[SourceChange]:
        return self.of(ChangeStatus.DELETED)

    @property
    def requires_processing(self) -> list[SourceChange]:
        """Sources needing work. **Empty means zero LLM calls are warranted.**"""
        return self.new + self.modified

    def summary(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "modified": len(self.modified),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
        }
