"""Deterministic corpus indexer.

Walks the vault, hashes every file, parses structure and metadata, resolves
links, and detects duplicates. **No LLM is involved at any point.** The same
input corpus always produces the same index — asserted by
:meth:`CorpusIndex.fingerprint` in the test suite, not merely intended.

The indexer is strictly read-only. It opens files for reading and never writes
to the vault, which is what makes exit criterion 1 (the 621-file corpus remains
unchanged) hold by construction rather than by discipline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..config import Settings
from ..domain import (
    ChangeStatus,
    Document,
    Source,
    SourceKind,
    Span,
    TrustTier,
)
from ..ids import text_hash
from ..logging import get_logger
from ..parsing.frontmatter import extract_wikilink_values, parse_frontmatter
from ..parsing.links import LinkIndex, resolve_markdown_link, resolve_wikilink
from ..parsing.markdown import ParsedMarkdown, parse_markdown
from .model import ChangeSet, CorpusIndex, IndexedFile, SourceChange

log = get_logger(__name__)

PARSER_NAME = "forge.markdown"
PARSER_VERSION = "0.1.0"


class CorpusIndexer:
    """Builds a :class:`CorpusIndex` from a vault directory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vault = settings.vault_path

    # -- discovery ---------------------------------------------------------

    def discover(self) -> list[str]:
        """Vault-relative POSIX paths of all Markdown files, sorted.

        Sorting matters: it is what makes the walk order deterministic across
        filesystems.
        """
        excludes = set(self.settings.exclude_dirs)
        found: list[str] = []
        for path in self._walk(self.vault, excludes):
            found.append(path.relative_to(self.vault).as_posix())
        return sorted(found)

    def _walk(self, root: Path, excludes: set[str]) -> Iterator[Path]:
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name)
        except (PermissionError, OSError):  # pragma: no cover - unreadable dir
            return
        for entry in entries:
            if entry.is_symlink():
                continue  # never follow symlinks out of the vault
            if entry.is_dir():
                if entry.name in excludes or entry.name.startswith("."):
                    continue
                yield from self._walk(entry, excludes)
            elif entry.is_file() and entry.suffix.lower() == ".md":
                yield entry

    # -- indexing ----------------------------------------------------------

    def build_index(self, paths: Sequence[str] | None = None) -> CorpusIndex:
        started = time.perf_counter()
        rel_paths = list(paths) if paths is not None else self.discover()
        link_index = LinkIndex.build(rel_paths)

        files: list[IndexedFile] = []
        for rel in rel_paths:
            try:
                files.append(self._index_file(rel, link_index))
            except Exception as exc:  # one bad file must not abort the run
                log.error("index_file_failed", path=rel, error=str(exc))

        index = CorpusIndex(
            vault_path=str(self.vault),
            files=files,
            duplicate_hashes=_duplicate_hashes(files),
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        log.info(
            "index_built",
            files=index.file_count,
            duplicates=len(index.duplicate_hashes),
            seconds=index.duration_seconds,
        )
        return index

    def _index_file(self, rel_path: str, link_index: LinkIndex) -> IndexedFile:
        abs_path = self.vault / rel_path
        raw_bytes = abs_path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")

        parsed = parse_markdown(text)
        fm = parse_frontmatter(parsed.frontmatter_raw)

        related = (
            tuple(extract_wikilink_values(parsed.frontmatter_raw, "related"))
            if parsed.frontmatter_raw
            else ()
        )

        links = [resolve_wikilink(w, rel_path, link_index) for w in parsed.wikilinks]
        for md in parsed.markdown_links:
            if (resolved := resolve_markdown_link(md, rel_path, link_index)) is not None:
                links.append(resolved)

        return IndexedFile(
            path=rel_path,
            # Hash the normalized text, not raw bytes: the corpus was authored
            # on Windows, and a CRLF/LF difference is not a content change.
            content_hash=text_hash(text),
            byte_size=len(raw_bytes),
            line_count=parsed.line_count,
            title=parsed.title,
            frontmatter_present=fm.present,
            frontmatter_valid=fm.valid,
            frontmatter_keys=tuple(sorted(fm.data.keys())),
            canonical=_as_bool(fm.data.get("canonical")),
            doc_type=_as_str(fm.data.get("type")),
            status=_as_str(fm.data.get("status")),
            tags=_extract_tags(fm.data, parsed),
            heading_count=len(parsed.headings),
            headings=tuple((h.level, h.text, h.line) for h in parsed.headings),
            code_block_count=parsed.code_block_count,
            code_languages=tuple(parsed.code_languages),
            wikilink_count=len(parsed.wikilinks),
            markdown_link_count=len(parsed.markdown_links),
            related=related,
            diagnostics=fm.diagnostics,
            repairs=fm.repairs,
            links=links,
        )

    # -- domain projection -------------------------------------------------

    def to_sources(self, index: CorpusIndex) -> list[Source]:
        """Project indexed files into canonical ``Source`` entities.

        Every file in the existing corpus is imported as
        ``TrustTier.USER_AUTHORED``. Downstream, claims drawn from it are
        ``USER_ASSERTION`` — never ``SOURCE_FACT``. The corpus is hand-written
        and largely uncited: it is authoritative for what the user believes,
        and it is not evidence for what is true. Collapsing that distinction
        would poison the tiering of everything built on top.
        """
        return [
            Source.for_path(
                f.path,
                kind=SourceKind.MARKDOWN,
                content_hash=f.content_hash,
                trust_tier=TrustTier.USER_AUTHORED,
                title=f.title,
                byte_size=f.byte_size,
                line_count=f.line_count,
            )
            for f in index.files
        ]

    def to_document_and_spans(self, indexed: IndexedFile, source: Source) -> tuple[Document, list[Span]]:
        """Build a Document and its heading-delimited Spans.

        Chunking is deterministic and structure-aware: spans break on heading
        boundaries, which the corpus uses very consistently.
        """
        document = Document(
            id=Document.make_id(source.id, indexed.content_hash),
            source_id=source.id,
            parser=PARSER_NAME,
            parser_version=PARSER_VERSION,
            content_hash=indexed.content_hash,
            headings=indexed.headings,
            frontmatter_present=indexed.frontmatter_present,
            frontmatter_valid=indexed.frontmatter_valid,
        )
        text = (self.vault / indexed.path).read_text(encoding="utf-8", errors="replace")
        spans = build_spans(document.id, text, indexed.headings)
        return document, spans


def build_spans(
    document_id: str, text: str, headings: Sequence[tuple[int, str, int]]
) -> list[Span]:
    """Split text into heading-delimited spans.

    Content before the first heading becomes a preamble span, so no bytes of
    the document are unattributable.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    total = len(lines)

    boundaries = [(lvl, title, ln) for lvl, title, ln in headings]
    spans: list[Span] = []
    ordinal = 0

    def add(start: int, end: int, path: tuple[str, ...], locator: str) -> None:
        nonlocal ordinal
        body = "\n".join(lines[start - 1 : end]).strip()
        if not body:
            return
        spans.append(
            Span(
                id=Span.make_id(document_id, ordinal, locator),
                document_id=document_id,
                ordinal=ordinal,
                locator=locator,
                heading_path=path,
                start_line=start,
                end_line=end,
                text=body,
                content_hash=text_hash(body),
            )
        )
        ordinal += 1

    if not boundaries:
        add(1, total, (), f"L1-L{total}")
        return spans

    first_line = boundaries[0][2]
    if first_line > 1:
        add(1, first_line - 1, (), f"L1-L{first_line - 1}")

    stack: list[tuple[int, str]] = []
    for i, (level, title, line) in enumerate(boundaries):
        end = boundaries[i + 1][2] - 1 if i + 1 < len(boundaries) else total
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        add(line, end, tuple(t for _, t in stack), f"L{line}-L{end}")
    return spans


def detect_changes(
    current: Iterable[IndexedFile], previous: dict[str, str]
) -> ChangeSet:
    """Compare the current index against ``{path: content_hash}`` from storage.

    This is the foundation of incremental processing: a source whose hash is
    unchanged is not reprocessed, and therefore costs zero LLM calls.
    """
    changes: list[SourceChange] = []
    seen: set[str] = set()

    for f in current:
        seen.add(f.path)
        old = previous.get(f.path)
        if old is None:
            status = ChangeStatus.NEW
        elif old != f.content_hash:
            status = ChangeStatus.MODIFIED
        else:
            status = ChangeStatus.UNCHANGED
        changes.append(
            SourceChange(path=f.path, status=status, old_hash=old, new_hash=f.content_hash)
        )

    for path, old in previous.items():
        if path not in seen:
            changes.append(SourceChange(path=path, status=ChangeStatus.DELETED, old_hash=old))

    changes.sort(key=lambda c: c.path)
    return ChangeSet(changes=changes)


# -- helpers ---------------------------------------------------------------


def _duplicate_hashes(files: Sequence[IndexedFile]) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = {}
    for f in files:
        by_hash.setdefault(f.content_hash, []).append(f.path)
    return {h: sorted(paths) for h, paths in sorted(by_hash.items()) if len(paths) > 1}


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return None


def _as_str(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_tags(data: dict[str, object], parsed: ParsedMarkdown) -> tuple[str, ...]:
    """Union of frontmatter tags and inline ``#tags``, order-preserved."""
    out: list[str] = []
    raw = data.get("tags")
    if isinstance(raw, str):
        out.extend(t.strip() for t in raw.replace(",", " ").split())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                out.append(item.strip())
    out.extend(parsed.tags)
    seen: set[str] = set()
    result: list[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return tuple(result)
