"""Corpus statistics.

Numbers here are computed from the filesystem on every run. The Phase 0 audit
found stale hand-maintained counts in three separate files; the fix is that no
count in Forge is ever maintained by hand again.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .model import CorpusIndex


@dataclass
class CorpusStats:
    file_count: int
    total_lines: int
    total_bytes: int
    by_folder: dict[str, dict[str, int]] = field(default_factory=dict)
    frontmatter_coverage_pct: float = 0.0
    doc_types: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    top_tags: dict[str, int] = field(default_factory=dict)
    canonical_count: int = 0
    heading_total: int = 0
    code_blocks: int = 0
    code_languages: dict[str, int] = field(default_factory=dict)
    wikilink_total: int = 0
    markdown_link_total: int = 0
    related_field_total: int = 0
    duplicate_hash_groups: int = 0
    duplicate_files: list[list[str]] = field(default_factory=list)
    largest_files: list[dict[str, Any]] = field(default_factory=list)
    filename_styles: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_count": self.file_count,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "frontmatter_coverage_pct": self.frontmatter_coverage_pct,
            "canonical_count": self.canonical_count,
            "heading_total": self.heading_total,
            "code_blocks": self.code_blocks,
            "wikilink_total": self.wikilink_total,
            "markdown_link_total": self.markdown_link_total,
            "related_field_total": self.related_field_total,
            "duplicate_hash_groups": self.duplicate_hash_groups,
            "by_folder": self.by_folder,
            "doc_types": self.doc_types,
            "statuses": self.statuses,
            "top_tags": self.top_tags,
            "code_languages": self.code_languages,
            "filename_styles": self.filename_styles,
            "duplicate_files": self.duplicate_files,
            "largest_files": self.largest_files,
        }


def compute_stats(index: CorpusIndex, *, top_n: int = 15) -> CorpusStats:
    by_folder: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "lines": 0, "with_frontmatter": 0}
    )
    doc_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    styles: Counter[str] = Counter()

    total_lines = total_bytes = headings = code_blocks = 0
    wl = ml = related = canonical = with_fm = 0

    for f in index.files:
        folder = f.path.split("/")[0] if "/" in f.path else "(root)"
        by_folder[folder]["files"] += 1
        by_folder[folder]["lines"] += f.line_count
        if f.frontmatter_present:
            by_folder[folder]["with_frontmatter"] += 1
            with_fm += 1

        total_lines += f.line_count
        total_bytes += f.byte_size
        headings += f.heading_count
        code_blocks += f.code_block_count
        wl += f.wikilink_count
        ml += f.markdown_link_count
        related += len(f.related)
        if f.canonical:
            canonical += 1
        if f.doc_type:
            doc_types[f.doc_type] += 1
        if f.status:
            statuses[f.status] += 1
        for t in f.tags:
            tags[t] += 1
        for lang in f.code_languages:
            languages[lang] += 1
        styles[_filename_style(f.path)] += 1

    largest = sorted(index.files, key=lambda f: -f.line_count)[:top_n]

    return CorpusStats(
        file_count=index.file_count,
        total_lines=total_lines,
        total_bytes=total_bytes,
        by_folder={k: dict(v) for k, v in sorted(by_folder.items(), key=lambda kv: -kv[1]["lines"])},
        frontmatter_coverage_pct=round(100 * with_fm / index.file_count, 1) if index.file_count else 0.0,
        doc_types=dict(doc_types.most_common()),
        statuses=dict(statuses.most_common()),
        top_tags=dict(tags.most_common(top_n)),
        canonical_count=canonical,
        heading_total=headings,
        code_blocks=code_blocks,
        code_languages=dict(languages.most_common()),
        wikilink_total=wl,
        markdown_link_total=ml,
        related_field_total=related,
        duplicate_hash_groups=len(index.duplicate_hashes),
        duplicate_files=[paths for paths in index.duplicate_hashes.values()],
        largest_files=[{"path": f.path, "lines": f.line_count} for f in largest],
        filename_styles=dict(styles.most_common()),
    )


def _filename_style(path: str) -> str:
    """Classify filename convention — the corpus uses two conflicting systems."""
    import re

    name = path.rsplit("/", 1)[-1]
    if re.match(r"^[a-z0-9]+(-[a-z0-9]+)*\.md$", name):
        return "kebab-case"
    if re.match(r"^[A-Z0-9_]+\.md$", name):
        return "SHOUT_CASE"
    if " " in name:
        return "Title Case with spaces"
    return "other"
