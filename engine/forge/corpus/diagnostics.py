"""Structured, machine-readable diagnostics over an indexed corpus.

Every report here is **read-only and advisory**. Nothing in this module writes
to the vault. Repair proposals are emitted as data for a human to review and
approve, per ADR-001 D2 (segregated write-back, no automatic in-place
enrichment in Phase 1).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..parsing.frontmatter import CODE_DESCRIPTIONS, DiagnosticCode, Severity
from ..parsing.links import RESOLVING_STATUSES, STATUS_DESCRIPTIONS, LinkStatus
from .model import CorpusIndex


@dataclass
class FrontmatterReport:
    total_files: int
    with_frontmatter: int
    without_frontmatter: int
    valid: int
    invalid: int
    by_code: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    files_with_errors: list[str] = field(default_factory=list)
    repairable_files: int = 0
    repair_proposals: list[dict[str, Any]] = field(default_factory=list)
    coverage_by_folder: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_files": self.total_files,
                "with_frontmatter": self.with_frontmatter,
                "without_frontmatter": self.without_frontmatter,
                "valid": self.valid,
                "invalid": self.invalid,
                "repairable_files": self.repairable_files,
            },
            "by_code": self.by_code,
            "by_severity": self.by_severity,
            "code_descriptions": {c.value: d for c, d in CODE_DESCRIPTIONS.items()},
            "coverage_by_folder": self.coverage_by_folder,
            "files_with_errors": self.files_with_errors,
            "repair_proposals": self.repair_proposals,
        }


@dataclass
class LinkReport:
    total_links: int
    wikilinks: int
    markdown_links: int
    by_status: dict[str, int] = field(default_factory=dict)
    unresolved_total: int = 0
    unresolved_distinct: int = 0
    #: target -> {count, status, sources, candidates}
    unresolved_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    ambiguous_targets: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_links": self.total_links,
                "wikilinks": self.wikilinks,
                "markdown_links": self.markdown_links,
                "unresolved_occurrences": self.unresolved_total,
                "unresolved_distinct_targets": self.unresolved_distinct,
            },
            "by_status": self.by_status,
            "status_descriptions": {s.value: d for s, d in STATUS_DESCRIPTIONS.items()},
            "unresolved_targets": self.unresolved_targets,
            "ambiguous_targets": self.ambiguous_targets,
        }


def frontmatter_report(index: CorpusIndex) -> FrontmatterReport:
    by_code: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    files_with_errors: list[str] = []
    proposals: list[dict[str, Any]] = []
    repairable = 0
    folder_cov: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "with_fm": 0})

    with_fm = valid = 0
    for f in index.files:
        folder = f.path.split("/")[0] if "/" in f.path else "(root)"
        folder_cov[folder]["total"] += 1
        if f.frontmatter_present:
            with_fm += 1
            folder_cov[folder]["with_fm"] += 1
        if f.frontmatter_valid:
            valid += 1

        has_error = False
        for d in f.diagnostics:
            by_code[d.code.value] += 1
            by_severity[d.severity.value] += 1
            if d.severity is Severity.ERROR:
                has_error = True
        if has_error:
            files_with_errors.append(f.path)

        if f.repairs:
            verified = [r for r in f.repairs if r.verified]
            if verified:
                repairable += 1
            proposals.append(
                {
                    "path": f.path,
                    "repairs": [r.to_dict() for r in f.repairs],
                    "all_verified": bool(verified) and len(verified) == len(f.repairs),
                }
            )

    return FrontmatterReport(
        total_files=index.file_count,
        with_frontmatter=with_fm,
        without_frontmatter=index.file_count - with_fm,
        valid=valid,
        invalid=with_fm - valid,
        by_code=dict(sorted(by_code.items())),
        by_severity=dict(sorted(by_severity.items())),
        files_with_errors=sorted(files_with_errors),
        repairable_files=repairable,
        repair_proposals=sorted(proposals, key=lambda p: str(p["path"])),
        coverage_by_folder={k: dict(v) for k, v in sorted(folder_cov.items())},
    )


def link_report(index: CorpusIndex) -> LinkReport:
    by_status: Counter[str] = Counter()
    unresolved: dict[str, dict[str, Any]] = {}
    ambiguous: dict[str, list[str]] = {}
    wikilinks = mdlinks = 0

    for f in index.files:
        wikilinks += f.wikilink_count
        mdlinks += f.markdown_link_count
        for link in f.links:
            by_status[link.status.value] += 1

            if link.status is LinkStatus.AMBIGUOUS:
                ambiguous.setdefault(link.target, sorted(link.candidates))

            if link.status in RESOLVING_STATUSES:
                continue

            entry = unresolved.setdefault(
                link.target,
                {
                    "status": link.status.value,
                    "count": 0,
                    "sources": [],
                    "candidates": list(link.candidates),
                },
            )
            entry["count"] = int(entry["count"]) + 1
            sources = entry["sources"]
            assert isinstance(sources, list)
            if link.source_path not in sources:
                sources.append(link.source_path)

    total_links = sum(by_status.values())
    unresolved_total = sum(int(v["count"]) for v in unresolved.values())

    ordered = dict(
        sorted(unresolved.items(), key=lambda kv: (-int(kv[1]["count"]), kv[0]))
    )
    for v in ordered.values():
        v["sources"] = sorted(v["sources"])  # type: ignore[arg-type]

    return LinkReport(
        total_links=total_links,
        wikilinks=wikilinks,
        markdown_links=mdlinks,
        by_status=dict(sorted(by_status.items())),
        unresolved_total=unresolved_total,
        unresolved_distinct=len(ordered),
        unresolved_targets=ordered,
        ambiguous_targets=dict(sorted(ambiguous.items())),
    )
