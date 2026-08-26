"""Ingestion reporting — the observability surface.

Structured logs plus a typed result record. Not an observability platform:
enough to answer "what happened, how long did it take, what did it cost", and
nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import ExtractionStatus, IngestionStatus
from .derivation import CacheStats


@dataclass
class SourceReport:
    """Outcome for one ingested source."""

    locator: str
    status: IngestionStatus
    source_id: str | None = None
    document_id: str | None = None
    content_hash: str | None = None

    spans: int = 0
    pages: int | None = None
    chars: int = 0

    extraction_status: ExtractionStatus = ExtractionStatus.SKIPPED_NO_PROVIDER
    concepts_proposed: int = 0
    claims_proposed: int = 0
    evidence_links: int = 0
    proposals_created: int = 0

    llm_calls: int = 0
    duration_seconds: float = 0.0
    detail: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    #: The provider itself became unusable while processing this source.
    provider_unavailable: bool = False
    provider_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (IngestionStatus.INGESTED, IngestionStatus.UNCHANGED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "status": self.status.value,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "spans": self.spans,
            "pages": self.pages,
            "chars": self.chars,
            "extraction_status": self.extraction_status.value,
            "concepts_proposed": self.concepts_proposed,
            "claims_proposed": self.claims_proposed,
            "evidence_links": self.evidence_links,
            "proposals_created": self.proposals_created,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "detail": self.detail,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class IngestionReport:
    """Aggregate outcome for one ingestion run."""

    sources: list[SourceReport] = field(default_factory=list)
    cache: CacheStats = field(default_factory=CacheStats)
    duration_seconds: float = 0.0

    #: The run stopped early rather than completing every target.
    aborted: bool = False
    abort_reason: str | None = None
    #: How many targets the run set out to process, so an abort can say
    #: "3 of 19" rather than just reporting the three it reached.
    attempted_targets: int = 0

    def add(self, report: SourceReport) -> SourceReport:
        self.sources.append(report)
        return report

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.sources)

    def by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sources:
            counts[s.status.value] = counts.get(s.status.value, 0) + 1
        return dict(sorted(counts.items()))

    def totals(self) -> dict[str, int]:
        return {
            "sources": len(self.sources),
            "spans": sum(s.spans for s in self.sources),
            "concepts_proposed": sum(s.concepts_proposed for s in self.sources),
            "claims_proposed": sum(s.claims_proposed for s in self.sources),
            "evidence_links": sum(s.evidence_links for s in self.sources),
            "proposals_created": sum(s.proposals_created for s in self.sources),
            "llm_calls": self.llm_calls,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": self.totals(),
            "by_status": self.by_status(),
            "cache": self.cache.to_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "sources": [s.to_dict() for s in self.sources],
        }

    def summary_line(self) -> str:
        t = self.totals()
        return (
            f"{t['sources']} source(s), {t['spans']} spans, "
            f"{t['concepts_proposed']} concepts, {t['claims_proposed']} claims, "
            f"{t['proposals_created']} proposals, {t['llm_calls']} LLM calls"
        )
