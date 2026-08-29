"""What would an extraction run cost, before you start it? Zero model calls.

Extraction is the only expensive thing this engine does — measured at 49.0 s
per call on the 8B local box, and 3,372 calls to cover the vault. Until now the
only way to learn a run's size was to start it and watch, which is exactly
backwards for a decision about whether to spend three hours.

Every input to the answer is already deterministic and already stored: which
spans the ingestion chunker produced, which of those the extractor would
select, and whether the derivation cache already holds a result for the
source's content hash. So the cost of an extraction run is *computable*, and
this module computes it.

**It calls the real selection and the real key.** ``ExtractionPlanner`` uses
``CandidateExtractor._select`` and ``extraction_key`` rather than restating
their rules. A cost preview that reimplements either would drift away from what
actually runs — and would then be worse than no preview, because it would be
believed. The same reasoning that made ``_spans_for_source`` filter by chunker:
predicting the cost of the wrong chunking is how 98 spans were reported as 208.

**Sources that were never ingested are reported, not estimated.** Their span
count is not knowable without chunking them, and this repo does not put a
number on something it did not measure.

One known way the plan can under-count: a source is only cached when its
extraction *succeeded*, so if the first of two byte-identical files fails, the
second pays again. The plan prices the run that works, which is the number
worth knowing before starting one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..extraction.extractor import CandidateExtractor
from ..logging import get_logger
from .chunking import CHUNK_STRATEGY
from .derivation import extraction_key

log = get_logger(__name__)

#: Model calls the extractor spends per selected span: one for concepts, one
#: for claims. Asserted by a test against the real extract loop, so a third
#: call added later cannot silently halve every estimate ever printed.
CALLS_PER_SPAN = 2


@dataclass
class SourcePlan:
    """The predicted cost of extracting one source."""

    locator: str
    #: cached | pending | duplicate | not_ingested | no_spans
    state: str
    spans: int = 0
    selected: int = 0
    calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "state": self.state,
            "spans": self.spans,
            "selected": self.selected,
            "calls": self.calls,
        }


@dataclass
class ExtractionPlan:
    """What a run over some path would cost, and what it cannot know."""

    model_id: str
    prompt_version: str
    max_spans: int
    sources: list[SourcePlan] = field(default_factory=list)
    #: Seconds per call, when a measured rate was supplied. Never a default:
    #: a wall-clock estimate with an invented rate is a fabricated measurement.
    seconds_per_call: float | None = None
    duration_seconds: float = 0.0

    @property
    def calls(self) -> int:
        return sum(s.calls for s in self.sources)

    @property
    def cached(self) -> list[SourcePlan]:
        return [s for s in self.sources if s.state == "cached"]

    @property
    def pending(self) -> list[SourcePlan]:
        return [s for s in self.sources if s.state == "pending"]

    @property
    def duplicates(self) -> list[SourcePlan]:
        """Sources whose bytes another source in this run already pays for."""
        return [s for s in self.sources if s.state == "duplicate"]

    @property
    def unknown(self) -> list[SourcePlan]:
        """Sources whose cost cannot be computed because they were never ingested."""
        return [s for s in self.sources if s.state == "not_ingested"]

    @property
    def estimated_hours(self) -> float | None:
        if self.seconds_per_call is None:
            return None
        return self.calls * self.seconds_per_call / 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "max_spans": self.max_spans,
            "calls": self.calls,
            "sources": len(self.sources),
            "cached": len(self.cached),
            "pending": len(self.pending),
            "duplicate": len(self.duplicates),
            "not_ingested": len(self.unknown),
            "seconds_per_call": self.seconds_per_call,
            "estimated_hours": self.estimated_hours,
            "duration_seconds": round(self.duration_seconds, 3),
            "detail": [s.to_dict() for s in self.sources],
        }


class ExtractionPlanner:
    """Compute an extraction run's cost without making a single call.

    Takes the same pipeline the run would use, so the discovery rules, the
    exclude list and the chunker filter are shared rather than copied.
    """

    def __init__(self, pipeline, extractor: CandidateExtractor) -> None:
        self.pipeline = pipeline
        self.store = pipeline.store
        self.extractor = extractor

    def plan(self, path: Path) -> ExtractionPlan:
        started = time.perf_counter()
        plan = ExtractionPlan(
            model_id=self.extractor.model_id(),
            prompt_version=self.extractor.prompt_version,
            max_spans=self.extractor.max_spans,
        )

        targets = self.pipeline._discover(path) if path.is_dir() else [path]

        # The derivation key is the *content* hash, so two files holding the
        # same bytes share one cache entry: the first pays, the second is a
        # hit. Pricing them independently inflated the fixture vault's estimate
        # from 16 calls to 18, and would inflate the real one further — six
        # project packs contain an `01-overview.md`, and `_index.md` recurs
        # throughout. Discovery order is shared with the run, so whichever
        # source the run charges is the one charged here.
        paid: set[str] = set()
        for target in targets:
            plan.sources.append(self._plan_one(target, paid))

        plan.duration_seconds = time.perf_counter() - started
        log.info(
            "extraction_plan",
            calls=plan.calls,
            pending=len(plan.pending),
            cached=len(plan.cached),
            not_ingested=len(plan.unknown),
        )
        return plan

    def _plan_one(self, path: Path, paid: set[str]) -> SourcePlan:
        locator = self.pipeline._locator(path)
        source = self.store.get_source_by_locator(locator)
        if source is None:
            # Its span count is not knowable without chunking it, and guessing
            # one would put a made-up number in a cost report.
            return SourcePlan(locator=locator, state="not_ingested")

        key = extraction_key(
            content_hash=source.content_hash,
            processor_version=self.extractor.version,
            model_id=self.extractor.model_id(),
            prompt_version=self.extractor.prompt_version,
            schema_version=self.extractor.schema_version,
        )
        if self.store.get_derivation(key.value()) is not None:
            return SourcePlan(locator=locator, state="cached")
        if key.value() in paid:
            return SourcePlan(locator=locator, state="duplicate")

        spans = [
            span
            for document in self.store.documents_for_source(source.id)
            for span in self.store.spans_for_document(document.id)
            if span.chunk_strategy == CHUNK_STRATEGY
        ]
        if not spans:
            return SourcePlan(locator=locator, state="no_spans")

        selected = self.extractor._select(spans)
        paid.add(key.value())
        return SourcePlan(
            locator=locator,
            state="pending",
            spans=len(spans),
            selected=len(selected),
            calls=len(selected) * CALLS_PER_SPAN,
        )
