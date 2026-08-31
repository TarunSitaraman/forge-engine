"""Extraction quality: dataset, scoring, and report.

**The gap this closes.** Every extraction comparison made in August 2026 was
confounded, because nothing measured extraction. A prompt rewrite was judged by
reading fifty proposals by hand; a reasoning-off experiment was run against the
*assessment* set because no extraction set existed, and reasoning-off then
turned out to have silently governed a 5.66-hour run. `docs/research/` records
both as unmeasurable. This makes them measurable.

**What is scored, and why these things.** Over-extraction was the observed
failure — a 25-concept sample was roughly 35-40% usable, returning `RAM`,
`Answer`, `Fluency`, `maxmemory` and `VARCHAR(n)` as concepts. So the headline
metric is not recall but **junk rate**: how often the extractor emits something
from a list of strings it has actually produced on this corpus and should not.
A set that only rewarded recall would score a maximally greedy extractor best,
which is precisely the failure mode.

Claims are scored on **grounding**, which needs no labels at all: a quote is
either present in its span or it is not, and that check is deterministic. The
denominator has to include the claims the extractor *dropped* for exactly that
reason, though — its returned `claims` have already passed `_grounded`, so
scoring only those reports 1.000 for every model ever tested.

**A case whose calls did not all return is not scored.** A timeout does not
raise; it returns a truncated result, and a case that emitted nothing cannot
emit anything forbidden. Folding those in made the first real run report
`junk=0.00` partly because output was missing rather than clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..extraction.extractor import _grounded
from ..parsing.links import normalize

DEFAULT_EXTRACTION_SET = Path("tests") / "fixtures" / "eval" / "extraction-v1.yaml"


class ExtractionDatasetError(Exception):
    """The evaluation set is malformed."""


@dataclass(frozen=True)
class ExtractionCase:
    id: str
    text: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    note: str = ""


@dataclass
class ExtractionDataset:
    version: int
    cases: list[ExtractionCase] = field(default_factory=list)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def load(cls, path: Path | None = None) -> ExtractionDataset:
        target = Path(path or DEFAULT_EXTRACTION_SET)
        if not target.is_file():
            raise ExtractionDatasetError(f"no extraction set at {target}")
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or "cases" not in raw:
            raise ExtractionDatasetError(f"{target}: expected a mapping with 'cases'")

        cases: list[ExtractionCase] = []
        for entry in raw["cases"]:
            missing = {"id", "text", "expected", "forbidden"} - set(entry)
            if missing:
                raise ExtractionDatasetError(
                    f"{target}: case {entry.get('id', '?')} missing {sorted(missing)}"
                )
            cases.append(
                ExtractionCase(
                    id=str(entry["id"]),
                    text=str(entry["text"]),
                    expected=tuple(entry["expected"]),
                    forbidden=tuple(entry["forbidden"]),
                    note=str(entry.get("note", "")),
                )
            )
        return cls(version=int(raw.get("version", 1)), cases=cases, path=target)


@dataclass
class CaseScore:
    """One case, scored. Names the specific strings that hit or missed."""

    case_id: str
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    junk: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    claims: int = 0
    grounded_claims: int = 0
    #: Claims the extractor discarded as ungrounded before returning. They are
    #: the only reason the grounding rate can move: `result.claims` holds
    #: survivors of the same check this eval applies, so scoring survivors
    #: alone yields 1.000 by construction and measures nothing.
    dropped_claims: int = 0
    llm_calls: int = 0
    error: str | None = None
    #: The extractor's own verdict on this case. Anything but "succeeded"
    #: means some call did not return, so the case's emitted set is truncated
    #: and scoring it would understate junk and recall alike.
    status: str = "succeeded"
    #: Failure kinds the extractor reported — `llm_error`, `ungrounded_claim`.
    failures: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Did every call for this case return? Only then is the score real."""
        return self.status == "succeeded" and self.error is None

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "recall": round(self.recall, 3),
            "found": self.found,
            "missed": self.missed,
            "junk": self.junk,
            "extra": self.extra,
            "claims": self.claims,
            "grounded_claims": self.grounded_claims,
            "dropped_claims": self.dropped_claims,
            "llm_calls": self.llm_calls,
            "error": self.error,
            "status": self.status,
            "complete": self.complete,
            "failures": self.failures,
        }


@dataclass
class ExtractionReport:
    model_id: str
    prompt_version: str
    scores: list[CaseScore] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def complete(self) -> list[CaseScore]:
        """Cases every call returned for. Every rate below is over these only.

        A timed-out case emits nothing, and nothing cannot be junk — so folding
        it in makes a broken run look clean. Observed on the first real run,
        2026-08-29: four timeouts in a 12-call run reported `junk=0.00`, which
        was in part the absence of output rather than the absence of junk.
        """
        return [s for s in self.scores if s.complete]

    @property
    def failed(self) -> list[CaseScore]:
        return [s for s in self.scores if not s.complete]

    @property
    def trustworthy(self) -> bool:
        """No score is quotable as a property of the model unless this holds."""
        return not self.failed and bool(self.scores)

    @property
    def recall(self) -> float:
        return _mean([s.recall for s in self.complete])

    @property
    def junk_rate(self) -> float:
        """Share of emitted concepts that are on the forbidden list.

        The headline number. Recall alone rewards a greedy extractor, which is
        the failure this corpus actually had.
        """
        emitted = sum(len(s.found) + len(s.junk) + len(s.extra) for s in self.complete)
        junk = sum(len(s.junk) for s in self.complete)
        return junk / emitted if emitted else 0.0

    @property
    def grounding_rate(self) -> float:
        """Share of claims the model produced whose quote was really in the span.

        The denominator must include the claims the extractor *dropped*. Its
        `claims` list has already survived `_grounded`, so re-checking only
        those returns 1.000 for any model, however badly it quotes — a metric
        that cannot fail is worse than no metric, because it reassures.
        """
        kept = sum(s.claims for s in self.complete)
        dropped = sum(s.dropped_claims for s in self.complete)
        grounded = sum(s.grounded_claims for s in self.complete)
        total = kept + dropped
        return grounded / total if total else 1.0

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "cases": len(self.scores),
            "scored": len(self.complete),
            "failed": len(self.failed),
            "trustworthy": self.trustworthy,
            "recall": round(self.recall, 3),
            "junk_rate": round(self.junk_rate, 3),
            "grounding_rate": round(self.grounding_rate, 3),
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 2),
            "scores": [s.to_dict() for s in self.scores],
        }

    def summary_line(self) -> str:
        scope = (
            f"{len(self.complete)}/{len(self.scores)} cases"
            if not self.trustworthy
            else f"{len(self.scores)} cases"
        )
        return (
            f"{self.model_id}  prompt={self.prompt_version}  "
            f"recall={self.recall:.2f}  junk={self.junk_rate:.2f}  "
            f"grounded={self.grounding_rate:.2f}  calls={self.llm_calls}  {scope}"
        )


def score_case(
    case: ExtractionCase,
    concepts: Sequence[str],
    claims: Sequence[tuple[str, str]] = (),
    *,
    llm_calls: int = 0,
    error: str | None = None,
    status: str = "succeeded",
    failures: Sequence[str] = (),
    dropped_claims: int = 0,
) -> CaseScore:
    """Score one case's extracted concepts and claims.

    Matching is through `normalize()` — the same comparison the link resolver
    and identity config use — so `B-tree index` and `B Tree Index` are one
    concept rather than a miss plus a false positive.
    """
    emitted = {normalize(c): c for c in concepts if c and c.strip()}
    expected = {normalize(e): e for e in case.expected}
    forbidden = {normalize(f): f for f in case.forbidden}

    found = [expected[k] for k in expected if k in emitted]
    missed = [expected[k] for k in expected if k not in emitted]
    junk = [emitted[k] for k in emitted if k in forbidden]
    extra = [emitted[k] for k in emitted if k not in expected and k not in forbidden]

    grounded = sum(1 for _, quote in claims if _grounded(quote, case.text))
    return CaseScore(
        case_id=case.id,
        found=sorted(found),
        missed=sorted(missed),
        junk=sorted(junk),
        extra=sorted(extra),
        claims=len(claims),
        grounded_claims=grounded,
        dropped_claims=dropped_claims,
        llm_calls=llm_calls,
        error=error,
        status=status,
        failures=list(failures),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _failure_lines(failures: Sequence[dict]) -> list[str]:
    """Deduplicate failures to `kind: message`, keeping the message.

    Six spans failing on one rejected model name is one fact, not six, but
    dropping the message leaves `llm_error` — which names the layer that
    caught the error and nothing about the error.
    """
    seen: dict[str, None] = {}
    for failure in failures:
        kind = str(failure.get("kind", "unknown"))
        message = str(failure.get("error", "")).strip()
        seen.setdefault(f"{kind}: {message}" if message else kind, None)
    return list(seen)


def run(
    dataset: ExtractionDataset,
    extractor: Any,
    *,
    span_factory: Any = None,
) -> ExtractionReport:
    """Drive the production extractor over the set, one span per case.

    Uses `CandidateExtractor` itself rather than re-implementing the prompt
    path, so what is scored is what ships — the same reason the assessment eval
    drives the real `EvidenceAssessor`.
    """
    import time

    from ..domain import Span
    from ..ids import text_hash

    def _span(case: ExtractionCase) -> Any:
        return Span(
            id=f"eval-{case.id}",
            document_id="eval-doc",
            ordinal=0,
            locator="L1",
            start_line=1,
            end_line=1,
            text=case.text,
            content_hash=text_hash(case.text),
        )

    make_span = span_factory or _span
    report = ExtractionReport(
        model_id=extractor.model_id(),
        prompt_version=extractor.prompt_version,
    )
    started = time.perf_counter()

    for case in dataset:
        try:
            result = extractor.extract([make_span(case)])
        except Exception as exc:  # a provider failure is a result, not a crash
            report.scores.append(
                score_case(
                    case, [], error=f"{type(exc).__name__}: {exc}"[:200], status="failed"
                )
            )
            continue
        # A timeout does not raise: `extract` catches it per call and returns
        # a PARTIAL result with fewer concepts. Reading only the exception path
        # scored a truncated case as a clean one.
        report.scores.append(
            score_case(
                case,
                [c.name for c in result.concepts],
                [(c.statement, c.evidence_quote) for c in result.claims],
                llm_calls=result.llm_calls,
                status=result.status.value,
                failures=_failure_lines(result.failures),
                dropped_claims=sum(
                    1 for f in result.failures if f.get("kind") == "ungrounded_quote"
                ),
            )
        )

    report.duration_seconds = time.perf_counter() - started
    return report
