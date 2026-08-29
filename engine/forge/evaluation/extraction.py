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
either present in its span or it is not, and that check is deterministic.
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
    llm_calls: int = 0
    error: str | None = None

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
            "llm_calls": self.llm_calls,
            "error": self.error,
        }


@dataclass
class ExtractionReport:
    model_id: str
    prompt_version: str
    scores: list[CaseScore] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def recall(self) -> float:
        return _mean([s.recall for s in self.scores])

    @property
    def junk_rate(self) -> float:
        """Share of emitted concepts that are on the forbidden list.

        The headline number. Recall alone rewards a greedy extractor, which is
        the failure this corpus actually had.
        """
        emitted = sum(len(s.found) + len(s.junk) + len(s.extra) for s in self.scores)
        junk = sum(len(s.junk) for s in self.scores)
        return junk / emitted if emitted else 0.0

    @property
    def grounding_rate(self) -> float:
        total = sum(s.claims for s in self.scores)
        grounded = sum(s.grounded_claims for s in self.scores)
        return grounded / total if total else 1.0

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "cases": len(self.scores),
            "recall": round(self.recall, 3),
            "junk_rate": round(self.junk_rate, 3),
            "grounding_rate": round(self.grounding_rate, 3),
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 2),
            "scores": [s.to_dict() for s in self.scores],
        }

    def summary_line(self) -> str:
        return (
            f"{self.model_id}  prompt={self.prompt_version}  "
            f"recall={self.recall:.2f}  junk={self.junk_rate:.2f}  "
            f"grounded={self.grounding_rate:.2f}  calls={self.llm_calls}"
        )


def score_case(
    case: ExtractionCase,
    concepts: Sequence[str],
    claims: Sequence[tuple[str, str]] = (),
    *,
    llm_calls: int = 0,
    error: str | None = None,
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
        llm_calls=llm_calls,
        error=error,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
                score_case(case, [], error=f"{type(exc).__name__}: {exc}"[:200])
            )
            continue
        report.scores.append(
            score_case(
                case,
                [c.name for c in result.concepts],
                [(c.statement, c.evidence_quote) for c in result.claims],
                llm_calls=result.llm_calls,
            )
        )

    report.duration_seconds = time.perf_counter() - started
    return report
