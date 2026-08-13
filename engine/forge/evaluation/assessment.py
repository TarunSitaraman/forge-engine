"""Evidence-assessment evaluation harness.

Measures the *pipeline*, not the model. Given an answer from some provider, the
questions this harness asks are:

* Did the structured output validate?
* Did every citation resolve to a span that was actually shown?
* Did the classification produce the right proposal — or correctly produce
  none?
* Did a repeated run hit the cache instead of paying again?
* How long did it take?

With the scripted provider (the CI default) the model's answer is fixed, so
classification accuracy is 1.0 by construction and is *not* evidence about any
real model. That is stated in the output rather than left for the reader to
infer. Point the harness at a real provider and the same cases measure
agreement instead — a genuinely different, and much weaker, claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..domain import AssessmentClass, ProposalType
from ..logging import get_logger

log = get_logger(__name__)

DEFAULT_ASSESSMENT_SET = Path("tests") / "fixtures" / "eval" / "assessment-v1.yaml"


class AssessmentDatasetError(Exception):
    """The evaluation set is malformed."""


@dataclass(frozen=True)
class AssessmentCase:
    id: str
    claim: str
    evidence: str
    expected_classification: AssessmentClass
    expected_proposal: ProposalType | None
    refined_statement: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expected_classification": self.expected_classification.value,
            "expected_proposal": self.expected_proposal.value if self.expected_proposal else None,
        }


@dataclass
class AssessmentDataset:
    version: int
    cases: list[AssessmentCase] = field(default_factory=list)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def by_classification(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            key = case.expected_classification.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    @classmethod
    def load(cls, path: Path | None = None) -> AssessmentDataset:
        target = Path(path or DEFAULT_ASSESSMENT_SET)
        if not target.is_file():
            raise AssessmentDatasetError(f"assessment set not found: {target}")

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or "cases" not in raw:
            raise AssessmentDatasetError(f"{target}: expected a mapping with a 'cases' key")

        cases: list[AssessmentCase] = []
        seen: set[str] = set()
        for entry in raw["cases"]:
            for required in ("id", "claim", "evidence", "expected_classification"):
                if required not in entry:
                    raise AssessmentDatasetError(f"{target}: case missing {required!r}: {entry}")
            if entry["id"] in seen:
                raise AssessmentDatasetError(f"{target}: duplicate case id {entry['id']!r}")
            seen.add(entry["id"])

            expected_proposal = entry.get("expected_proposal")
            cases.append(
                AssessmentCase(
                    id=str(entry["id"]),
                    claim=str(entry["claim"]).strip(),
                    evidence=str(entry["evidence"]).strip(),
                    expected_classification=AssessmentClass(entry["expected_classification"]),
                    expected_proposal=(
                        ProposalType(expected_proposal) if expected_proposal else None
                    ),
                    refined_statement=str(entry.get("refined_statement", "")),
                    note=str(entry.get("note", "")),
                )
            )

        return cls(version=int(raw.get("version", 1)), cases=cases, path=target)


@dataclass
class CaseResult:
    case_id: str
    expected: str
    actual: str | None = None
    structured_output_valid: bool = False
    grounded: bool = False
    proposal_correct: bool = False
    classification_correct: bool = False
    cached_on_repeat: bool = False
    latency_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_id,
            "expected": self.expected,
            "actual": self.actual,
            "structured_output_valid": self.structured_output_valid,
            "grounded": self.grounded,
            "classification_correct": self.classification_correct,
            "proposal_correct": self.proposal_correct,
            "cached_on_repeat": self.cached_on_repeat,
            "latency_ms": round(self.latency_ms, 2),
            "detail": self.detail,
        }


@dataclass
class AssessmentReport:
    provider_id: str
    model_id: str
    scripted: bool
    results: list[CaseResult] = field(default_factory=list)

    def _rate(self, attribute: str) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if getattr(r, attribute)) / len(self.results)

    @property
    def structured_output_validity(self) -> float:
        return self._rate("structured_output_valid")

    @property
    def grounding_rate(self) -> float:
        return self._rate("grounded")

    @property
    def classification_accuracy(self) -> float:
        return self._rate("classification_correct")

    @property
    def proposal_correctness(self) -> float:
        return self._rate("proposal_correct")

    @property
    def cache_effectiveness(self) -> float:
        return self._rate("cached_on_repeat")

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def to_dict(self, *, include_cases: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "scripted": self.scripted,
            "cases": len(self.results),
            "structured_output_validity": round(self.structured_output_validity, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "classification_accuracy": round(self.classification_accuracy, 4),
            "proposal_correctness": round(self.proposal_correctness, 4),
            "cache_effectiveness": round(self.cache_effectiveness, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }
        if self.scripted:
            payload["caveat"] = (
                "Run against a scripted provider: classification_accuracy is 1.0 by "
                "construction and says nothing about any real model. The pipeline "
                "metrics (structured output, grounding, proposal correctness, cache) "
                "are meaningful."
            )
        if include_cases:
            payload["results"] = [r.to_dict() for r in self.results]
        return payload

    def headline(self) -> str:
        return (
            f"{self.provider_id}/{self.model_id:<20} "
            f"valid={self.structured_output_validity:.2f} "
            f"grounded={self.grounding_rate:.2f} "
            f"class={self.classification_accuracy:.2f} "
            f"proposal={self.proposal_correctness:.2f} "
            f"cache={self.cache_effectiveness:.2f} "
            f"{self.mean_latency_ms:.0f}ms/case"
        )
