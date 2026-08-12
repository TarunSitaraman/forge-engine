"""Retrieval metrics: Recall@k, Precision@k, MRR.

Standard definitions, implemented plainly. Two choices worth stating because
they change what the numbers mean:

* **Ranking is at the source-document level**, not the span level. A query is
  answered by a document; retrieving three spans of the same document is one
  result, not three. Ranking spans would inflate precision for any document
  that happened to be chunked finely.
* **Recall@k is capped by k.** With four relevant documents and k=5 a perfect
  system scores 1.0; with six relevant documents and k=5 the ceiling is 5/6.
  This is reported rather than hidden, because a metric with an unreachable
  maximum invites misreading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class QueryScore:
    """Metrics for one query."""

    query_id: str
    category: str
    retrieved: tuple[str, ...]
    relevant: tuple[str, ...]

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    reciprocal_rank: float = 0.0
    first_hit_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "category": self.category,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "rr": round(self.reciprocal_rank, 4),
            "first_hit_rank": self.first_hit_rank,
            "retrieved_top5": list(self.retrieved[:5]),
            "relevant": list(self.relevant),
        }


@dataclass
class MetricSummary:
    """Aggregate metrics for one retrieval method."""

    method: str
    queries: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    #: Queries where nothing relevant appeared in the top 10 at all.
    total_misses: int = 0
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0
    scores: list[QueryScore] = field(default_factory=list)

    def to_dict(self, *, include_scores: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "queries": self.queries,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "mrr": round(self.mrr, 4),
            "total_misses": self.total_misses,
            "latency_ms_per_query": round(self.latency_ms, 2),
            "by_category": {
                cat: {k: round(v, 4) for k, v in vals.items()}
                for cat, vals in sorted(self.by_category.items())
            },
        }
        if include_scores:
            out["scores"] = [s.to_dict() for s in self.scores]
        return out

    def headline(self) -> str:
        return (
            f"{self.method:<22} R@5={self.recall_at_5:.3f}  R@10={self.recall_at_10:.3f}  "
            f"P@5={self.precision_at_5:.3f}  MRR={self.mrr:.3f}  "
            f"misses={self.total_misses}  {self.latency_ms:.1f}ms/q"
        )


def score_query(
    query_id: str, category: str, retrieved: Sequence[str], relevant: Sequence[str]
) -> QueryScore:
    """Score one ranked result list against its labels."""
    relevant_set = set(relevant)
    # De-duplicate while preserving rank order: the same document retrieved
    # twice occupies one rank, not two.
    ordered: list[str] = []
    for item in retrieved:
        if item not in ordered:
            ordered.append(item)

    score = QueryScore(
        query_id=query_id,
        category=category,
        retrieved=tuple(ordered),
        relevant=tuple(relevant),
    )
    if not relevant_set:
        return score

    top5, top10 = ordered[:5], ordered[:10]
    hits5 = [d for d in top5 if d in relevant_set]
    hits10 = [d for d in top10 if d in relevant_set]

    score.recall_at_5 = len(hits5) / len(relevant_set)
    score.recall_at_10 = len(hits10) / len(relevant_set)
    score.precision_at_5 = len(hits5) / 5.0

    for rank, doc in enumerate(ordered, start=1):
        if doc in relevant_set:
            score.reciprocal_rank = 1.0 / rank
            score.first_hit_rank = rank
            break

    return score


def summarize(method: str, scores: Sequence[QueryScore], *, latency_ms: float = 0.0) -> MetricSummary:
    """Aggregate per-query scores into a summary."""
    summary = MetricSummary(method=method, queries=len(scores), latency_ms=latency_ms)
    if not scores:
        return summary

    summary.recall_at_5 = _mean(s.recall_at_5 for s in scores)
    summary.recall_at_10 = _mean(s.recall_at_10 for s in scores)
    summary.precision_at_5 = _mean(s.precision_at_5 for s in scores)
    summary.mrr = _mean(s.reciprocal_rank for s in scores)
    summary.total_misses = sum(1 for s in scores if s.first_hit_rank is None or s.first_hit_rank > 10)
    summary.scores = list(scores)

    by_category: dict[str, list[QueryScore]] = {}
    for score in scores:
        by_category.setdefault(score.category, []).append(score)
    summary.by_category = {
        category: {
            "recall@5": _mean(s.recall_at_5 for s in group),
            "recall@10": _mean(s.recall_at_10 for s in group),
            "precision@5": _mean(s.precision_at_5 for s in group),
            "mrr": _mean(s.reciprocal_rank for s in group),
            "queries": float(len(group)),
        }
        for category, group in sorted(by_category.items())
    }
    return summary


def compare(baseline: MetricSummary, candidate: MetricSummary) -> dict[str, Any]:
    """Difference between two methods, with an explicit verdict.

    The verdict is deliberately conservative: a change under one point of
    recall is reported as noise on a 24-query set, because it is.
    """
    deltas = {
        "recall@5": candidate.recall_at_5 - baseline.recall_at_5,
        "recall@10": candidate.recall_at_10 - baseline.recall_at_10,
        "precision@5": candidate.precision_at_5 - baseline.precision_at_5,
        "mrr": candidate.mrr - baseline.mrr,
    }
    primary = deltas["recall@10"]
    if abs(primary) < 0.01 and abs(deltas["mrr"]) < 0.01:
        verdict = "no measurable difference"
    elif primary > 0 or (abs(primary) < 0.01 and deltas["mrr"] > 0):
        verdict = "improvement"
    else:
        verdict = "regression"

    return {
        "baseline": baseline.method,
        "candidate": candidate.method,
        "deltas": {k: round(v, 4) for k, v in deltas.items()},
        "verdict": verdict,
        "note": (
            f"{baseline.queries} queries; differences below 0.01 are treated as noise "
            f"at this sample size"
        ),
    }


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
