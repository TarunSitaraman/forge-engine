"""Retrieval evaluation: labelled dataset, metrics, and a comparison runner."""

from .assessment import (
    DEFAULT_ASSESSMENT_SET,
    AssessmentCase,
    AssessmentDataset,
    AssessmentDatasetError,
    AssessmentReport,
    CaseResult,
)
from .dataset import DEFAULT_DATASET, DatasetError, EvalDataset, EvalQuery
from .metrics import MetricSummary, QueryScore, compare, score_query, summarize
from .runner import DEFAULT_FUSION_WEIGHTS, EvaluationRun, RetrievalEvaluator

__all__ = [
    "EvalDataset",
    "EvalQuery",
    "DatasetError",
    "DEFAULT_ASSESSMENT_SET",
    "AssessmentCase",
    "AssessmentDataset",
    "AssessmentDatasetError",
    "AssessmentReport",
    "CaseResult",
    "DEFAULT_DATASET",
    "RetrievalEvaluator",
    "EvaluationRun",
    "DEFAULT_FUSION_WEIGHTS",
    "MetricSummary",
    "QueryScore",
    "score_query",
    "summarize",
    "compare",
]
