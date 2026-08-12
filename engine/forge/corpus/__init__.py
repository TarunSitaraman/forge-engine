"""Deterministic corpus analysis: indexing, diagnostics, statistics, conventions.

Nothing in this package calls an LLM.
"""

from .conventions import ConventionReport, analyze_conventions
from .diagnostics import FrontmatterReport, LinkReport, frontmatter_report, link_report
from .indexer import CorpusIndexer, build_spans, detect_changes
from .model import ChangeSet, CorpusIndex, IndexedFile, SourceChange
from .pipeline import IndexPipeline, IndexResult, load_store
from .stats import CorpusStats, compute_stats

__all__ = [
    "CorpusIndexer",
    "detect_changes",
    "build_spans",
    "CorpusIndex",
    "IndexedFile",
    "ChangeSet",
    "SourceChange",
    "IndexPipeline",
    "IndexResult",
    "load_store",
    "compute_stats",
    "CorpusStats",
    "frontmatter_report",
    "link_report",
    "FrontmatterReport",
    "LinkReport",
    "analyze_conventions",
    "ConventionReport",
]
