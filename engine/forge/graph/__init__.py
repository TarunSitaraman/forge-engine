"""Knowledge graph over SQLite: bounded traversal, measurement, integrity."""

from .graph import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_NODE_BUDGET,
    GRAPH_VERSION,
    SUPPORTED_GRAPH_TYPES,
    GraphMetrics,
    KnowledgeGraph,
    Neighbor,
    Path,
)
from .integrity import CODE_DESCRIPTIONS, Finding, IntegrityCode, IntegrityReport, check_integrity

__all__ = [
    "KnowledgeGraph",
    "GraphMetrics",
    "Neighbor",
    "Path",
    "check_integrity",
    "IntegrityReport",
    "IntegrityCode",
    "Finding",
    "CODE_DESCRIPTIONS",
    "SUPPORTED_GRAPH_TYPES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_NODE_BUDGET",
    "GRAPH_VERSION",
]
