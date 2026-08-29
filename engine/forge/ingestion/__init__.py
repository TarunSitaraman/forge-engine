"""Phase 2 ingestion: acquisition -> spans -> provenance -> optional extraction."""

from .chunking import CHUNK_STRATEGY, build_spans, split_sentences
from .derivation import CacheStats, DerivationKey, embedding_key, extraction_key, parse_key
from .pipeline import PIPELINE_VERSION, IngestionPipeline, IngestOptions
from .plan import CALLS_PER_SPAN, ExtractionPlan, ExtractionPlanner, SourcePlan
from .report import IngestionReport, SourceReport

__all__ = [
    "IngestionPipeline",
    "IngestOptions",
    "IngestionReport",
    "SourceReport",
    "build_spans",
    "split_sentences",
    "CHUNK_STRATEGY",
    "DerivationKey",
    "CacheStats",
    "extraction_key",
    "embedding_key",
    "parse_key",
    "PIPELINE_VERSION",
    "ExtractionPlanner",
    "ExtractionPlan",
    "SourcePlan",
    "CALLS_PER_SPAN",
]
