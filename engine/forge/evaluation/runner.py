"""Retrieval evaluation runner.

Runs the labelled query set against one or more retrieval methods and reports
metrics. Its purpose is to make "is this better?" a measured question.

Three methods can be compared:

* ``lexical``  — FTS5/BM25. The baseline, and the default retrieval path.
* ``semantic`` — cosine over stored embeddings.
* ``hybrid``   — weighted fusion of the two, swept across several weights
  rather than assuming a 50/50 split.

Nothing here tunes retrieval. The set is for measuring; optimizing against it
would make the numbers meaningless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..embeddings.base import EmbeddingProvider, NullEmbeddingProvider
from ..logging import get_logger
from ..matching.matcher import cosine
from ..retrieval.search import SearchQuery, SearchService
from ..storage.sqlite_store import SqliteStore
from .dataset import EvalDataset, EvalQuery
from .metrics import MetricSummary, QueryScore, compare, score_query, summarize

log = get_logger(__name__)

#: Fusion weights swept for hybrid retrieval. `weight` is the share given to
#: the semantic score; the remainder goes to lexical. 0.0 and 1.0 are included
#: as sanity anchors — they must reproduce the pure methods.
DEFAULT_FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Documents retrieved per query before truncation to k.
RETRIEVAL_DEPTH = 30


@dataclass
class EvaluationRun:
    """Results of evaluating one or more methods."""

    dataset_version: int
    dataset_path: str
    queries: int
    summaries: list[MetricSummary] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def best(self) -> MetricSummary | None:
        return max(self.summaries, key=lambda s: (s.recall_at_10, s.mrr), default=None)

    def to_dict(self, *, include_scores: bool = False) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "dataset_path": self.dataset_path,
            "queries": self.queries,
            "summaries": [s.to_dict(include_scores=include_scores) for s in self.summaries],
            "comparisons": self.comparisons,
            "notes": self.notes,
        }


class RetrievalEvaluator:
    """Evaluates retrieval methods against the labelled set."""

    def __init__(
        self,
        store: SqliteStore,
        *,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings or NullEmbeddingProvider()
        self.search = SearchService(store, embeddings=self.embeddings)

    # -- entry point -------------------------------------------------------

    def run(
        self,
        dataset: EvalDataset,
        *,
        methods: Sequence[str] = ("lexical",),
        fusion_weights: Sequence[float] = DEFAULT_FUSION_WEIGHTS,
    ) -> EvaluationRun:
        run = EvaluationRun(
            dataset_version=dataset.version,
            dataset_path=str(dataset.path),
            queries=len(dataset),
        )

        semantic_ready = self._semantic_ready()
        if "semantic" in methods or "hybrid" in methods:
            if not semantic_ready:
                # Name what *is* stored. The build and eval commands take the
                # same --provider flag and must agree; when they do not, the
                # bare "no embeddings" message sends people to rebuild vectors
                # that already exist under a different model id.
                available = self.store.embedded_models()
                if available:
                    have = ", ".join(f"{m} ({n})" for m, n in sorted(available.items()))
                    run.notes.append(
                        f"semantic and hybrid skipped: no embeddings for model "
                        f"{self.embeddings.model_id!r}. Stored: {have}. "
                        f"Re-run with a matching --provider, or build vectors "
                        f"for this model."
                    )
                else:
                    run.notes.append(
                        "semantic and hybrid skipped: no embeddings stored at all. "
                        "Build them with `forge embeddings build --provider ollama`."
                    )

        if "lexical" in methods:
            run.summaries.append(self._evaluate(dataset, "lexical"))

        if "semantic" in methods and semantic_ready:
            run.summaries.append(self._evaluate(dataset, "semantic"))

        if "hybrid" in methods and semantic_ready:
            for weight in fusion_weights:
                # 0.0 and 1.0 duplicate the pure methods; they are kept as
                # anchors when running hybrid alone, and skipped otherwise.
                if weight in (0.0, 1.0) and len(methods) > 1:
                    continue
                run.summaries.append(
                    self._evaluate(dataset, f"hybrid(w={weight:g})", semantic_weight=weight)
                )

        baseline = next((s for s in run.summaries if s.method == "lexical"), None)
        if baseline is not None:
            for summary in run.summaries:
                if summary is not baseline:
                    run.comparisons.append(compare(baseline, summary))

        log.info(
            "retrieval_eval_complete",
            methods=[s.method for s in run.summaries],
            queries=len(dataset),
        )
        return run

    # -- per-method --------------------------------------------------------

    def _evaluate(
        self, dataset: EvalDataset, method: str, *, semantic_weight: float = 0.0
    ) -> MetricSummary:
        scores: list[QueryScore] = []
        started = time.perf_counter()

        for query in dataset:
            retrieved = self._retrieve(query, method, semantic_weight)
            scores.append(score_query(query.id, query.category, retrieved, query.relevant))

        elapsed_ms = (time.perf_counter() - started) * 1000 / max(1, len(dataset))
        return summarize(method, scores, latency_ms=elapsed_ms)

    def _retrieve(self, query: EvalQuery, method: str, semantic_weight: float) -> list[str]:
        """Return ranked source locators for one query."""
        if method == "lexical":
            return self._lexical(query.query)
        if method == "semantic":
            return self._semantic(query.query)
        return self._hybrid(query.query, semantic_weight)

    def _lexical(self, text: str) -> list[str]:
        hits = self.search.search(SearchQuery(text=text, limit=RETRIEVAL_DEPTH))
        return self._to_documents(((h.source.locator, h.score) for h in hits if h.source))

    def _semantic(self, text: str) -> list[str]:
        scored = self._semantic_scores(text)
        return self._to_documents(scored)

    def _hybrid(self, text: str, semantic_weight: float) -> list[str]:
        """Weighted fusion of normalized lexical and semantic scores.

        Scores are min-max normalized per method before fusion — BM25 and
        cosine are on incomparable scales, and blending them raw would let
        whichever happens to have the larger range dominate regardless of
        weight.
        """
        lexical_hits = self.search.search(SearchQuery(text=text, limit=RETRIEVAL_DEPTH))
        # Aggregate to the *best* span per document, matching `_to_documents`.
        #
        # Both of these were built with `dict(...)` over lists sorted by
        # descending score. `dict` keeps the last occurrence, so every document
        # collapsed to its **worst** span while the lexical and semantic
        # methods used its best. Measured 2026-08-28, that alone made hybrid
        # score below both of the signals it blends — impossible for a convex
        # combination, and the tell that the inputs were not what they claimed.
        lexical = _best_per_key((h.source.locator, h.score) for h in lexical_hits if h.source)
        semantic = _best_per_key(self._semantic_scores(text))

        lexical_norm = _normalize(lexical)
        semantic_norm = _normalize(semantic)

        fused: dict[str, float] = {}
        for locator in set(lexical_norm) | set(semantic_norm):
            fused[locator] = (1 - semantic_weight) * lexical_norm.get(locator, 0.0) + (
                semantic_weight * semantic_norm.get(locator, 0.0)
            )
        return self._to_documents(fused.items())

    def _semantic_scores(self, text: str) -> list[tuple[str, float]]:
        """Cosine similarity between the query and every stored span vector."""
        if not self.embeddings.available:
            return []
        try:
            # Must declare itself as a query: a model with asymmetric task
            # prefixes embeds questions and documents into different regions,
            # and the stored spans were embedded as documents.
            query_vector = self.embeddings.embed([text], task="query")[0]
        except TypeError:
            query_vector = self.embeddings.embed([text])[0]
        except Exception as exc:  # pragma: no cover - provider failure
            log.warning("semantic_eval_unavailable", error=str(exc)[:120])
            return []

        stored = self.store.get_embeddings("span", self.embeddings.model_id)
        scored: list[tuple[str, float]] = []
        for span_id, vector in stored:
            span = self.store.get_span(span_id)
            if span is None:
                continue
            document = self.store.get_document(span.document_id)
            source = self.store.get_source(document.source_id) if document else None
            if source is None:
                continue
            scored.append((source.locator, cosine(query_vector, vector)))

        scored.sort(key=lambda pair: -pair[1])
        return scored[: RETRIEVAL_DEPTH * 3]

    def _to_documents(self, scored: Any) -> list[str]:
        """Collapse span-level hits to a ranked list of distinct documents.

        A document's score is its best span. Ranking spans directly would let a
        finely-chunked document occupy several ranks and inflate precision.
        """
        best: dict[str, float] = {}
        for locator, score in scored:
            if locator not in best or score > best[locator]:
                best[locator] = score
        return [loc for loc, _ in sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))]

    def _semantic_ready(self) -> bool:
        return self.embeddings.available and bool(
            self.store.get_embeddings("span", self.embeddings.model_id)
        )


def _best_per_key(pairs: Any) -> dict[str, float]:
    """Collapse (key, score) pairs to the highest score per key.

    Never use `dict(pairs)` for this: it keeps the *last* pair, which on a
    descending-sorted list is the minimum.
    """
    best: dict[str, float] = {}
    for key, score in pairs:
        if key not in best or score > best[key]:
            best[key] = score
    return best


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}
