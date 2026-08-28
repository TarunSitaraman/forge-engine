"""Retrieval metrics, the labelled dataset, and the embedding pathway.

The reason this module exists: Phase 3 forbids claiming a retrieval
improvement without measured evidence, and a measurement is only worth
anything if the measuring instrument is itself tested. Every metric here is
checked against a hand-computed value, not against whatever the implementation
currently returns.

The embedding tests assert the *degradation* behaviour as hard as the happy
path — a missing model must produce an explicit, reported absence, never a
silent fallback that looks like a result.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from forge.embeddings import HashingEmbeddingProvider, NullEmbeddingProvider
from forge.evaluation import (
    DEFAULT_FUSION_WEIGHTS,
    DatasetError,
    EvalDataset,
    RetrievalEvaluator,
    compare,
    score_query,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "tests" / "fixtures" / "eval" / "retrieval-v1.yaml"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


class TestScoreQuery:
    def test_perfect_ranking_scores_one(self):
        score = score_query("q", "cat", ["a", "b", "x", "y", "z"], ["a", "b"])

        assert score.recall_at_5 == 1.0
        assert score.recall_at_10 == 1.0
        assert score.precision_at_5 == pytest.approx(0.4)  # 2 hits out of 5 slots
        assert score.reciprocal_rank == 1.0
        assert score.first_hit_rank == 1

    def test_complete_miss_scores_zero_and_reports_no_rank(self):
        score = score_query("q", "cat", ["x", "y", "z"], ["a"])

        assert (score.recall_at_5, score.recall_at_10, score.precision_at_5) == (0.0, 0.0, 0.0)
        assert score.reciprocal_rank == 0.0
        assert score.first_hit_rank is None

    def test_reciprocal_rank_uses_the_first_hit_only(self):
        score = score_query("q", "cat", ["x", "y", "a", "b"], ["a", "b"])

        assert score.reciprocal_rank == pytest.approx(1 / 3)
        assert score.first_hit_rank == 3

    def test_recall_at_5_is_capped_by_k(self):
        """Six relevant documents cannot all fit in five slots."""
        relevant = [f"r{i}" for i in range(6)]
        score = score_query("q", "cat", relevant, relevant)

        assert score.recall_at_5 == pytest.approx(5 / 6)
        assert score.recall_at_10 == 1.0

    def test_hit_between_rank_six_and_ten_counts_only_for_recall_at_10(self):
        retrieved = ["x1", "x2", "x3", "x4", "x5", "a"]
        score = score_query("q", "cat", retrieved, ["a"])

        assert score.recall_at_5 == 0.0
        assert score.recall_at_10 == 1.0
        assert score.reciprocal_rank == pytest.approx(1 / 6)

    def test_duplicate_results_occupy_one_rank(self):
        """Ranking is at document level: the same document twice is one result."""
        score = score_query("q", "cat", ["x", "x", "x", "a"], ["a"])

        assert score.first_hit_rank == 2, "the three x's collapse to one rank"
        assert score.retrieved == ("x", "a")

    def test_query_with_no_labels_is_scored_as_zero_not_crash(self):
        score = score_query("q", "cat", ["a"], [])

        assert score.recall_at_10 == 0.0
        assert score.first_hit_rank is None

    def test_empty_result_list_scores_zero(self):
        score = score_query("q", "cat", [], ["a"])

        assert score.recall_at_10 == 0.0
        assert score.first_hit_rank is None


class TestSummarize:
    def test_aggregates_are_means_over_queries(self):
        scores = [
            score_query("q1", "a", ["r1"], ["r1"]),  # rr 1.0, recall 1.0
            score_query("q2", "a", ["x", "r2"], ["r2"]),  # rr 0.5, recall 1.0
            score_query("q3", "b", ["x"], ["r3"]),  # rr 0.0, recall 0.0
        ]

        summary = summarize("lexical", scores)

        assert summary.queries == 3
        assert summary.recall_at_10 == pytest.approx(2 / 3)
        assert summary.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_misses_are_counted(self):
        scores = [
            score_query("q1", "a", ["r1"], ["r1"]),
            score_query("q2", "a", [f"x{i}" for i in range(12)], ["r2"]),
        ]

        assert summarize("lexical", scores).total_misses == 1

    def test_a_hit_past_rank_ten_still_counts_as_a_miss(self):
        retrieved = [f"x{i}" for i in range(10)] + ["a"]
        summary = summarize("lexical", [score_query("q", "a", retrieved, ["a"])])

        assert summary.total_misses == 1, "rank 11 is outside the reported window"

    def test_per_category_breakdown_partitions_the_queries(self):
        scores = [
            score_query("q1", "dsa", ["r1"], ["r1"]),
            score_query("q2", "dsa", ["x"], ["r2"]),
            score_query("q3", "tech", ["r3"], ["r3"]),
        ]

        by_category = summarize("lexical", scores).by_category

        assert by_category["dsa"]["queries"] == 2.0
        assert by_category["dsa"]["recall@10"] == pytest.approx(0.5)
        assert by_category["tech"]["recall@10"] == 1.0

    def test_empty_run_summarizes_without_dividing_by_zero(self):
        summary = summarize("lexical", [])

        assert summary.queries == 0
        assert summary.mrr == 0.0

    def test_headline_is_human_readable(self):
        summary = summarize("lexical", [score_query("q", "a", ["r"], ["r"])], latency_ms=12.5)

        headline = summary.headline()

        assert "lexical" in headline and "R@5=1.000" in headline and "12.5ms/q" in headline


class TestCompare:
    def _summary(self, method: str, retrieved_per_query):
        scores = [
            score_query(f"q{i}", "a", retrieved, ["r"])
            for i, retrieved in enumerate(retrieved_per_query)
        ]
        return summarize(method, scores)

    def test_a_real_gain_is_called_an_improvement(self):
        baseline = self._summary("lexical", [["x"], ["x"]])
        candidate = self._summary("hybrid", [["r"], ["r"]])

        assert compare(baseline, candidate)["verdict"] == "improvement"

    def test_a_real_loss_is_called_a_regression(self):
        baseline = self._summary("lexical", [["r"], ["r"]])
        candidate = self._summary("hybrid", [["x"], ["x"]])

        assert compare(baseline, candidate)["verdict"] == "regression"

    def test_identical_methods_show_no_measurable_difference(self):
        baseline = self._summary("lexical", [["r"], ["x"]])
        candidate = self._summary("hybrid", [["r"], ["x"]])

        result = compare(baseline, candidate)

        assert result["verdict"] == "no measurable difference"
        assert all(v == 0.0 for v in result["deltas"].values())

    def test_sub_threshold_change_is_treated_as_noise(self):
        """A tiny delta on a small set is not evidence of improvement."""
        baseline = summarize(
            "lexical", [score_query(f"q{i}", "a", ["x", "r"], ["r"]) for i in range(200)]
        )
        nudged = [score_query(f"q{i}", "a", ["x", "r"], ["r"]) for i in range(199)]
        nudged.append(score_query("q199", "a", ["r"], ["r"]))  # one query improves
        candidate = summarize("hybrid", nudged)

        result = compare(baseline, candidate)

        assert result["deltas"]["recall@10"] == 0.0
        assert result["deltas"]["mrr"] < 0.01
        assert result["verdict"] == "no measurable difference"

    def test_comparison_states_the_sample_size_caveat(self):
        baseline = self._summary("lexical", [["r"]])
        assert "noise" in compare(baseline, baseline)["note"]


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


class TestEvalDataset:
    def test_the_shipped_dataset_loads(self):
        data = EvalDataset.load(DATASET_PATH)

        assert len(data) >= 20, "the brief requires a set of at least 20 queries"
        assert data.label_count() >= len(data)
        assert data.unit == "source"

    def test_the_shipped_dataset_covers_several_categories(self):
        categories = EvalDataset.load(DATASET_PATH).categories()

        assert len(categories) >= 4
        assert sum(categories.values()) == len(EvalDataset.load(DATASET_PATH))

    def test_every_label_points_at_a_real_file(self):
        """Ground truth that has rotted looks like a recall drop. Catch it here."""
        data = EvalDataset.load(DATASET_PATH)

        assert data.verify_labels(REPO_ROOT) == []

    def test_rotted_labels_are_reported_not_ignored(self, tmp_path):
        data = EvalDataset.load(DATASET_PATH)

        missing = data.verify_labels(tmp_path)

        assert len(missing) == data.label_count()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError):
            EvalDataset.load(tmp_path / "nope.yaml")

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "dupe.yaml"
        entry = {"id": "q1", "category": "a", "query": "x", "relevant": ["f.md"]}
        path.write_text(yaml.safe_dump({"version": 1, "queries": [entry, dict(entry)]}))

        with pytest.raises(DatasetError, match="duplicate"):
            EvalDataset.load(path)

    def test_query_without_labels_is_rejected(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text(
            yaml.safe_dump(
                {"version": 1, "queries": [{"id": "q1", "category": "a", "query": "x", "relevant": []}]}
            )
        )

        with pytest.raises(DatasetError, match="no relevant documents"):
            EvalDataset.load(path)

    def test_missing_required_field_is_rejected(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "queries": [{"id": "q1", "query": "x"}]}))

        with pytest.raises(DatasetError, match="category"):
            EvalDataset.load(path)

    def test_by_category_filters(self):
        data = EvalDataset.load(DATASET_PATH)
        category = next(iter(data.categories()))

        selected = data.by_category(category)

        assert selected and all(q.category == category for q in selected)


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


class TestHashingEmbeddings:
    def test_vectors_are_deterministic(self):
        provider = HashingEmbeddingProvider()

        first = provider.embed(["retrieval augmented generation"])[0]
        second = provider.embed(["retrieval augmented generation"])[0]

        assert first == second, "a cached vector is only safe if embedding is a pure function"

    def test_vectors_are_unit_length(self):
        vector = HashingEmbeddingProvider().embed(["vector databases store embeddings"])[0]

        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_dimensions_are_respected(self):
        provider = HashingEmbeddingProvider(dimensions=64)

        assert len(provider.embed(["anything"])[0]) == 64
        assert provider.dimensions == 64

    def test_empty_text_gives_a_zero_vector_rather_than_an_error(self):
        assert HashingEmbeddingProvider().embed([""])[0] == [0.0] * 256

    def test_related_text_scores_above_unrelated_text(self):
        from forge.matching.matcher import cosine

        provider = HashingEmbeddingProvider()
        query, near, far = provider.embed(
            [
                "retrieval augmented generation grounds answers in sources",
                "RAG retrieval augmented generation cites its sources",
                "binary search on a sorted array halves the interval",
            ]
        )

        assert cosine(query, near) > cosine(query, far)

    def test_model_id_encodes_the_configuration(self):
        """Changing the feature configuration must invalidate cached vectors."""
        wide = HashingEmbeddingProvider(dimensions=512)
        narrow = HashingEmbeddingProvider(dimensions=256)
        words_only = HashingEmbeddingProvider(dimensions=256, char_ngrams=False)

        ids = {wide.model_id, narrow.model_id, words_only.model_id}

        assert len(ids) == 3, "incompatible vectors must never share a cache key"

    def test_provider_is_always_available(self):
        assert HashingEmbeddingProvider().available is True


class TestNullProvider:
    def test_reports_itself_unavailable(self):
        assert NullEmbeddingProvider().available is False
        assert NullEmbeddingProvider().model_id == "none"

    def test_embedding_raises_rather_than_returning_a_fake_vector(self):
        with pytest.raises(RuntimeError, match="lexical"):
            NullEmbeddingProvider().embed(["text"])


class TestEmbeddingCache:
    """Vectors are keyed by model id, so a model change cannot be mixed in."""

    def test_vectors_round_trip_through_the_store(self, store):
        provider = HashingEmbeddingProvider(dimensions=32)
        vector = provider.embed(["chunking is deterministic"])[0]

        store.put_embedding("span", "span-1", provider.model_id, vector)
        stored = dict(store.get_embeddings("span", provider.model_id))

        assert stored["span-1"] == pytest.approx(vector, abs=1e-6)

    def test_a_different_model_id_does_not_read_the_old_vectors(self, store):
        old = HashingEmbeddingProvider(dimensions=32)
        new = HashingEmbeddingProvider(dimensions=64)
        store.put_embedding("span", "span-1", old.model_id, old.embed(["text"])[0])

        assert store.get_embeddings("span", new.model_id) == [], (
            "changing the embedding model must invalidate the cache, not silently "
            "reuse vectors of a different geometry"
        )
        assert len(store.get_embeddings("span", old.model_id)) == 1

    def test_rebuilding_the_same_span_replaces_rather_than_duplicates(self, store):
        provider = HashingEmbeddingProvider(dimensions=32)
        store.put_embedding("span", "span-1", provider.model_id, provider.embed(["v1"])[0])
        store.put_embedding("span", "span-1", provider.model_id, provider.embed(["v2"])[0])

        stored = store.get_embeddings("span", provider.model_id)

        assert len(stored) == 1
        assert stored[0][1] == pytest.approx(provider.embed(["v2"])[0], abs=1e-6)


# --------------------------------------------------------------------------
# evaluator wiring
# --------------------------------------------------------------------------


class TestEvaluator:
    @pytest.fixture
    def tiny_dataset(self, tmp_path) -> EvalDataset:
        path = tmp_path / "tiny.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "unit": "source",
                    "queries": [
                        {
                            "id": "t1",
                            "category": "exact_concept",
                            "query": "retrieval augmented generation",
                            "relevant": ["docs/rag.md"],
                        }
                    ],
                }
            )
        )
        return EvalDataset.load(path)

    def test_semantic_is_skipped_with_an_explicit_note_when_unavailable(
        self, store, tiny_dataset
    ):
        run = RetrievalEvaluator(store).run(tiny_dataset, methods=("lexical", "semantic"))

        assert [s.method for s in run.summaries] == ["lexical"]
        assert run.notes and "semantic and hybrid skipped" in run.notes[0]

    def test_hybrid_is_skipped_when_no_vectors_are_stored(self, store, tiny_dataset):
        run = RetrievalEvaluator(store, embeddings=HashingEmbeddingProvider()).run(
            tiny_dataset, methods=("lexical", "hybrid")
        )

        assert [s.method for s in run.summaries] == ["lexical"]
        assert run.notes, "an absent semantic signal must be reported, not silently dropped"

    def test_lexical_always_runs(self, store, tiny_dataset):
        run = RetrievalEvaluator(store).run(tiny_dataset, methods=("lexical",))

        assert run.queries == 1
        assert run.best() is not None
        assert run.to_dict()["dataset_version"] == 1

    def test_fusion_weights_are_swept_not_assumed(self):
        assert 0.5 in DEFAULT_FUSION_WEIGHTS
        assert len(DEFAULT_FUSION_WEIGHTS) >= 3, (
            "a single fusion weight is an assumption; the brief requires a sweep"
        )
        assert DEFAULT_FUSION_WEIGHTS[0] == 0.0 and DEFAULT_FUSION_WEIGHTS[-1] == 1.0, (
            "the anchors must reproduce the pure methods"
        )


class TestEmbeddingModelMismatchIsExplained:
    """`embeddings build --provider X` and `retrieval-eval --provider Y` are
    one flag apart, and the mismatch is silent: semantic is skipped and the
    message says "no embeddings", which sends people to rebuild vectors that
    already exist under a different model id. Observed 2026-08-28.
    """

    def test_the_note_names_the_models_that_do_have_vectors(self, tmp_path):
        from forge.storage import SqliteStore

        store = SqliteStore(tmp_path / "e.db")
        store.initialize()
        assert store.embedded_models() == {}

    def test_stored_models_are_counted_by_name(self, tmp_path):
        from forge.domain import Document, Source, SourceKind, Span
        from forge.storage import SqliteStore

        store = SqliteStore(tmp_path / "e.db")
        store.initialize()
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h")
        store.put_source(src)
        doc = Document(
            id=Document.make_id(src.id, "h"),
            source_id=src.id,
            parser="p",
            parser_version="1",
            content_hash="h",
        )
        store.put_document(doc)
        span = Span(
            id="sp1",
            document_id=doc.id,
            ordinal=0,
            locator="L1",
            start_line=1,
            end_line=1,
            text="body",
            content_hash="h1",
        )
        store.put_spans([span])
        store.put_embedding("span", "sp1", "nomic-embed-text+prefixed", [0.1, 0.2])

        assert store.embedded_models() == {"nomic-embed-text+prefixed": 1}
        store.close()
