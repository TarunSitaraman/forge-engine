"""The cost preview must be exact, free, and honest about what it cannot know.

Extraction is the only expensive operation in this engine — 49.0 s/call
measured on 8B-local, 3,372 calls to cover the vault. A preview that is wrong
is worse than no preview, because a number in a cost report gets believed.
"""

from __future__ import annotations

import pytest

from forge.extraction.extractor import CandidateExtractor
from forge.ingestion import (
    CALLS_PER_SPAN,
    CHUNK_STRATEGY,
    ExtractionPlanner,
    IngestionPipeline,
    IngestOptions,
    extraction_key,
)
from forge.llm.base import CALLS
from forge.storage import SqliteStore


@pytest.fixture
def planner(settings, tmp_path, scripted_extractor):
    store = SqliteStore(tmp_path / "plan.db")
    store.initialize()
    extractor = scripted_extractor()
    extractor.max_spans = 12
    pipeline = IngestionPipeline(settings, store, extractor=extractor)
    yield ExtractionPlanner(pipeline, extractor), pipeline, store, extractor
    store.close()


class TestPlanningIsFree:
    def test_planning_makes_no_model_call(self, planner, settings):
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        CALLS.reset()
        plan_er.plan(settings.vault_path)
        assert CALLS.count == 0

    def test_a_never_ingested_tree_is_unknown_not_zero(self, planner, settings):
        """The dangerous failure is reporting an unpriced corpus as free."""
        plan_er, _, _, _ = planner
        plan = plan_er.plan(settings.vault_path)
        assert plan.calls == 0
        assert plan.unknown, "uningested sources must be reported, not silently dropped"
        assert len(plan.pending) == 0


class TestPlanMatchesTheRun:
    def test_predicted_calls_equal_the_calls_the_run_actually_spends(
        self, planner, settings
    ):
        """The whole point. A preview that drifts from the run is a liability."""
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())

        predicted = plan_er.plan(settings.vault_path).calls
        assert predicted > 0

        CALLS.reset()
        report = pipeline.ingest_path(settings.vault_path, IngestOptions(extract=True))
        assert report.totals()["llm_calls"] == predicted

    def test_a_completed_run_prices_the_next_one_at_zero(self, planner, settings):
        """This is what makes extraction incremental: you pay for what changed."""
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions(extract=True))
        plan = plan_er.plan(settings.vault_path)
        assert plan.calls == 0
        assert plan.pending == []
        assert plan.cached

    def test_calls_per_span_is_two_in_the_real_extract_loop(self, planner, settings):
        """Pins the constant every estimate multiplies by.

        A third call added to `extract` without updating this would silently
        halve every cost report the command has ever printed.
        """
        plan_er, pipeline, store, extractor = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        source = next(s for s in store.list_sources())
        spans = [
            span
            for document in store.documents_for_source(source.id)
            for span in store.spans_for_document(document.id)
            if span.chunk_strategy == CHUNK_STRATEGY
        ]
        selected = extractor._select(spans)
        assert selected

        CALLS.reset()
        extractor.extract(spans)
        assert CALLS.count == len(selected) * CALLS_PER_SPAN


class TestPlanRespectsTheSameRules:
    def test_it_uses_the_ingestion_chunker_only(self, planner, settings):
        """`forge index` spans are a different chunking and must not be priced.

        Costing the wrong chunker is how 98 spans were once reported as 208,
        and 196 calls as 416.
        """
        from forge.corpus import IndexPipeline

        plan_er, pipeline, store, _ = planner
        IndexPipeline(settings, store).run(write_reports=False)
        pipeline.ingest_path(settings.vault_path, IngestOptions())

        predicted = plan_er.plan(settings.vault_path).calls
        CALLS.reset()
        report = pipeline.ingest_path(settings.vault_path, IngestOptions(extract=True))
        assert report.totals()["llm_calls"] == predicted

    def test_max_spans_caps_the_predicted_cost(self, planner, settings):
        plan_er, pipeline, store, extractor = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        wide = plan_er.plan(settings.vault_path).calls

        narrow_extractor = CandidateExtractor(extractor.provider, max_spans=1)
        narrow = ExtractionPlanner(pipeline, narrow_extractor).plan(settings.vault_path)
        assert narrow.calls <= wide
        assert all(s.calls <= CALLS_PER_SPAN for s in narrow.pending)

    def test_a_changed_file_becomes_pending_again(self, planner, settings, tmp_path):
        """Cost tracks edits, which is the incremental claim in one assertion."""
        plan_er, pipeline, store, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions(extract=True))
        assert plan_er.plan(settings.vault_path).calls == 0

        target = next(settings.vault_path.rglob("*.md"))
        target.write_text(
            target.read_text() + "\n\n## Added section\n\n"
            "A new paragraph long enough to be selected for extraction, "
            "carrying a real assertion about a real subject.\n"
        )
        pipeline.ingest_path(settings.vault_path, IngestOptions())

        plan = plan_er.plan(settings.vault_path)
        assert plan.calls > 0
        assert [s.locator for s in plan.pending] == [pipeline._locator(target)]


class TestContentHashSharing:
    """Two files with the same bytes share one cache entry, so the run pays once.

    Pricing them independently over-counted the fixture vault by 2 calls, and
    would over-count the real one further: six project packs contain an
    `01-overview.md` and `_index.md` recurs throughout.
    """

    def test_identical_files_are_priced_once(self, planner, settings):
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        plan = plan_er.plan(settings.vault_path)

        assert plan.duplicates, "the fixture vault holds byte-identical notes"
        assert all(s.calls == 0 for s in plan.duplicates)

    def test_a_duplicate_is_reported_not_hidden(self, planner, settings):
        """It costs nothing, but it is still in scope and a reader should see it."""
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        plan = plan_er.plan(settings.vault_path)
        assert all(d in plan.sources for d in plan.duplicates)
        assert plan.to_dict()["duplicate"] == len(plan.duplicates)


class TestEstimateHonesty:
    def test_no_rate_means_no_wall_clock_estimate(self, planner, settings):
        """Repo rule: no measurement claim without a measurement."""
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        plan = plan_er.plan(settings.vault_path)
        assert plan.seconds_per_call is None
        assert plan.estimated_hours is None

    def test_a_supplied_rate_is_used_as_given(self, planner, settings):
        plan_er, pipeline, _, _ = planner
        pipeline.ingest_path(settings.vault_path, IngestOptions())
        plan = plan_er.plan(settings.vault_path)
        plan.seconds_per_call = 49.0
        assert plan.estimated_hours == pytest.approx(plan.calls * 49.0 / 3600)
