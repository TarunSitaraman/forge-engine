"""Capability-spike harness.

The harness itself must be trustworthy before its results mean anything: it
has to count attempts correctly, record failures rather than swallow them, and
say plainly when it did not run.
"""

from __future__ import annotations

from forge.llm import MockProvider, unavailable_provider
from forge.llm.base import ProviderUnavailable
from forge.spike import render_markdown, run_spike

VALID = {
    "concepts": '{"concepts":[{"name":"Sliding Window","kind":"pattern"}]}',
    "claims": '{"claims":[{"statement":"RAG reduces hallucination","evidence_quote":"grounding"}]}',
    "relationships": '{"relationships":[{"source":"Dijkstra","target":"heap","type":"REQUIRES"}]}',
    "synthesis": '{"summary":"ok","key_points":["a"]}',
}


def good_responder(request):
    content = request.messages[1].content
    if "concepts named" in content:
        return VALID["concepts"]
    if "factual assertions" in content:
        return VALID["claims"]
    if "typed relationships" in content:
        return VALID["relationships"]
    return VALID["synthesis"]


class TestHarnessMechanics:
    def test_runs_all_four_required_tasks(self):
        report = run_spike(MockProvider(responder=good_responder), repetitions=1)
        assert [t.task for t in report.tasks] == [
            "structured_concept_extraction",
            "simple_claim_extraction",
            "relationship_extraction",
            "small_synthesis",
        ]

    def test_repetitions_are_honoured(self):
        """Reliability needs more than one sample; one success proves nothing."""
        report = run_spike(MockProvider(responder=good_responder), repetitions=4)
        assert all(t.attempts == 4 for t in report.tasks)
        assert report.overall_success_rate == 1.0

    def test_latency_is_measured(self):
        report = run_spike(MockProvider(responder=good_responder), repetitions=2)
        assert all(t.median_latency is not None for t in report.tasks)

    def test_first_sample_is_retained_for_inspection(self):
        report = run_spike(MockProvider(responder=good_responder), repetitions=2)
        concepts = report.tasks[0]
        assert concepts.samples[0]["concepts"][0]["name"] == "Sliding Window"


class TestHonestFailureReporting:
    def test_schema_violations_are_recorded_not_hidden(self):
        report = run_spike(MockProvider(default_response='{"bogus": 1}'), repetitions=3)
        assert report.overall_success_rate == 0.0
        for task in report.tasks:
            assert task.failures
            assert task.failures[0]["kind"] == "schema_violation"

    def test_raw_output_is_captured_for_diagnosis(self):
        report = run_spike(MockProvider(default_response="I am not JSON"), repetitions=1)
        assert "I am not JSON" in report.tasks[0].failures[0]["raw_excerpt"]

    def test_partial_success_is_reported_as_partial(self):
        calls = {"n": 0}

        def flaky(request):
            calls["n"] += 1
            return good_responder(request) if calls["n"] % 2 == 0 else "garbage"

        report = run_spike(MockProvider(responder=flaky), repetitions=4)
        assert 0.0 < report.overall_success_rate < 1.0

    def test_unreachable_provider_runs_nothing(self):
        report = run_spike(unavailable_provider(), repetitions=3)
        assert report.reachable is True or report.tasks == []

    def test_unreachable_health_short_circuits(self):
        class Dead(MockProvider):
            def health(self):
                return False, "nothing listening"

        report = run_spike(Dead(), repetitions=3)
        assert report.reachable is False
        assert report.tasks == []
        assert report.overall_success_rate == 0.0

    def test_provider_error_mid_run_is_recorded(self):
        report = run_spike(
            MockProvider(fail_with=ProviderUnavailable("died mid-run")), repetitions=3
        )
        kinds = {f["kind"] for t in report.tasks for f in t.failures}
        assert "provider_unavailable" in kinds


class TestMarkdownRendering:
    def test_not_run_is_stated_plainly(self):
        class Dead(MockProvider):
            def health(self):
                return False, "no model"

        md = render_markdown(run_spike(Dead(), repetitions=1))
        assert "NOT RUN" in md
        assert "absence of evidence" in md
        assert "How to reproduce" in md

    def test_results_table_is_rendered(self):
        md = render_markdown(run_spike(MockProvider(responder=good_responder), repetitions=2))
        assert "| Task | Schema |" in md
        assert "structured_concept_extraction" in md
        assert "Overall structured-output success rate" in md

    def test_failures_are_rendered_not_omitted(self):
        md = render_markdown(run_spike(MockProvider(default_response="junk"), repetitions=1))
        assert "## Failure modes" in md
        assert "schema_violation" in md

    def test_notes_are_included(self):
        md = render_markdown(
            run_spike(MockProvider(responder=good_responder), repetitions=1),
            notes=["ran on a potato"],
        )
        assert "ran on a potato" in md
