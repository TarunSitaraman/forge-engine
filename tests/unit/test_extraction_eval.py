"""Extraction quality scoring.

Closes the gap `docs/research/` records twice: the `0.2.0` -> `0.3.0` prompt
rewrite was judged by reading fifty proposals by hand, and the reasoning-off
experiment had to run against the *assessment* set because no extraction set
existed — which is why its conclusion covered classification and said nothing
about extraction, while reasoning-off was silently governing a 5.66-hour run.
"""

from __future__ import annotations

import pytest

from forge.evaluation.extraction import (
    ExtractionCase,
    ExtractionDataset,
    ExtractionDatasetError,
    ExtractionReport,
    score_case,
)

CASE = ExtractionCase(
    id="c1",
    text="Redis offers RDB snapshots and AOF logging. The maxmemory directive caps the dataset.",
    expected=("Redis", "AOF"),
    forbidden=("maxmemory", "directive"),
)


class TestScoring:
    def test_a_perfect_extraction(self):
        score = score_case(CASE, ["Redis", "AOF"])
        assert score.recall == 1.0
        assert score.junk == []
        assert score.missed == []

    def test_a_miss_is_counted(self):
        score = score_case(CASE, ["Redis"])
        assert score.recall == 0.5
        assert score.missed == ["AOF"]

    def test_a_forbidden_concept_is_junk(self):
        score = score_case(CASE, ["Redis", "AOF", "maxmemory"])
        assert score.recall == 1.0
        assert score.junk == ["maxmemory"]

    def test_an_unlisted_concept_is_extra_not_junk(self):
        """Defensible-but-unlabelled output must not be punished as junk."""
        score = score_case(CASE, ["Redis", "AOF", "persistence"])
        assert score.extra == ["persistence"]
        assert score.junk == []

    def test_matching_normalizes_like_the_rest_of_the_engine(self):
        """`B-tree index` and `B Tree Index` are one concept, not a miss plus a hit."""
        case = ExtractionCase(id="c", text="t", expected=("B-tree index",), forbidden=())
        assert score_case(case, ["B Tree Index"]).recall == 1.0

    def test_blank_concepts_are_ignored(self):
        assert score_case(CASE, ["Redis", "", "   ", "AOF"]).recall == 1.0


class TestJunkRateIsTheHeadline:
    """Recall alone would rank a maximally greedy extractor best, which is the
    exact failure this corpus had: a 25-concept sample was ~35-40% usable."""

    def _report(self, concepts):
        report = ExtractionReport(model_id="m", prompt_version="p")
        report.scores.append(score_case(CASE, concepts))
        return report

    def test_a_careful_extractor_scores_well(self):
        report = self._report(["Redis", "AOF"])
        assert report.recall == 1.0
        assert report.junk_rate == 0.0

    def test_a_greedy_extractor_gets_full_recall_and_is_still_penalised(self):
        report = self._report(["Redis", "AOF", "maxmemory", "directive"])
        assert report.recall == 1.0, "recall alone cannot tell these apart"
        assert report.junk_rate == 0.5, "junk rate must"

    def test_the_two_are_distinguishable(self):
        careful = self._report(["Redis", "AOF"])
        greedy = self._report(["Redis", "AOF", "maxmemory", "directive"])
        assert careful.recall == greedy.recall
        assert careful.junk_rate < greedy.junk_rate


class TestGrounding:
    def test_a_verbatim_quote_is_grounded(self):
        score = score_case(CASE, [], [("s", "Redis offers RDB snapshots")])
        assert score.grounded_claims == 1

    def test_a_fabricated_quote_is_not(self):
        score = score_case(CASE, [], [("s", "Redis has no persistence whatsoever")])
        assert score.grounded_claims == 0

    def test_grounding_rate_is_one_when_no_claims_were_made(self):
        """Nothing asserted is nothing wrong — not a zero."""
        report = ExtractionReport(model_id="m", prompt_version="p")
        report.scores.append(score_case(CASE, ["Redis"]))
        assert report.grounding_rate == 1.0


class TestDataset:
    def test_the_shipped_set_loads(self):
        data = ExtractionDataset.load("tests/fixtures/eval/extraction-v1.yaml")
        assert len(data) >= 5
        assert all(c.expected and c.forbidden for c in data)

    def test_every_case_carries_a_trap(self):
        """A span with no plausible forbidden candidate cannot distinguish a
        careful extractor from a lucky one."""
        data = ExtractionDataset.load("tests/fixtures/eval/extraction-v1.yaml")
        for case in data:
            assert case.forbidden, f"{case.id} has no forbidden candidates"

    def test_expected_and_forbidden_never_overlap(self):
        from forge.parsing.links import normalize

        data = ExtractionDataset.load("tests/fixtures/eval/extraction-v1.yaml")
        for case in data:
            overlap = {normalize(e) for e in case.expected} & {
                normalize(f) for f in case.forbidden
            }
            assert not overlap, f"{case.id}: {overlap} is both required and forbidden"

    def test_a_missing_file_is_a_clean_error(self):
        with pytest.raises(ExtractionDatasetError):
            ExtractionDataset.load("tests/fixtures/eval/nope.yaml")


class TestRunnerUsesTheRealExtractor:
    def test_it_drives_candidate_extractor(self):
        """Score what ships, not a re-implementation of the prompt path."""
        from forge.evaluation.extraction import run
        from forge.extraction import CandidateExtractor
        from forge.llm import MockProvider

        data = ExtractionDataset.load("tests/fixtures/eval/extraction-v1.yaml")
        report = run(data, CandidateExtractor(MockProvider(default_response="{}"), max_spans=1))
        assert len(report.scores) == len(data)
        assert report.prompt_version.startswith("extract-prompts/")

    def test_a_provider_failure_becomes_a_scored_error_not_a_crash(self):
        from forge.evaluation.extraction import run

        class _Broken:
            version = "x"
            prompt_version = "p"
            schema_version = "s"

            def model_id(self):
                return "broken"

            def extract(self, spans):
                raise RuntimeError("provider exploded")

        data = ExtractionDataset.load("tests/fixtures/eval/extraction-v1.yaml")
        report = run(data, _Broken())
        assert all(s.error and "provider exploded" in s.error for s in report.scores)
