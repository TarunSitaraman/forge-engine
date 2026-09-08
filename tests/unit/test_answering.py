"""Vault-grounded answering: one model call, every statement cited.

The economic inversion the direction plan rests on. Answering over retrieved
spans costs one call; pre-extracting the corpus so a question *might* be
answerable was measured at 3,372 calls and 153 hours.
"""

from __future__ import annotations

from forge.answering import NOT_IN_VAULT, Answerer
from forge.answering.service import Answer
from forge.domain import Span
from forge.llm import MockProvider
from forge.retrieval.search import SearchHit


def _span(text, sid="sp1"):
    return Span(
        id=sid,
        document_id="d1",
        ordinal=0,
        locator="L1-L5",
        start_line=1,
        end_line=5,
        text=text,
        content_hash="h",
    )


class _FakeSearch:
    """Returns fixed hits, so answering is tested apart from retrieval."""

    def __init__(self, hits):
        self._hits = hits
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return list(self._hits)


def _hits(n):
    return [
        SearchHit(span=_span(f"passage {i} body", f"sp{i}"), document=None, source=None, score=1.0)
        for i in range(1, n + 1)
    ]


class TestRetrievalFailure:
    def test_no_hits_means_no_model_call(self):
        """A retrieval miss is not something a model should paper over."""
        from forge.llm.base import CALLS

        CALLS.reset()
        answer = Answerer(_FakeSearch([]), MockProvider(default_response="anything")).ask("q")
        assert answer.llm_calls == 0
        assert CALLS.count == 0
        assert answer.answered is False
        assert answer.text.startswith(NOT_IN_VAULT)

    def test_no_provider_still_reports_retrieval(self):
        answer = Answerer(_FakeSearch(_hits(3)), None).ask("q")
        assert answer.answered is False
        assert len(answer.passages) == 3
        assert answer.llm_calls == 0


class TestCitationVerification:
    def test_valid_citations_are_recorded_in_order(self):
        provider = MockProvider(default_response="Alpha [2]. Beta [1]. Gamma [2] again.")
        answer = Answerer(_FakeSearch(_hits(3)), provider).ask("q")
        assert answer.cited == [2, 1]
        assert answer.invalid_citations == []
        assert answer.grounded is True

    def test_a_citation_to_a_passage_never_supplied_is_a_defect(self):
        """The same discipline as extraction's quote grounding, for free."""
        provider = MockProvider(default_response="Claim [7].")
        answer = Answerer(_FakeSearch(_hits(3)), provider).ask("q")
        assert answer.invalid_citations == [7]
        assert answer.grounded is False

    def test_an_unsupplied_number_is_kept_out_of_cited(self):
        """The invariant the printed source list depends on.

        `sources()` drops any citation naming a passage nobody supplied, so it
        matches `cited` only while `cited` holds no such number. `Answerer`
        keeps that true by routing them to `invalid_citations` instead.
        """
        provider = MockProvider(default_response="A [2]. B [7]. C [3].")
        answer = Answerer(_FakeSearch(_hits(3)), provider).ask("q")

        assert answer.cited == [2, 3]
        assert answer.invalid_citations == [7]
        assert len(answer.sources()) == len(answer.cited)

    def test_each_number_is_paired_with_its_own_passage(self):
        """Not with whatever survived filtering at the same index.

        The CLI used to print its source list by zipping `cited` against
        `sources()`, which is correct only while the invariant above holds.
        Nothing enforced it. An Answer built any other way, from the API or
        restored from a store, slid every source up by one and printed a
        number against another passage's citation, which reads as verified.

        `cited_sources()` pairs at the point the numbers are known, so the
        list is right whatever `cited` contains. Turning on B905 is what
        pointed at the zip.
        """
        hits = _hits(3)
        answer = Answer(
            question="q",
            text="A [2]. B [7]. C [3].",
            passages=hits,
            cited=[2, 7, 3],  # as no Answerer would build it, and a restore might
            invalid_citations=[],
        )

        pairs = answer.cited_sources()
        assert [n for n, _ in pairs] == [2, 3], "7 names no passage, so it has no source"
        for n, citation in pairs:
            assert hits[n - 1].citation == citation
        assert answer.sources() == [c for _, c in pairs]

    def test_an_uncited_answer_is_not_grounded(self):
        provider = MockProvider(default_response="Some confident prose with no citation.")
        answer = Answerer(_FakeSearch(_hits(3)), provider).ask("q")
        assert answer.cited == []
        assert answer.grounded is False

    def test_not_in_vault_is_reported_as_unanswered(self):
        provider = MockProvider(default_response="NOT IN VAULT\nNothing covers this.")
        answer = Answerer(_FakeSearch(_hits(3)), provider).ask("q")
        assert answer.answered is False

    def test_one_question_costs_one_call(self):
        provider = MockProvider(default_response="Answer [1].")
        assert Answerer(_FakeSearch(_hits(5)), provider).ask("q").llm_calls == 1


class TestQueryConstruction:
    def test_engine_docs_are_excluded_by_default(self):
        search = _FakeSearch(_hits(1))
        Answerer(search, MockProvider(default_response="x [1]")).ask("q")
        assert search.last_query.exclude_sources == ("docs/",)

    def test_the_title_boost_is_off(self):
        """Changed 2026-09-06 from asserting 1.25, and the old value was the bug.

        The boost shipped on a sweep over 1,724 spans that showed R@10
        0.489 -> 0.510. Re-run on the post-split corpus, every boost is a
        regression: 1.25 costs 0.062 of R@10 and 0.200 of `fuzzy_concept`,
        the category a real question most often falls into.

        `ask()` and `RetrievalEvaluator._lexical` issue the identical
        `SearchService.search(SearchQuery(...))`, so that measurement is of
        this call and not of a neighbouring one.
        """
        search = _FakeSearch(_hits(1))
        Answerer(search, MockProvider(default_response="x [1]")).ask("q")
        assert search.last_query.title_boost == 1.0

    def test_the_boost_remains_available_to_callers(self):
        """Defaulted off, not deleted. A caller with a measurement may set it."""
        from forge.retrieval.search import SearchQuery

        assert SearchQuery(text="q", title_boost=1.5).title_boost == 1.5

    def test_answering_defaults_to_semantic_retrieval(self):
        """Measured 2026-08-28: lexical is the worst option on every metric.

        lexical R@10 0.588, semantic 0.733, hybrid(0.75) 0.774. Answering used
        to default to lexical-only, which the labelled set says is the weakest
        retriever available once real embeddings exist.
        """
        search = _FakeSearch(_hits(1))
        Answerer(search, MockProvider(default_response="x [1]")).ask("q")
        assert search.last_query.semantic is True
        assert search.last_query.semantic_weight == 0.75

    def test_semantic_can_still_be_turned_off(self):
        search = _FakeSearch(_hits(1))
        Answerer(search, MockProvider(default_response="x [1]")).ask("q", semantic=False)
        assert search.last_query.semantic is False
