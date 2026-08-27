"""Vault-grounded answering: one model call, every statement cited.

The economic inversion the direction plan rests on. Answering over retrieved
spans costs one call; pre-extracting the corpus so a question *might* be
answerable was measured at 3,372 calls and 153 hours.
"""

from __future__ import annotations

import pytest

from forge.answering import NOT_IN_VAULT, Answerer
from forge.llm import MockProvider
from forge.retrieval.search import SearchHit
from forge.domain import Span


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

    def test_the_title_boost_is_applied(self):
        search = _FakeSearch(_hits(1))
        Answerer(search, MockProvider(default_response="x [1]")).ask("q")
        assert search.last_query.title_boost == 1.25
