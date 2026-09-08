"""Cross-span deduplication of extracted proposals.

The extractor sends one span per call, so the model cannot know another span
already produced the same claim. Asking the prompt to deduplicate was a
category error; deduplication is deterministic and belongs in code, where it
also applies retroactively to proposals that already exist.
"""

from __future__ import annotations

import pytest
from forge.domain import (
    Derivation,
    EntityType,
    Proposal,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    SafetyClass,
)
from forge.proposals.dedup import (
    CLAIM_SIMILARITY,
    cluster_claims,
    cluster_concepts,
    find_duplicates,
    similarity,
)


def _proposal(ident, kind, label):
    return Proposal(
        id=ident,
        type=kind,
        safety=SafetyClass.MODEL_GENERATED,
        target_entity_type=(
            EntityType.CONCEPT if kind is ProposalType.NEW_CONCEPT else EntityType.CLAIM
        ),
        operation=ProposedOperation(action="create", target=label, after=label),
        reason="test",
        evidence_span_ids=("sp1",),
        provenance=Provenance(
            tier=ProvenanceTier.EXTRACTED_CLAIM,
            derivation=Derivation.MODEL,
            confidence=0.5,
            agent="test/1",
            model_id="m",
        ),
    )


def _concept(ident, name):
    return _proposal(ident, ProposalType.NEW_CONCEPT, name)


def _claim(ident, statement):
    return _proposal(ident, ProposalType.NEW_CLAIM, statement)


class TestConceptAliases:
    """Three alias pairs appeared in 14 concepts from one document."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Reranking", "Reranker"),
            ("Chunking", "Chunk"),
            ("embedding model", "embedding models"),
        ],
    )
    def test_inflections_of_one_concept_cluster(self, a, b):
        clusters = cluster_concepts([_concept("p1", a), _concept("p2", b)])
        assert len(clusters) == 1
        assert sorted(clusters[0].labels) == sorted([a, b])

    def test_distinct_concepts_do_not_cluster(self):
        assert cluster_concepts([_concept("p1", "Reranking"), _concept("p2", "Chunking")]) == []

    def test_the_fuller_label_is_suggested(self):
        """The longer phrasing is likelier to be the name a reader searches."""
        clusters = cluster_concepts([_concept("p1", "Reranker"), _concept("p2", "Reranking")])
        assert clusters[0].suggested == "Reranking"

    def test_claims_are_not_treated_as_concepts(self):
        assert cluster_concepts([_claim("p1", "some statement about things")]) == []


class TestClaimNearDuplicates:
    DUPES = (
        "Increasing top-k without checking relevance dilutes context quality.",
        "Increasing top-k in a RAG system can degrade answer quality by introducing "
        "irrelevant content that distracts the model.",
    )
    DISTINCT = (
        "Fixed-size chunking on structured documents is a mistake.",
        "FastAPI derives request validation from Python type hints.",
    )

    #: Two phrasings of one fact from the first real extraction run. At 0.423
    #: they are comfortably over the threshold.
    _CLOSE_A = (
        "Increasing top-k can worsen answers because more retrieved chunks add "
        "irrelevant content that dilutes the context."
    )
    _CLOSE_B = (
        "Increasing top-k without checking relevance dilutes the context and can "
        "worsen generation quality."
    )
    CLOSE_DUPES = (_CLOSE_A, _CLOSE_B)
    #: Opposites, scoring 0.604, higher than all but one true duplicate.
    #: Word overlap cannot tell a statement from its converse, and no threshold
    #: fixes that.
    OPPOSITES = (
        "Retrieval failures cannot be fixed by prompt engineering.",
        "Generation failures cannot be fixed by retrieval tuning.",
    )

    def test_a_real_observed_duplicate_pair_clusters(self):
        clusters = cluster_claims(
            [_claim("p1", self.CLOSE_DUPES[0]), _claim("p2", self.CLOSE_DUPES[1])]
        )
        assert len(clusters) == 1
        assert len(clusters[0].proposal_ids) == 2

    def test_a_distant_paraphrase_is_missed_and_that_is_the_accepted_cost(self):
        """0.35 keeps 4 of 8 known duplicates. This is one of the 4 it drops.

        Kept as a test rather than deleted: the miss is a known property of the
        operating point, not an accident, and a future change that catches this
        pair should have to notice it is also re-admitting the 67 false pairs
        that made the report unusable at 0.22.
        """
        assert similarity(*self.DUPES) < CLAIM_SIMILARITY
        assert cluster_claims([_claim("p1", self.DUPES[0]), _claim("p2", self.DUPES[1])]) == []

    def test_opposites_cluster_and_no_threshold_can_stop_them(self):
        """The ceiling on this measure, pinned so it cannot be forgotten."""
        assert similarity(*self.OPPOSITES) > similarity(*self.CLOSE_DUPES)
        clusters = cluster_claims(
            [_claim("p1", self.OPPOSITES[0]), _claim("p2", self.OPPOSITES[1])]
        )
        assert len(clusters) == 1, "if this ever stops clustering, the measure changed"

    def test_unrelated_claims_do_not_cluster(self):
        assert cluster_claims([_claim("p1", self.DISTINCT[0]), _claim("p2", self.DISTINCT[1])]) == []

    def test_the_threshold_sits_above_the_noise(self):
        """The populations overlap, so this guards the one direction that holds:
        plainly unrelated claims stay apart."""
        assert similarity(*self.CLOSE_DUPES) >= CLAIM_SIMILARITY
        assert similarity(*self.DISTINCT) < CLAIM_SIMILARITY

    def test_a_singleton_is_not_a_cluster(self):
        assert cluster_claims([_claim("p1", self.CLOSE_DUPES[0])]) == []

    def test_three_phrasings_of_one_fact_form_one_cluster(self):
        third = (
            "Increasing top-k adds irrelevant retrieved content that dilutes the "
            "context and can worsen the answer."
        )
        clusters = cluster_claims(
            [
                _claim("p1", self.CLOSE_DUPES[0]),
                _claim("p2", self.CLOSE_DUPES[1]),
                _claim("p3", third),
            ]
        )
        assert len(clusters) == 1
        assert len(clusters[0].proposal_ids) == 3


class TestNothingIsDecidedAutomatically:
    def test_dedup_makes_no_model_calls(self):
        from forge.llm.base import CALLS

        CALLS.reset()
        find_duplicates([_concept("p1", "Reranking"), _concept("p2", "Reranker")])
        assert CALLS.count == 0

    def test_it_reports_rather_than_merges(self):
        """A cluster names a survivor; choosing one stays a human decision."""
        clusters = find_duplicates([_concept("p1", "Reranking"), _concept("p2", "Reranker")])
        assert clusters[0].suggested
        assert len(clusters[0].proposal_ids) == 2
