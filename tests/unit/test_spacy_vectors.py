"""Tests for the spaCy static-vector provider.

spaCy is an optional extra and its model is a separate download, so anything
needing either skips. What is tested without them is the part that must hold
regardless: that an absent model is an absent provider rather than a crash.
"""

from __future__ import annotations

import pytest

from forge.embeddings.base import EmbeddingProvider
from forge.embeddings.spacy_vectors import SpacyEmbeddingProvider


class TestAbsenceIsNotAnError:
    """`available` is the contract every caller branches on. It must answer
    rather than raise, however broken the environment is."""

    def test_a_missing_model_reports_unavailable(self):
        provider = SpacyEmbeddingProvider("definitely-not-a-real-model")
        assert provider.available is False

    def test_a_missing_model_still_returns_vectors_of_a_usable_shape(self):
        """Callers embed before checking in some paths; a ragged return would
        corrupt the store rather than fail loudly."""
        provider = SpacyEmbeddingProvider("definitely-not-a-real-model")
        got = provider.embed(["a", "b"])
        assert len(got) == 2
        assert len({len(v) for v in got}) == 1

    def test_repeated_failure_does_not_retry_the_import(self):
        provider = SpacyEmbeddingProvider("definitely-not-a-real-model")
        assert provider.available is False
        assert provider.available is False
        assert provider._load_failed is True

    def test_it_satisfies_the_protocol(self):
        assert isinstance(SpacyEmbeddingProvider(), EmbeddingProvider)

    def test_the_model_id_names_the_model(self):
        assert SpacyEmbeddingProvider("en_core_web_md").model_id == "spacy-en_core_web_md"


class TestWithTheModelInstalled:
    @pytest.fixture(autouse=True)
    def _require_model(self):
        provider = SpacyEmbeddingProvider()
        if not provider.available:
            pytest.skip("en_core_web_md not installed")
        self.provider = provider

    def test_vectors_are_l2_normalized(self):
        """Cosine is a dot product everywhere downstream; an unnormalized
        vector silently rescales every similarity it takes part in."""
        (vector,) = self.provider.embed(["retrieval augmented generation"])
        assert sum(x * x for x in vector) == pytest.approx(1.0, abs=1e-5)

    def test_dimensions_match_the_vectors_returned(self):
        (vector,) = self.provider.embed(["anything"])
        assert len(vector) == self.provider.dimensions

    def test_synonymous_text_beats_unrelated_text(self):
        """The one property hashing cannot have, and the only reason this
        provider exists: similarity without shared vocabulary."""
        from forge.matching.matcher import cosine

        query, near, far = self.provider.embed(
            [
                "retrieval augmented generation",
                "grounding a language model in external documents",
                "kubernetes pod scheduling and node affinity",
            ]
        )
        assert cosine(query, near) > cosine(query, far)

    def test_empty_text_is_a_zero_vector_not_a_crash(self):
        (vector,) = self.provider.embed([""])
        assert len(vector) == self.provider.dimensions
