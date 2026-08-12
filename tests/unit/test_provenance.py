"""Provenance rules — the floor rule and its companions.

These are enforced in the domain layer, so the tests construct objects directly
rather than going through a service. If it is possible to build a violating
object, it is possible to persist one.
"""

from __future__ import annotations

import pytest

from forge.domain import (
    Claim,
    ClaimLink,
    Derivation,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    LinkType,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    ProvenanceViolation,
    deterministic_provenance,
    floor_tier,
    validate_claim,
    violates_floor,
)


def inp(tier: ProvenanceTier, eid: str = "x") -> ProvenanceInput:
    return ProvenanceInput(entity_type=EntityType.SPAN, entity_id=eid, tier=tier)


class TestFloorRule:
    @pytest.mark.parametrize(
        "tiers,expected",
        [
            ([ProvenanceTier.SOURCE_FACT], ProvenanceTier.SOURCE_FACT),
            (
                [ProvenanceTier.SOURCE_FACT, ProvenanceTier.SYNTHESIS],
                ProvenanceTier.SYNTHESIS,
            ),
            (
                [ProvenanceTier.EXTRACTED_CLAIM, ProvenanceTier.MODEL_INFERENCE],
                ProvenanceTier.MODEL_INFERENCE,
            ),
        ],
    )
    def test_floor_is_the_weakest_input(self, tiers, expected):
        assert floor_tier(tiers) is expected

    def test_no_inputs_means_no_constraint(self):
        assert floor_tier([]) is None
        assert not violates_floor(ProvenanceTier.SOURCE_FACT, [])

    def test_synthesis_cannot_be_claimed_as_source_fact(self):
        """The central guarantee: generated content cannot launder itself."""
        with pytest.raises(ProvenanceViolation, match="floor violated"):
            Provenance(
                tier=ProvenanceTier.SOURCE_FACT,
                derivation=Derivation.DETERMINISTIC,
                agent="t",
                inputs=(inp(ProvenanceTier.SYNTHESIS),),
            )

    def test_inference_over_extraction_cannot_be_extraction(self):
        with pytest.raises(ProvenanceViolation):
            Provenance(
                tier=ProvenanceTier.EXTRACTED_CLAIM,
                derivation=Derivation.MODEL,
                agent="t",
                model_id="m",
                inputs=(inp(ProvenanceTier.MODEL_INFERENCE),),
            )

    def test_weaker_or_equal_is_allowed(self):
        p = Provenance(
            tier=ProvenanceTier.MODEL_INFERENCE,
            derivation=Derivation.MODEL,
            agent="t",
            model_id="m",
            inputs=(inp(ProvenanceTier.SOURCE_FACT),),
        )
        assert p.tier is ProvenanceTier.MODEL_INFERENCE

    def test_derive_helper_enforces_the_rule(self):
        parent = deterministic_provenance("t", ProvenanceTier.SYNTHESIS)
        with pytest.raises(ProvenanceViolation):
            parent.derive(
                tier=ProvenanceTier.SOURCE_FACT,
                agent="c",
                derivation=Derivation.DETERMINISTIC,
                entity_type=EntityType.CLAIM,
                entity_id="c1",
            )

    def test_derive_records_parent_as_input(self):
        parent = deterministic_provenance("t", ProvenanceTier.SOURCE_FACT)
        child = parent.derive(
            tier=ProvenanceTier.MODEL_INFERENCE,
            agent="c",
            derivation=Derivation.MODEL,
            entity_type=EntityType.SPAN,
            entity_id="s1",
            model_id="m",
        )
        assert child.inputs[0].entity_id == "s1"
        assert child.inputs[0].tier is ProvenanceTier.SOURCE_FACT


class TestDerivationRules:
    def test_model_derivation_requires_model_id(self):
        with pytest.raises(ProvenanceViolation, match="requires model_id"):
            Provenance(tier=ProvenanceTier.MODEL_INFERENCE, derivation=Derivation.MODEL, agent="t")

    def test_deterministic_must_not_carry_model_id(self):
        with pytest.raises(ProvenanceViolation, match="not model work"):
            Provenance(
                tier=ProvenanceTier.SOURCE_FACT,
                derivation=Derivation.DETERMINISTIC,
                agent="t",
                model_id="m",
            )

    def test_model_cannot_produce_source_fact(self):
        with pytest.raises(ProvenanceViolation, match="cannot produce SOURCE_FACT"):
            Provenance(
                tier=ProvenanceTier.SOURCE_FACT,
                derivation=Derivation.MODEL,
                agent="t",
                model_id="m",
            )

    def test_violation_is_not_a_pydantic_validation_error(self):
        """It must be catchable specifically, not buried in field errors."""
        from pydantic import ValidationError

        with pytest.raises(ProvenanceViolation):
            Provenance(tier=ProvenanceTier.SYNTHESIS, derivation=Derivation.MODEL, agent="t")
        try:
            Provenance(tier=ProvenanceTier.SYNTHESIS, derivation=Derivation.MODEL, agent="t")
        except Exception as exc:
            assert not isinstance(exc, ValidationError)


class TestEvidenceRequirement:
    def _claim(self, tier: ProvenanceTier) -> Claim:
        prov = (
            deterministic_provenance("t", tier)
            if tier is not ProvenanceTier.MODEL_INFERENCE
            else Provenance(tier=tier, derivation=Derivation.MODEL, agent="t", model_id="m")
        )
        return Claim(id="c1", statement="something", provenance=prov)

    @pytest.mark.parametrize(
        "tier",
        [
            ProvenanceTier.SOURCE_FACT,
            ProvenanceTier.EXTRACTED_CLAIM,
            ProvenanceTier.MODEL_INFERENCE,
            ProvenanceTier.SYNTHESIS,
        ],
    )
    def test_non_user_claims_require_evidence(self, tier):
        with pytest.raises(ProvenanceViolation, match="requires"):
            validate_claim(self._claim(tier), [])

    def test_user_assertion_needs_no_evidence(self):
        validate_claim(self._claim(ProvenanceTier.USER_ASSERTION), [])

    def test_evidence_satisfies_requirement(self):
        claim = self._claim(ProvenanceTier.EXTRACTED_CLAIM)
        ev = EvidenceLink(
            id="e1",
            claim_id="c1",
            span_id="s1",
            relation=EvidenceRelation.PARAPHRASES,
            provenance=deterministic_provenance("t", ProvenanceTier.EXTRACTED_CLAIM),
        )
        validate_claim(claim, [ev])

    def test_evidence_for_a_different_claim_does_not_count(self):
        claim = self._claim(ProvenanceTier.EXTRACTED_CLAIM)
        other = EvidenceLink(
            id="e1",
            claim_id="OTHER",
            span_id="s1",
            relation=EvidenceRelation.PARAPHRASES,
            provenance=deterministic_provenance("t", ProvenanceTier.EXTRACTED_CLAIM),
        )
        with pytest.raises(ProvenanceViolation):
            validate_claim(claim, [other])

    def test_model_cannot_assert_a_verbatim_quote(self):
        with pytest.raises(ProvenanceViolation, match="QUOTES"):
            EvidenceLink(
                id="e1",
                claim_id="c1",
                span_id="s1",
                relation=EvidenceRelation.QUOTES,
                provenance=Provenance(
                    tier=ProvenanceTier.EXTRACTED_CLAIM,
                    derivation=Derivation.MODEL,
                    agent="t",
                    model_id="m",
                ),
            )


class TestLinkTypeDiscipline:
    @pytest.mark.parametrize(
        "link_type", [LinkType.SUPPORTS, LinkType.CONTRADICTS, LinkType.REFINES, LinkType.PART_OF]
    )
    def test_semantic_links_cannot_be_deterministic(self, link_type):
        with pytest.raises(ProvenanceViolation, match="semantic"):
            ClaimLink(
                id="l1",
                from_id="a",
                to_id="b",
                type=link_type,
                provenance=deterministic_provenance("indexer"),
            )

    @pytest.mark.parametrize("link_type", [LinkType.MENTIONS, LinkType.DERIVED_FROM])
    def test_deterministic_links_are_allowed(self, link_type):
        link = ClaimLink(
            id="l1",
            from_id="a",
            to_id="b",
            type=link_type,
            provenance=deterministic_provenance("indexer"),
        )
        assert link.type is link_type

    def test_related_to_requires_a_score(self):
        with pytest.raises(ValueError, match="score"):
            ClaimLink(
                id="l1",
                from_id="a",
                to_id="b",
                type=LinkType.RELATED_TO,
                provenance=deterministic_provenance("indexer"),
            )

    def test_related_to_with_score_is_fine(self):
        link = ClaimLink(
            id="l1",
            from_id="a",
            to_id="b",
            type=LinkType.RELATED_TO,
            score=0.82,
            provenance=deterministic_provenance("indexer"),
        )
        assert link.score == 0.82

    def test_self_links_rejected(self):
        with pytest.raises(ValueError, match="self-link"):
            ClaimLink(
                id="l1",
                from_id="a",
                to_id="a",
                type=LinkType.MENTIONS,
                provenance=deterministic_provenance("indexer"),
            )
