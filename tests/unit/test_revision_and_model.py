"""Revision records and canonical-model validation/serialization."""

from __future__ import annotations

import pytest

from forge.domain import (
    Claim,
    ClaimStatus,
    Concept,
    ConceptKind,
    Document,
    EntityType,
    ProvenanceTier,
    Revision,
    RevisionOp,
    Source,
    SourceKind,
    Span,
    TrustTier,
    deterministic_provenance,
    record_change,
    record_create,
    record_invalidate,
    record_supersede,
    validate_supersession,
)


class TestRevisionShapes:
    def test_create_requires_after_and_forbids_before(self):
        rev = record_create(EntityType.SOURCE, "s1", {"a": 1})
        assert rev.op is RevisionOp.CREATE and rev.before is None
        with pytest.raises(ValueError, match="must not carry"):
            Revision(
                entity_type=EntityType.SOURCE,
                entity_id="s1",
                op=RevisionOp.CREATE,
                before={"x": 1},
                after={"y": 2},
            )
        with pytest.raises(ValueError, match="must carry an `after`"):
            Revision(entity_type=EntityType.SOURCE, entity_id="s1", op=RevisionOp.CREATE)

    def test_change_requires_both_states(self):
        rev = record_change(EntityType.SOURCE, "s1", {"h": "a"}, {"h": "b"})
        assert rev.before != rev.after
        with pytest.raises(ValueError, match="requires both"):
            Revision(
                entity_type=EntityType.SOURCE,
                entity_id="s1",
                op=RevisionOp.CHANGE,
                after={"h": "b"},
            )

    def test_invalidate_requires_before(self):
        rev = record_invalidate(EntityType.SOURCE, "s1", {"h": "a"})
        assert rev.after is None
        with pytest.raises(ValueError, match="must carry the `before`"):
            Revision(
                entity_type=EntityType.SOURCE, entity_id="s1", op=RevisionOp.INVALIDATE
            )

    def test_supersede_retains_both_states(self):
        rev = record_supersede(
            EntityType.CLAIM, "c1", {"s": "old"}, {"s": "old", "status": "superseded"},
            superseded_by="c2",
        )
        assert rev.before is not None and rev.after is not None
        assert rev.cause == "c2"
        assert "c2" in (rev.note or "")

    def test_revisions_are_immutable(self):
        rev = record_create(EntityType.SOURCE, "s1", {"a": 1})
        with pytest.raises(Exception):
            rev.entity_id = "s2"  # type: ignore[misc]

    def test_revision_roundtrips(self):
        rev = record_create(EntityType.SOURCE, "s1", {"a": 1})
        assert Revision.model_validate_json(rev.model_dump_json()) == rev


class TestSupersessionValidation:
    def _claim(self, cid: str) -> Claim:
        return Claim(
            id=cid,
            statement=f"statement {cid}",
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
        )

    def test_valid_supersession(self):
        from forge.domain import utc_now

        old = self._claim("c1").model_copy(
            update={
                "status": ClaimStatus.SUPERSEDED,
                "superseded_by": "c2",
                "valid_to": utc_now(),
            }
        )
        validate_supersession(old, self._claim("c2"))

    def test_cannot_supersede_self(self):
        with pytest.raises(ValueError, match="itself"):
            validate_supersession(self._claim("c1"), self._claim("c1"))

    def test_old_must_be_marked_superseded(self):
        with pytest.raises(ValueError, match="must be marked SUPERSEDED"):
            validate_supersession(self._claim("c1"), self._claim("c2"))

    def test_claim_marked_superseded_needs_pointer(self):
        with pytest.raises(ValueError, match="superseded_by"):
            Claim(
                id="c1",
                statement="x",
                status=ClaimStatus.SUPERSEDED,
                provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
            )


class TestEntityValidation:
    def test_deterministic_ids_are_stable(self):
        a = Source.for_path("A/b.md", kind=SourceKind.MARKDOWN, content_hash="h")
        b = Source.for_path("A/b.md", kind=SourceKind.MARKDOWN, content_hash="different")
        assert a.id == b.id, "identity follows the path, not the content"

    def test_document_id_follows_content(self):
        assert Document.make_id("s1", "h1") != Document.make_id("s1", "h2")

    def test_concept_id_is_case_insensitive(self):
        assert Concept.make_id("Binary Search") == Concept.make_id("binary search")

    def test_span_rejects_inverted_lines(self):
        with pytest.raises(ValueError, match="precedes"):
            Span(
                id="x",
                document_id="d",
                ordinal=0,
                locator="L5-L1",
                start_line=5,
                end_line=1,
                text="t",
                content_hash="h",
            )

    def test_claim_rejects_empty_statement(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Claim(
                id="c",
                statement="   ",
                provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
            )

    def test_unknown_fields_rejected(self):
        """extra='forbid' catches typos that would otherwise vanish silently."""
        with pytest.raises(Exception):
            Source(
                id="x",
                kind=SourceKind.MARKDOWN,
                locator="a.md",
                content_hash="h",
                nonsense_field=1,  # type: ignore[call-arg]
            )

    def test_entities_roundtrip_through_json(self):
        concept = Concept(
            id=Concept.make_id("RAG"),
            canonical_name="RAG",
            kind=ConceptKind.TECHNOLOGY,
            aliases=("Retrieval-Augmented Generation",),
            vault_path="Technologies/Docs/rag.md",
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
        )
        assert Concept.model_validate_json(concept.model_dump_json()) == concept

    def test_source_defaults_to_unverified_trust(self):
        s = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h")
        assert s.trust_tier is TrustTier.UNVERIFIED
