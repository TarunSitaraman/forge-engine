"""Storage interface behaviour and the guarantees it must enforce."""

from __future__ import annotations

import pytest

from forge.domain import (
    Claim,
    ClaimLink,
    ClaimStatus,
    Concept,
    ConceptKind,
    Document,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    LinkType,
    Provenance,
    ProvenanceTier,
    ProvenanceViolation,
    RevisionOp,
    Source,
    SourceKind,
    Span,
    TrustTier,
    Derivation,
    deterministic_provenance,
)
from forge.storage import SqliteStore, Store


def user_claim(cid: str, text: str = "a belief") -> Claim:
    return Claim(
        id=cid,
        statement=text,
        provenance=deterministic_provenance("test", ProvenanceTier.USER_ASSERTION),
    )


class TestProtocolConformance:
    def test_sqlite_store_satisfies_protocol(self, store):
        assert isinstance(store, Store)

    def test_reset_is_safe_and_idempotent(self, store):
        store.put_source(
            Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h")
        )
        store.reset()
        assert store.counts()["sources"] == 0
        store.reset()  # again — must not raise


class TestSourceLifecycle:
    def test_put_and_get(self, store):
        src = Source.for_path(
            "DSA/01_Patterns/DFS.md",
            kind=SourceKind.MARKDOWN,
            content_hash="h1",
            trust_tier=TrustTier.USER_AUTHORED,
        )
        store.put_source(src)
        assert store.get_source(src.id) == src
        assert store.get_source_by_locator("DSA/01_Patterns/DFS.md") == src

    def test_create_writes_a_create_revision(self, store):
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h1")
        store.put_source(src)
        revs = store.revisions_for(EntityType.SOURCE, src.id)
        assert [r.op for r in revs] == [RevisionOp.CREATE]

    def test_content_change_writes_a_change_revision(self, store):
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h1")
        store.put_source(src)
        store.put_source(src.model_copy(update={"content_hash": "h2"}))
        revs = store.revisions_for(EntityType.SOURCE, src.id)
        assert [r.op for r in revs] == [RevisionOp.CREATE, RevisionOp.CHANGE]
        assert revs[1].before["content_hash"] == "h1"
        assert revs[1].after["content_hash"] == "h2"

    def test_reput_identical_source_adds_no_revision(self, store):
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h1")
        store.put_source(src)
        store.put_source(src)
        assert len(store.revisions_for(EntityType.SOURCE, src.id)) == 1

    def test_delete_invalidates_and_preserves_history(self, store):
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h1")
        store.put_source(src)
        store.delete_source(src.id)
        assert store.get_source(src.id) is None
        revs = store.revisions_for(EntityType.SOURCE, src.id)
        assert revs[-1].op is RevisionOp.INVALIDATE
        assert revs[-1].before["content_hash"] == "h1", "history survives deletion"


class TestClaimStorage:
    def test_unevidenced_claim_is_rejected_at_the_storage_boundary(self, store):
        claim = Claim(
            id="c1",
            statement="X improves Y",
            provenance=Provenance(
                tier=ProvenanceTier.EXTRACTED_CLAIM,
                derivation=Derivation.MODEL,
                agent="extractor",
                model_id="m",
            ),
        )
        with pytest.raises(ProvenanceViolation):
            store.put_claim(claim)
        assert store.get_claim("c1") is None

    def test_evidenced_claim_is_stored_with_its_evidence(self, store):
        prov = Provenance(
            tier=ProvenanceTier.EXTRACTED_CLAIM,
            derivation=Derivation.MODEL,
            agent="extractor",
            model_id="m",
        )
        claim = Claim(id="c1", statement="X improves Y", provenance=prov)
        ev = EvidenceLink(
            id="e1",
            claim_id="c1",
            span_id="s1",
            relation=EvidenceRelation.PARAPHRASES,
            provenance=prov,
        )
        store.put_claim(claim, [ev])
        assert store.get_claim("c1") is not None
        assert [e.id for e in store.evidence_for_claim("c1")] == ["e1"]

    def test_user_assertion_needs_no_evidence(self, store):
        store.put_claim(user_claim("c1"))
        assert store.get_claim("c1") is not None


class TestSupersession:
    def test_old_claim_is_retained_not_deleted(self, store):
        store.put_claim(user_claim("c1", "old understanding"))
        store.supersede_claim("c1", user_claim("c2", "new understanding"))

        old = store.get_claim("c1")
        assert old is not None, "supersession must never delete"
        assert old.status is ClaimStatus.SUPERSEDED
        assert old.superseded_by == "c2"
        assert old.valid_to is not None
        assert old.statement == "old understanding", "prior content preserved verbatim"

    def test_supersession_records_both_states(self, store):
        store.put_claim(user_claim("c1", "old"))
        store.supersede_claim("c1", user_claim("c2", "new"), cause="src-42")
        revs = store.revisions_for(EntityType.CLAIM, "c1")
        supersede = [r for r in revs if r.op is RevisionOp.SUPERSEDE][0]
        assert supersede.before["status"] == "active"
        assert supersede.after["status"] == "superseded"
        assert supersede.cause == "src-42"

    def test_new_claim_is_active_and_recorded(self, store):
        store.put_claim(user_claim("c1"))
        store.supersede_claim("c1", user_claim("c2"))
        assert store.get_claim("c2").status is ClaimStatus.ACTIVE
        assert [r.op for r in store.revisions_for(EntityType.CLAIM, "c2")] == [RevisionOp.CREATE]

    def test_superseding_unknown_claim_raises(self, store):
        with pytest.raises(KeyError):
            store.supersede_claim("nope", user_claim("c2"))


class TestDocumentsSpansConceptsLinks:
    def test_documents_and_spans(self, store):
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
        spans = [
            Span(
                id=Span.make_id(doc.id, i, f"L{i}"),
                document_id=doc.id,
                ordinal=i,
                locator=f"L{i}",
                start_line=i,
                end_line=i + 1,
                text=f"text {i}",
                content_hash=f"h{i}",
            )
            for i in range(3)
        ]
        store.put_spans(spans)
        assert [s.ordinal for s in store.spans_for_document(doc.id)] == [0, 1, 2]
        assert store.get_span(spans[0].id).text == "text 0"

    def test_cascade_delete_removes_documents(self, store):
        src = Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h")
        store.put_source(src)
        store.put_document(
            Document(
                id=Document.make_id(src.id, "h"),
                source_id=src.id,
                parser="p",
                parser_version="1",
                content_hash="h",
            )
        )
        store.delete_source(src.id)
        assert store.documents_for_source(src.id) == []

    def test_concepts_by_name(self, store):
        c = Concept(
            id=Concept.make_id("DFS"),
            canonical_name="DFS",
            kind=ConceptKind.PATTERN,
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
        )
        store.put_concept(c)
        assert store.get_concept_by_name("DFS") == c
        assert [x.canonical_name for x in store.list_concepts()] == ["DFS"]

    def test_links_by_direction(self, store):
        link = ClaimLink(
            id=ClaimLink.make_id("a", "b", LinkType.MENTIONS),
            from_id="a",
            to_id="b",
            type=LinkType.MENTIONS,
            provenance=deterministic_provenance("indexer"),
        )
        store.put_link(link)
        assert [x.id for x in store.links_from("a")] == [link.id]
        assert [x.id for x in store.links_to("b")] == [link.id]
        assert store.links_from("b") == []


class TestRevisionOrdering:
    def test_revisions_are_totally_ordered(self, store):
        """Timestamps can collide; `seq` must still order them."""
        for i in range(20):
            store.put_source(
                Source.for_path(f"f{i}.md", kind=SourceKind.MARKDOWN, content_hash="h")
            )
        recent = store.recent_revisions(limit=20)
        assert len(recent) == 20
        assert store.count_revisions() == 20

    def test_rebuild_from_scratch_is_possible(self, tmp_path):
        """Derived state is disposable: deleting the DB loses nothing."""
        db = tmp_path / "s.db"
        s1 = SqliteStore(db)
        s1.initialize()
        s1.put_source(Source.for_path("a.md", kind=SourceKind.MARKDOWN, content_hash="h"))
        s1.close()
        db.unlink()
        s2 = SqliteStore(db)
        s2.initialize()
        assert s2.counts()["sources"] == 0
        s2.close()
