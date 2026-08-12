"""Proposal activation, identity decisions, and relationship gating.

The property under test throughout: **approved evidence becomes persistent,
traceable knowledge without losing provenance or history — and without
guessing.**
"""

from __future__ import annotations

import pytest

from forge.activation import (
    ActivationOutcome,
    ProposalActivator,
    RelationshipActivator,
    RelationshipCandidate,
)
from forge.domain import (
    Claim,
    Concept,
    Derivation,
    Document,
    EntityType,
    EvidenceRelation,
    IdentityState,
    LinkType,
    Proposal,
    ProposalStatus,
    ProposalTransitionError,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    RevisionOp,
    SafetyClass,
    Source,
    SourceKind,
    Span,
    deterministic_provenance,
)
from forge.identity import CollisionResolution, ConceptIdentity, IdentityConfig, IdentityService


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def seeded(store):
    """A store with one source, document, and span to evidence against."""
    source = Source.for_path("papers/rag.pdf", kind=SourceKind.PDF, content_hash="h1")
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "h1"),
        source_id=source.id,
        parser="forge.pdf",
        parser_version="1",
        content_hash="h1",
    )
    store.put_document(document)
    span = Span(
        id=Span.make_id(document.id, 0, "p.1"),
        document_id=document.id,
        ordinal=0,
        locator="p.1 L1-L3",
        heading_path=("Retrieval Augmented Generation",),
        start_line=1,
        end_line=3,
        text="RAG grounds generation in retrieved passages, reducing hallucination.",
        content_hash="s1",
        page=1,
    )
    store.put_spans([span])
    return store, source, document, span


def model_provenance() -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent="CandidateExtractor",
        model_id="llama3.1:8b",
    )


def concept_proposal(span_id: str, name: str = "Retrieval Augmented Generation", **kw) -> Proposal:
    details = kw.pop("details", {"kind": "technology"})
    return Proposal(
        id=kw.pop("pid", "pc1"),
        type=ProposalType.NEW_CONCEPT,
        safety=kw.pop("safety", SafetyClass.MODEL_GENERATED),
        operation=ProposedOperation(action="create_concept", target=name, details=details),
        reason="extracted from source",
        evidence_span_ids=(span_id,),
        provenance=model_provenance(),
        **kw,
    )


def claim_proposal(span_id: str, **kw) -> Proposal:
    return Proposal(
        id=kw.pop("pid", "pl1"),
        type=ProposalType.NEW_CLAIM,
        safety=SafetyClass.MODEL_GENERATED,
        operation=ProposedOperation(
            action="create_claim",
            target="RAG grounds generation",
            after="RAG grounds generation in retrieved passages",
            details={
                "evidence_quote": "RAG grounds generation in retrieved passages",
                "concept": kw.pop("concept", ""),
            },
        ),
        reason="extracted with a verbatim quote",
        evidence_span_ids=(span_id,),
        provenance=model_provenance(),
        **kw,
    )


@pytest.fixture
def activator(store):
    return ProposalActivator(store, identity=IdentityService(IdentityConfig()))


# --------------------------------------------------------------------------


class TestConceptActivation:
    def test_approved_concept_becomes_canonical(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.CREATED
        concept = store.get_concept(result.entity_id)
        assert concept is not None
        assert concept.canonical_name == "Retrieval Augmented Generation"
        assert concept.kind.value == "technology"

    def test_concept_records_its_origin(self, seeded, activator):
        """Answers: which proposal created this concept?"""
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        concept = store.get_concept(result.entity_id)
        assert concept.origin_proposal_id == proposal.id
        assert concept.origin_span_ids == (span.id,)

    def test_activation_marks_the_proposal(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        stored = store.get_proposal(proposal.id)
        assert stored.status is ProposalStatus.ACTIVATED
        assert stored.activated_entity_id == result.entity_id
        assert stored.activated_entity_type is EntityType.CONCEPT
        assert stored.activated_at is not None

    def test_activation_preserves_the_human_decision(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve(by="alice", note="checked")
        store.put_proposal(proposal)
        activator.activate(proposal)

        stored = store.get_proposal(proposal.id)
        assert stored.decided_by == "alice"
        assert stored.decision_note == "checked"

    def test_concept_creation_writes_a_revision(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        revisions = store.revisions_for(EntityType.CONCEPT, result.entity_id)
        assert [r.op for r in revisions] == [RevisionOp.CREATE]

    def test_provenance_stays_model_derived(self, seeded, activator):
        """Approval is a decision, not a stronger warrant."""
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        concept = store.get_concept(result.entity_id)
        assert concept.provenance.derivation is Derivation.MODEL
        assert concept.provenance.model_id == "llama3.1:8b"


class TestClaimActivation:
    def test_approved_claim_becomes_canonical_with_evidence(self, seeded, activator):
        store, _, _, span = seeded
        proposal = claim_proposal(span.id).approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.CREATED
        assert result.evidence_links == 1
        claim = store.get_claim(result.entity_id)
        assert claim is not None
        evidence = store.evidence_for_claim(claim.id)
        assert len(evidence) == 1
        assert evidence[0].span_id == span.id

    def test_evidence_link_is_deterministically_provenanced(self, seeded, activator):
        """The claim is model-derived; the *link* was verified in code."""
        store, _, _, span = seeded
        proposal = claim_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        evidence = store.evidence_for_claim(result.entity_id)[0]
        assert evidence.relation is EvidenceRelation.QUOTES
        assert evidence.provenance.derivation is Derivation.DETERMINISTIC
        assert store.get_claim(result.entity_id).provenance.derivation is Derivation.MODEL

    def test_non_verbatim_quote_becomes_a_paraphrase(self, seeded, activator):
        store, _, _, span = seeded
        proposal = claim_proposal(span.id)
        proposal = proposal.model_copy(
            update={
                "operation": proposal.operation.model_copy(
                    update={"details": {"evidence_quote": "a loose restatement", "concept": ""}}
                )
            }
        ).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        assert store.evidence_for_claim(result.entity_id)[0].relation is EvidenceRelation.PARAPHRASES

    def test_claim_links_to_its_concept_when_one_exists(self, seeded, activator):
        store, _, _, span = seeded
        concept = concept_proposal(span.id).approve()
        store.put_proposal(concept)
        concept_result = activator.activate(concept)

        claim = claim_proposal(span.id, concept="Retrieval Augmented Generation").approve()
        store.put_proposal(claim)
        claim_result = activator.activate(claim)

        assert store.get_claim(claim_result.entity_id).subject_concept_id == concept_result.entity_id

    def test_claim_does_not_create_its_concept(self, seeded, activator):
        """Mentioning a concept is not approval to add it to the model."""
        store, _, _, span = seeded
        proposal = claim_proposal(span.id, concept="Never Approved Concept").approve()
        store.put_proposal(proposal)
        activator.activate(proposal)

        assert store.list_concepts() == []

    def test_claim_without_resolvable_evidence_is_refused(self, seeded, activator):
        store, _, _, _ = seeded
        proposal = claim_proposal("span-that-does-not-exist").approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.REFUSED
        assert "no evidence span resolves" in result.reason
        assert store.list_claims() == []


class TestIdempotency:
    def test_approve_activate_twice_creates_one_entity(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)

        first = activator.activate(proposal)
        second = activator.activate(store.get_proposal(proposal.id))

        assert first.outcome is ActivationOutcome.CREATED
        assert second.outcome is ActivationOutcome.ALREADY_ACTIVE
        assert len(store.list_concepts()) == 1

    def test_full_cycle_approve_activate_reindex_activate(self, seeded, activator):
        """The exact sequence the phase brief requires."""
        store, _, _, span = seeded
        concept = concept_proposal(span.id).approve()
        claim = claim_proposal(span.id, concept="Retrieval Augmented Generation").approve()
        store.put_proposal(concept)
        store.put_proposal(claim)

        activator.activate(concept)
        activator.activate(claim)
        baseline = store.counts()

        # "re-index": the same span is written again, as a re-ingest would.
        store.put_spans([span])

        activator.activate(store.get_proposal(concept.id))
        activator.activate(store.get_proposal(claim.id))

        after = store.counts()
        assert after["concepts"] == baseline["concepts"] == 1
        assert after["claims"] == baseline["claims"] == 1
        assert after["evidence_links"] == baseline["evidence_links"] == 1
        assert after["revisions"] == baseline["revisions"], "no duplicate revisions"

    def test_repeated_activation_writes_no_extra_revisions(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        result = activator.activate(proposal)

        for _ in range(3):
            activator.activate(store.get_proposal(proposal.id))

        revisions = store.revisions_for(EntityType.CONCEPT, result.entity_id)
        assert [r.op for r in revisions] == [RevisionOp.CREATE]

    def test_entity_identity_is_deterministic(self, seeded, activator, tmp_path):
        from forge.storage import SqliteStore

        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        first_id = activator.activate(proposal).entity_id

        other = SqliteStore(tmp_path / "other.db")
        other.initialize()
        other.put_source(store.list_sources()[0])
        for document in store.documents_for_source(store.list_sources()[0].id):
            other.put_document(document)
            other.put_spans(store.spans_for_document(document.id))
        other.put_proposal(proposal)
        second_id = ProposalActivator(
            other, identity=IdentityService(IdentityConfig())
        ).activate(proposal).entity_id
        other.close()

        assert first_id == second_id


class TestLifecycleAndFailure:
    def test_unapproved_proposal_is_refused(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id)
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.REFUSED
        assert "not approved" in result.reason
        assert store.list_concepts() == []

    def test_rejected_proposal_is_refused(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).reject()
        store.put_proposal(proposal)
        assert activator.activate(proposal).outcome is ActivationOutcome.REFUSED

    def test_superseded_proposal_is_refused(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).supersede("other")
        store.put_proposal(proposal)
        assert activator.activate(proposal).outcome is ActivationOutcome.REFUSED

    def test_metadata_repairs_are_not_activated(self, seeded, activator):
        store, _, _, _ = seeded
        proposal = Proposal(
            id="pm1",
            type=ProposalType.METADATA_REPAIR,
            safety=SafetyClass.DETERMINISTIC_VERIFIED,
            operation=ProposedOperation(action="replace_frontmatter_line", target="a.md"),
            reason="r",
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
        ).approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)
        assert result.outcome is ActivationOutcome.REFUSED
        assert "--apply" in result.reason

    def test_persistence_failure_is_reported_not_swallowed(self, seeded, activator, monkeypatch):
        """A failed write must never be reported as a successful activation."""
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)

        def explode(_concept):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(store, "put_concept", explode)
        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.FAILED
        assert "disk on fire" in result.reason
        # The proposal stays APPROVED so it can be retried.
        assert store.get_proposal(proposal.id).status is ProposalStatus.APPROVED

    def test_activated_status_requires_an_entity(self):
        proposal = concept_proposal("s1").approve()
        with pytest.raises(ValueError, match="records no activated_entity_id"):
            proposal.model_copy(update={"status": ProposalStatus.ACTIVATED})._check()

    def test_activated_is_terminal_except_supersede(self, seeded, activator):
        store, _, _, span = seeded
        proposal = concept_proposal(span.id).approve()
        store.put_proposal(proposal)
        activator.activate(proposal)
        activated = store.get_proposal(proposal.id)

        with pytest.raises(ProposalTransitionError):
            activated.approve()
        assert activated.supersede("other").status is ProposalStatus.SUPERSEDED

    def test_activate_approved_processes_the_queue(self, seeded, activator):
        store, _, _, span = seeded
        for i in range(3):
            store.put_proposal(
                concept_proposal(span.id, name=f"Concept {i}", pid=f"p{i}").approve()
            )

        report = activator.activate_approved()

        assert report.counts() == {"created": 3}
        assert len(store.list_concepts()) == 3


class TestAmbiguityProtection:
    def _ambiguous(self, span_id: str, name: str = "Heap") -> Proposal:
        return Proposal(
            id="pa1",
            type=ProposalType.CONCEPT_MATCH,
            safety=SafetyClass.AMBIGUOUS,
            operation=ProposedOperation(
                action="resolve_ambiguous_concept",
                target=name,
                details={
                    "match_kind": "ambiguous",
                    "candidates": [
                        {"canonical_name": name, "vault_path": "DSA/01_Patterns/Heap.md"},
                        {"canonical_name": name, "vault_path": "DSA/03_DataStructures/Heap.md"},
                    ],
                },
            ),
            reason="two canonical homes",
            evidence_span_ids=(span_id,),
            provenance=model_provenance(),
        )

    def _identity_with_heap(self, *, decided: str | None = None) -> IdentityService:
        config = IdentityConfig()
        config.record_collision(
            CollisionResolution(
                name="Heap",
                identities=(
                    ConceptIdentity("Heap", namespace="pattern", vault_path="DSA/01_Patterns/Heap.md"),
                    ConceptIdentity(
                        "Heap", namespace="data-structure", vault_path="DSA/03_DataStructures/Heap.md"
                    ),
                ),
                default=decided,
            )
        )
        return IdentityService(config)

    def test_approved_but_unresolved_ambiguity_is_refused(self, seeded, store):
        """Approving a proposal is not deciding which Heap was meant."""
        _, _, _, span = seeded
        activator = ProposalActivator(store, identity=self._identity_with_heap())
        proposal = self._ambiguous(span.id).approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.REFUSED
        assert "unresolved collision" in result.reason
        assert "forge identity decide" in result.reason
        assert store.list_concepts() == []

    def test_resolved_collision_activates_into_its_namespace(self, seeded, store):
        _, _, _, span = seeded
        activator = ProposalActivator(
            store, identity=self._identity_with_heap(decided="data-structure/Heap")
        )
        proposal = self._ambiguous(span.id).approve()
        store.put_proposal(proposal)

        result = activator.activate(proposal)

        assert result.outcome is ActivationOutcome.CREATED
        concept = store.get_concept(result.entity_id)
        assert concept.qualified_name == "data-structure/Heap"
        assert concept.namespace == "data-structure"

    def test_both_namespaced_heaps_can_coexist(self, store):
        """The point of resolving a collision is two concepts, not one winner."""
        provenance = deterministic_provenance("t", ProvenanceTier.USER_ASSERTION)
        for namespace in ("pattern", "data-structure"):
            store.put_concept(
                Concept(
                    id=Concept.make_id("Heap", namespace),
                    canonical_name="Heap",
                    namespace=namespace,
                    provenance=provenance,
                    origin_proposal_id="seed",
                )
            )
        assert {c.qualified_name for c in store.concepts_named("Heap")} == {
            "pattern/Heap",
            "data-structure/Heap",
        }

    @pytest.mark.parametrize("name", ["Heap", "Binary Search", "Trie"])
    def test_known_collisions_stay_protected_by_default(self, name):
        """No default resolutions ship. Silence means ambiguous."""
        service = IdentityService(IdentityConfig())
        assert service.resolve(name).state is IdentityState.NEW
        assert not service.resolve(name).decided


class TestIdentityConfig:
    def test_scaffold_documents_without_deciding(self):
        service = IdentityService(IdentityConfig())
        added, _ = service.scaffold(
            {"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]}
        )
        assert added == 1
        resolution = service.config.resolution_for("Heap")
        assert resolution is not None
        assert resolution.default is None, "scaffolding must not decide"
        assert service.resolve("Heap").state is IdentityState.AMBIGUOUS

    def test_scaffold_preserves_existing_decisions(self):
        service = IdentityService(IdentityConfig())
        index = {"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]}
        service.scaffold(index)
        service.decide("Heap", "pattern/Heap")

        added, skipped = service.scaffold(index)

        assert (added, skipped) == (0, 1)
        assert service.config.resolution_for("Heap").default == "pattern/Heap"

    def test_decide_changes_resolution(self):
        service = IdentityService(IdentityConfig())
        service.scaffold({"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]})
        service.decide("Heap", "data-structure/Heap", by="tarun")

        resolution = service.resolve("Heap")
        assert resolution.state is IdentityState.RESOLVED_BY_USER
        assert resolution.identity.qualified_name == "data-structure/Heap"

    def test_decide_rejects_an_unlisted_identity(self):
        service = IdentityService(IdentityConfig())
        service.scaffold({"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]})
        with pytest.raises(ValueError, match="not an identity"):
            service.decide("Heap", "invented/Heap")

    def test_clear_returns_to_undecided(self):
        service = IdentityService(IdentityConfig())
        service.scaffold({"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]})
        service.decide("Heap", "pattern/Heap")
        service.clear("Heap")
        assert service.resolve("Heap").state is IdentityState.AMBIGUOUS

    def test_config_round_trips(self, tmp_path):
        service = IdentityService(IdentityConfig())
        service.scaffold({"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]})
        service.decide("Heap", "pattern/Heap", by="tarun")
        path = service.config.save(tmp_path / "ci.yaml")

        reloaded = IdentityService(IdentityConfig.load(path))
        assert reloaded.resolve("Heap").identity.qualified_name == "pattern/Heap"

    def test_absent_config_is_valid(self, tmp_path):
        config = IdentityConfig.load(tmp_path / "missing.yaml")
        assert config.collisions == {}

    def test_malformed_default_is_rejected(self, tmp_path):
        from forge.identity import IdentityConfigError

        path = tmp_path / "bad.yaml"
        path.write_text(
            "version: 1\ncollisions:\n  - name: Heap\n    identities:\n"
            "      - canonical_name: Heap\n        namespace: pattern\n"
            "    default: nonsense/Heap\n"
        )
        with pytest.raises(IdentityConfigError, match="not one of its identities"):
            IdentityConfig.load(path)

    def test_matcher_learns_from_decisions(self):
        from forge.matching import ConceptMatcher

        index = {"heap": ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]}
        service = IdentityService(IdentityConfig())
        service.scaffold(index)

        before = ConceptMatcher([], ambiguity_index=index, identity=service).match("Heap")
        assert before.is_ambiguous and before.best is None

        service.decide("Heap", "data-structure/Heap")
        after = ConceptMatcher([], ambiguity_index=index, identity=service).match("Heap")

        assert after.identity_state is IdentityState.RESOLVED_BY_USER
        assert after.best.qualified_name == "data-structure/Heap"


class TestRelationshipGating:
    def _two_concepts(self, store):
        provenance = deterministic_provenance("t", ProvenanceTier.USER_ASSERTION)
        ids = []
        for name in ("Alpha Concept", "Beta Concept"):
            concept = Concept(
                id=Concept.make_id(name),
                canonical_name=name,
                provenance=provenance,
                origin_proposal_id="seed",
            )
            store.put_concept(concept)
            ids.append(concept.id)
        return ids

    def test_relationship_created_with_enough_evidence(self, seeded):
        store, _, _, span = seeded
        a, b = self._two_concepts(store)
        activator = RelationshipActivator(store, min_cooccurrence=1)

        report = activator.activate(
            [
                RelationshipCandidate(
                    from_concept_id=a,
                    to_concept_id=b,
                    type=LinkType.RELATED_TO,
                    span_ids=(span.id,),
                    score=0.5,
                    rationale="co-occurs",
                )
            ]
        )

        assert report.created == 1
        links = store.all_links()
        assert links[0].type is LinkType.RELATED_TO
        assert links[0].score == 0.5

    def test_insufficient_cooccurrence_is_rejected(self, seeded):
        store, _, _, span = seeded
        a, b = self._two_concepts(store)
        activator = RelationshipActivator(store, min_cooccurrence=2)

        report = activator.activate(
            [RelationshipCandidate(a, b, LinkType.RELATED_TO, (span.id,), 0.2, "one span")]
        )

        assert report.created == 0
        assert "at least 2" in report.rejected[0]["reason"]
        assert store.all_links() == []

    def test_no_evidence_is_rejected(self, store):
        a, b = self._two_concepts(store)
        report = RelationshipActivator(store).activate(
            [RelationshipCandidate(a, b, LinkType.RELATED_TO, (), None, "")]
        )
        assert report.created == 0
        assert "without evidence is a guess" in report.rejected[0]["reason"]

    def test_unknown_endpoint_is_rejected(self, seeded):
        store, _, _, span = seeded
        a, _ = self._two_concepts(store)
        report = RelationshipActivator(store, min_cooccurrence=1).activate(
            [RelationshipCandidate(a, "not-a-concept", LinkType.RELATED_TO, (span.id,), 0.5, "")]
        )
        assert "not a canonical concept" in report.rejected[0]["reason"]

    def test_unsupported_type_is_rejected(self, seeded):
        store, _, _, span = seeded
        a, b = self._two_concepts(store)
        report = RelationshipActivator(store, min_cooccurrence=1).activate(
            [RelationshipCandidate(a, b, LinkType.CONTRADICTS, (span.id,), 0.5, "")]
        )
        assert "outside the Phase 3 vocabulary" in report.rejected[0]["reason"]

    def test_self_relationship_is_rejected(self, seeded):
        store, _, _, span = seeded
        a, _ = self._two_concepts(store)
        report = RelationshipActivator(store, min_cooccurrence=1).activate(
            [RelationshipCandidate(a, a, LinkType.RELATED_TO, (span.id,), 0.5, "")]
        )
        assert "self-relationship" in report.rejected[0]["reason"]

    def test_relationship_activation_is_idempotent(self, seeded):
        store, _, _, span = seeded
        a, b = self._two_concepts(store)
        activator = RelationshipActivator(store, min_cooccurrence=1)
        candidate = RelationshipCandidate(a, b, LinkType.RELATED_TO, (span.id,), 0.5, "co-occurs")

        first = activator.activate([candidate])
        second = activator.activate([candidate])

        assert (first.created, second.created) == (1, 0)
        assert second.already_present == 1
        assert len(store.all_links()) == 1

    def test_discovery_finds_shared_spans(self, seeded):
        store, _, document, _ = seeded
        store.put_spans(
            [
                Span(
                    id=Span.make_id(document.id, i, f"p.{i}"),
                    document_id=document.id,
                    ordinal=i,
                    locator=f"p.{i}",
                    start_line=1,
                    end_line=2,
                    text="Alpha Concept and Beta Concept appear together here",
                    content_hash=f"c{i}",
                )
                for i in (1, 2)
            ]
        )
        self._two_concepts(store)

        candidates = RelationshipActivator(store).discover_cooccurrence()

        assert candidates
        assert len(candidates[0].span_ids) >= 2

    def test_discovery_is_deterministic(self, seeded):
        store, *_ = seeded
        self._two_concepts(store)
        activator = RelationshipActivator(store)
        first = [c.to_dict() for c in activator.discover_cooccurrence()]
        second = [c.to_dict() for c in activator.discover_cooccurrence()]
        assert first == second
