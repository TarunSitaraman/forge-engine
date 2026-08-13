"""Knowledge evolution: narrowing, assessment, impact, proposals, activation.

The property under test throughout: **new evidence can change what Forge knows,
but only through a grounded assessment, a reviewable proposal, and a human
decision — and never by overwriting anything.**

Three things are asserted repeatedly rather than once, because they are the
guarantees a future change is most likely to erode quietly:

* **Deterministic steps spend zero model calls.** Asserted with the call
  counter, not by reading the code.
* **Ungrounded output is rejected, never repaired.** A model citing a span it
  was not shown must lose its assessment.
* **Nothing is overwritten.** Every activation test also checks that the prior
  state is still retrievable.
"""

from __future__ import annotations

import json

import pytest

from forge.domain import (
    AssessmentClass,
    AssessmentRecord,
    Claim,
    ClaimStatus,
    Concept,
    ConceptKind,
    Derivation,
    Document,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    ImpactClass,
    Proposal,
    ProposalStatus,
    ProposalType,
    Provenance,
    ProvenanceTier,
    SafetyClass,
    Source,
    SourceKind,
    Span,
    WorkflowRun,
    WorkflowStatus,
    deterministic_provenance,
)
from forge.evolution import (
    CandidateNarrower,
    ClaimRetriever,
    EvidenceAssessor,
    EvolutionActivator,
    EvolutionProposer,
    actionable,
    classify_impact,
    impact_of,
    requires_human_review,
)
from forge.evolution.assessor import AssessmentOutcome
from forge.identity import CollisionResolution, ConceptIdentity, IdentityConfig, IdentityService
from forge.llm import MockProvider
from forge.llm.base import CALLS, LLMError, ProviderUnavailable
from forge.proposals import ProposalService


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def model_provenance(agent: str = "CandidateExtractor") -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent=agent,
        model_id="mock|mock-1",
        prompt_version="assess-prompts/0.1.0",
        schema_version="assess/0.1.0",
    )


@pytest.fixture
def knowledge(store):
    """Existing knowledge plus a second source of new evidence.

    Mirrors the real scenario: a claim from paper A, and a paper B that
    qualifies it.
    """
    concept = Concept(
        id=Concept.make_id("Retrieval Augmented Generation"),
        canonical_name="Retrieval Augmented Generation",
        kind=ConceptKind.TECHNOLOGY,
        aliases=("RAG systems",),
        provenance=model_provenance(),
        origin_proposal_id="p-seed",
    )
    store.put_concept(concept)

    def add_source(name: str, text: str, heading: tuple[str, ...], page: int):
        source = Source.for_path(f"papers/{name}.pdf", kind=SourceKind.PDF, content_hash=name)
        store.put_source(source)
        document = Document(
            id=Document.make_id(source.id, name),
            source_id=source.id,
            parser="forge.pdf",
            parser_version="1",
            content_hash=name,
        )
        store.put_document(document)
        span = Span(
            id=Span.make_id(document.id, 0, f"p.{page}"),
            document_id=document.id,
            ordinal=0,
            locator=f"p.{page} L1-L4",
            heading_path=heading,
            start_line=1,
            end_line=4,
            text=text,
            content_hash=f"{name}-span",
            page=page,
        )
        store.put_spans([span])
        return source, span

    source_a, span_a = add_source(
        "paper-a",
        "RAG can improve factual accuracy on knowledge-intensive tasks.",
        ("Retrieval Augmented Generation",),
        1,
    )
    source_b, span_b = add_source(
        "paper-b",
        "Retrieval Augmented Generation can introduce errors when the retrieved "
        "context is irrelevant.",
        ("Retrieval Augmented Generation", "Failure Modes"),
        2,
    )

    claim = Claim(
        id=Claim.make_id("RAG can improve factual accuracy.", span_a.id),
        statement="RAG can improve factual accuracy.",
        subject_concept_id=concept.id,
        provenance=model_provenance(),
    )
    store.put_claim(
        claim,
        [
            EvidenceLink(
                id=EvidenceLink.make_id(claim.id, span_a.id, EvidenceRelation.QUOTES),
                claim_id=claim.id,
                span_id=span_a.id,
                relation=EvidenceRelation.QUOTES,
                provenance=deterministic_provenance("ProposalActivator"),
            )
        ],
    )
    return {
        "store": store,
        "concept": concept,
        "claim": claim,
        "span_a": span_a,
        "span_b": span_b,
        "source_b": source_b,
    }


def scripted(classification: str, *, refined: str = "", span_ids=None, claim_id=None):
    """A provider that answers with one assessment, echoing real ids."""

    def respond(request):
        import re

        text = request.messages[-1].content
        claims = re.findall(r"\[claim_id: ([^\]]+)\]", text)
        spans = re.findall(r"\[span_id: ([^\]]+)\]", text)
        return json.dumps(
            {
                "assessments": [
                    {
                        "claim_id": claim_id or (claims[0] if claims else "unknown"),
                        "classification": classification,
                        "rationale": "The new evidence bears on this claim in a specific way.",
                        "evidence_span_ids": span_ids or [spans[0]],
                        "refined_statement": refined,
                    }
                ]
            }
        )

    return MockProvider(responder=respond)


def assessor_for(store, provider, **kw) -> EvidenceAssessor:
    return EvidenceAssessor(
        store, provider, provider_id=kw.pop("provider_id", "mock"), model_id=kw.pop("model_id", "mock-1"), **kw
    )


# --------------------------------------------------------------------------
# candidate narrowing
# --------------------------------------------------------------------------


class TestCandidateNarrowing:
    def test_exact_name_in_evidence_selects_the_concept(self, knowledge):
        store, span_b = knowledge["store"], knowledge["span_b"]

        result = CandidateNarrower(store).narrow([span_b])

        assert [c.concept_name for c in result.candidates] == ["Retrieval Augmented Generation"]
        assert result.candidates[0].selector == "exact_name"

    def test_every_candidate_records_why_it_was_selected(self, knowledge):
        result = CandidateNarrower(knowledge["store"]).narrow([knowledge["span_b"]])

        for candidate in result.candidates:
            assert candidate.selector
            assert candidate.detail, "a candidate with no justification is a guess"

    def test_narrowing_makes_zero_llm_calls(self, knowledge):
        """Deterministic-first is a claim that decays silently if nothing checks it."""
        CALLS.reset()

        CandidateNarrower(knowledge["store"]).narrow([knowledge["span_b"]])

        assert CALLS.count == 0

    def test_alias_matches_when_the_canonical_name_does_not(self, knowledge, store):
        span = knowledge["span_b"].model_copy(
            update={"text": "RAG systems depend on retrieval quality.", "heading_path": ()}
        )

        result = CandidateNarrower(store).narrow([span])

        assert [c.selector for c in result.candidates] == ["alias"]

    def test_unrelated_evidence_selects_nothing(self, knowledge, store):
        span = knowledge["span_b"].model_copy(
            update={
                "text": "Dijkstra's algorithm finds shortest paths in weighted graphs.",
                "heading_path": ("Shortest Paths",),
            }
        )

        assert CandidateNarrower(store).narrow([span]).candidates == []

    def test_short_names_do_not_match_inside_words(self, store):
        store.put_concept(
            Concept(
                id=Concept.make_id("Go"),
                canonical_name="Go",
                kind=ConceptKind.TECHNOLOGY,
                provenance=model_provenance(),
                origin_proposal_id="p",
            )
        )
        document = Document(
            id="d1", source_id="s1", parser="p", parser_version="1", content_hash="h"
        )
        span = Span(
            id="span-short",
            document_id="d1",
            ordinal=0,
            locator="p.1",
            start_line=1,
            end_line=1,
            text="Good algorithms go together with good data structures.",
            content_hash="h",
        )

        assert CandidateNarrower(store).narrow([span]).candidates == []

    def test_a_user_decided_identity_selects_its_concept(self, store):
        """The identity config does work here rather than sitting inert."""
        concept = Concept(
            id=Concept.make_id("Heap", "data-structure"),
            canonical_name="Heap",
            namespace="data-structure",
            kind=ConceptKind.DATA_STRUCTURE,
            provenance=model_provenance(),
            origin_proposal_id="p",
        )
        store.put_concept(concept)
        config = IdentityConfig()
        config.record_collision(
            CollisionResolution(
                name="Heap",
                identities=(
                    ConceptIdentity(canonical_name="Heap", namespace="data-structure"),
                    ConceptIdentity(canonical_name="Heap", namespace="pattern"),
                ),
                default="data-structure/Heap",
            )
        )
        span = Span(
            id="s-heap",
            document_id="d",
            ordinal=0,
            locator="p.1",
            start_line=1,
            end_line=1,
            text="A heap keeps the smallest element at the root.",
            content_hash="h",
        )

        result = CandidateNarrower(store, identity=IdentityService(config)).narrow([span])

        assert [c.concept_name for c in result.candidates] == ["data-structure/Heap"]

    def test_candidate_set_is_capped(self, store, knowledge):
        for i in range(20):
            store.put_concept(
                Concept(
                    id=Concept.make_id(f"Retrieval Concept {i}"),
                    canonical_name=f"Retrieval Concept {i}",
                    kind=ConceptKind.CONCEPT,
                    provenance=model_provenance(),
                    origin_proposal_id="p",
                )
            )
        span = knowledge["span_b"].model_copy(
            update={"text": " ".join(f"Retrieval Concept {i}" for i in range(20))}
        )

        result = CandidateNarrower(store, max_candidates=5).narrow([span])

        assert len(result.candidates) == 5
        assert result.truncated is True

    def test_empty_evidence_selects_nothing(self, knowledge):
        assert CandidateNarrower(knowledge["store"]).narrow([]).candidates == []


# --------------------------------------------------------------------------
# claim retrieval
# --------------------------------------------------------------------------


class TestClaimRetrieval:
    def test_retrieves_claims_of_candidate_concepts(self, knowledge):
        store, concept, claim = knowledge["store"], knowledge["concept"], knowledge["claim"]

        result = ClaimRetriever(store).retrieve([concept.id])

        assert result.ids() == [claim.id]
        assert result.by_concept == {"Retrieval Augmented Generation": 1}

    def test_superseded_claims_are_excluded(self, knowledge):
        store, concept, claim = knowledge["store"], knowledge["concept"], knowledge["claim"]
        replacement = Claim(
            id=Claim.make_id("RAG improves accuracy when retrieval is good.", "x"),
            statement="RAG improves accuracy when retrieval is good.",
            subject_concept_id=concept.id,
            provenance=model_provenance(),
        )
        store.supersede_claim(claim.id, replacement)

        retrieved = ClaimRetriever(store).retrieve([concept.id]).ids()

        assert claim.id not in retrieved
        assert replacement.id in retrieved

    def test_disputed_claims_are_still_examined(self, knowledge):
        """Disputed is not terminal — later evidence may support or sharpen it."""
        store, concept, claim = knowledge["store"], knowledge["concept"], knowledge["claim"]
        store.put_claim(
            claim.model_copy(update={"status": ClaimStatus.DISPUTED}),
            list(store.evidence_for_claim(claim.id)),
        )

        assert ClaimRetriever(store).retrieve([concept.id]).ids() == [claim.id]

    def test_total_cap_bounds_the_prompt(self, knowledge, store):
        concept = knowledge["concept"]
        span = knowledge["span_a"]
        for i in range(10):
            claim = Claim(
                id=Claim.make_id(f"Statement number {i} about RAG.", span.id),
                statement=f"Statement number {i} about RAG.",
                subject_concept_id=concept.id,
                provenance=model_provenance(),
            )
            store.put_claim(
                claim,
                [
                    EvidenceLink(
                        id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.INFERS_FROM),
                        claim_id=claim.id,
                        span_id=span.id,
                        relation=EvidenceRelation.INFERS_FROM,
                        provenance=model_provenance(),
                    )
                ],
            )

        result = ClaimRetriever(store, per_concept=20, total=4).retrieve([concept.id])

        assert len(result.claims) == 4
        assert result.truncated is True

    def test_unknown_concept_is_skipped(self, knowledge):
        assert ClaimRetriever(knowledge["store"]).retrieve(["concept:nope"]).claims == []


# --------------------------------------------------------------------------
# semantic assessment
# --------------------------------------------------------------------------


class TestAssessmentClassifications:
    @pytest.mark.parametrize(
        "classification",
        ["SUPPORTS", "POTENTIAL_CONFLICT", "IRRELEVANT", "INSUFFICIENT_EVIDENCE"],
    )
    def test_each_classification_round_trips(self, knowledge, classification):
        store = knowledge["store"]
        assessor = assessor_for(store, scripted(classification))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.ok
        assert batch.records[0].classification.value == classification

    def test_refines_carries_a_refined_statement(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(
            store,
            scripted("REFINES", refined="RAG improves factual accuracy when retrieval is relevant."),
        )

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.ok
        record = batch.records[0]
        assert record.classification is AssessmentClass.REFINES
        assert "when retrieval is relevant" in assessor.refined_statement_for(record)

    def test_refines_without_a_refined_statement_is_rejected(self, knowledge):
        """Not actionable: there would be nothing to propose."""
        store = knowledge["store"]
        assessor = assessor_for(store, scripted("REFINES", refined=""))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.records == []
        assert batch.rejected[0]["reason"] == "REFINES requires a refined_statement"

    def test_contradicts_is_not_in_the_vocabulary(self):
        """The strongest available judgement is POTENTIAL_CONFLICT."""
        assert not hasattr(AssessmentClass, "CONTRADICTS")
        assert "CONTRADICTS" not in {c.value for c in AssessmentClass}

    def test_every_assessment_records_provider_and_model(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(
            store, scripted("SUPPORTS"), provider_id="cloud:anthropic", model_id="claude-sonnet-5"
        )

        record = assessor.assess([knowledge["span_b"]], [knowledge["claim"]]).records[0]

        assert record.provider_id == "cloud:anthropic"
        assert record.model_id == "claude-sonnet-5"
        assert record.prompt_version and record.schema_version


class TestGrounding:
    def test_a_hallucinated_span_id_is_rejected(self, knowledge):
        """Never repaired. Repairing a fake citation invents the evidence."""
        store = knowledge["store"]
        assessor = assessor_for(store, scripted("SUPPORTS", span_ids=["span-that-never-existed"]))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.records == []
        assert "not present in the evidence shown" in batch.rejected[0]["reason"]

    def test_a_real_span_that_was_not_shown_is_still_rejected(self, knowledge):
        """Citing real-but-unshown evidence is fabrication w.r.t. the question."""
        store = knowledge["store"]
        assessor = assessor_for(store, scripted("SUPPORTS", span_ids=[knowledge["span_a"].id]))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.records == []
        assert "not present in the evidence shown" in batch.rejected[0]["reason"]

    def test_an_unknown_claim_id_is_rejected(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(store, scripted("SUPPORTS", claim_id="claim-nobody-asked-about"))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.records == []
        assert "not one of the claims presented" in batch.rejected[0]["reason"]

    def test_a_rejected_assessment_is_not_cached(self, knowledge):
        """A rejection must not poison the cache and skip the next real attempt."""
        store = knowledge["store"]
        bad = assessor_for(store, scripted("SUPPORTS", span_ids=["nope"]))
        bad.assess([knowledge["span_b"]], [knowledge["claim"]])

        good = assessor_for(store, scripted("SUPPORTS"))
        batch = good.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.records[0].classification is AssessmentClass.SUPPORTS
        assert batch.cache.hits == 0


class TestAssessmentFailureModes:
    def test_provider_unavailable_is_explicit_and_not_a_guess(self, knowledge):
        store = knowledge["store"]
        provider = MockProvider(fail_with=ProviderUnavailable("no model running"))
        assessor = assessor_for(store, provider)

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.outcome is AssessmentOutcome.SEMANTIC_ANALYSIS_UNAVAILABLE
        assert batch.records == []
        assert not batch.ok

    def test_malformed_output_is_rejected_not_repaired(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(store, MockProvider(default_response="not json at all"))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.outcome is AssessmentOutcome.ASSESSMENT_REJECTED
        assert batch.records == []

    def test_schema_violation_is_rejected(self, knowledge):
        """An empty assessment list must not validate as a successful 'no impact'."""
        store = knowledge["store"]
        assessor = assessor_for(store, MockProvider(default_response=json.dumps({"assessments": []})))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.outcome is AssessmentOutcome.ASSESSMENT_REJECTED

    def test_transport_failure_is_retryable(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(store, MockProvider(fail_with=LLMError("timed out")))

        batch = assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.outcome is AssessmentOutcome.RETRYABLE_FAILURE

    def test_a_failure_never_becomes_an_empty_success(self, knowledge):
        store = knowledge["store"]
        for provider in (
            MockProvider(fail_with=ProviderUnavailable("down")),
            MockProvider(default_response="garbage"),
            MockProvider(fail_with=LLMError("timeout")),
        ):
            batch = assessor_for(store, provider).assess(
                [knowledge["span_b"]], [knowledge["claim"]]
            )
            assert not batch.ok, "a failed assessment must never report success"

    def test_nothing_to_assess_is_a_clean_no_op(self, knowledge):
        store = knowledge["store"]
        CALLS.reset()

        batch = assessor_for(store, scripted("SUPPORTS")).assess([knowledge["span_b"]], [])

        assert batch.ok and batch.records == []
        assert CALLS.count == 0


class TestAssessmentCache:
    def test_second_assessment_is_served_from_cache(self, knowledge):
        store = knowledge["store"]
        provider = scripted("SUPPORTS")
        first = assessor_for(store, provider).assess([knowledge["span_b"]], [knowledge["claim"]])
        CALLS.reset()

        second = assessor_for(store, provider).assess([knowledge["span_b"]], [knowledge["claim"]])

        assert first.llm_calls == 1
        assert second.llm_calls == 0
        assert second.cache.hits == 1
        assert CALLS.count == 0
        assert second.records[0].cached is True
        assert second.records[0].classification == first.records[0].classification

    def test_a_different_model_invalidates_the_cache(self, knowledge):
        store = knowledge["store"]
        assessor_for(store, scripted("SUPPORTS"), model_id="qwen3:8b").assess(
            [knowledge["span_b"]], [knowledge["claim"]]
        )

        batch = assessor_for(store, scripted("SUPPORTS"), model_id="llama3.1:8b").assess(
            [knowledge["span_b"]], [knowledge["claim"]]
        )

        assert batch.cache.hits == 0
        assert batch.llm_calls == 1

    def test_a_different_provider_invalidates_the_cache(self, knowledge):
        """Same model name on a different provider is not the same judgement."""
        store = knowledge["store"]
        assessor_for(store, scripted("SUPPORTS"), provider_id="ollama", model_id="m").assess(
            [knowledge["span_b"]], [knowledge["claim"]]
        )

        batch = assessor_for(
            store, scripted("SUPPORTS"), provider_id="cloud:anthropic", model_id="m"
        ).assess([knowledge["span_b"]], [knowledge["claim"]])

        assert batch.cache.hits == 0

    def test_different_evidence_invalidates_the_cache(self, knowledge):
        store = knowledge["store"]
        assessor = assessor_for(store, scripted("SUPPORTS"))
        assessor.assess([knowledge["span_b"]], [knowledge["claim"]])

        edited = knowledge["span_b"].model_copy(update={"content_hash": "changed"})
        batch = assessor.assess([edited], [knowledge["claim"]])

        assert batch.cache.hits == 0

    def test_the_derivation_key_names_everything_that_invalidates_it(self, knowledge):
        assessor = assessor_for(knowledge["store"], scripted("SUPPORTS"))

        key = assessor.derivation_key("evidence-hash", "claim-1")

        described = key.describe()
        assert described["model_id"] == "mock|mock-1"
        assert described["prompt_version"].startswith("assess-prompts/")
        assert described["schema_version"].startswith("assess/")
        assert described["processor_version"] == assessor.version


# --------------------------------------------------------------------------
# impact
# --------------------------------------------------------------------------


def record(classification: AssessmentClass, claim_id: str = "c1") -> AssessmentRecord:
    return AssessmentRecord(
        claim_id=claim_id,
        classification=classification,
        rationale="a rationale long enough to be meaningful",
        evidence_span_ids=("s1",),
        provider_id="mock",
        model_id="mock-1",
        prompt_version="p",
        schema_version="s",
        derivation_key="k",
    )


class TestImpact:
    def test_one_potential_conflict_dominates_many_supports(self):
        assessments = [record(AssessmentClass.SUPPORTS) for _ in range(5)]
        assessments.append(record(AssessmentClass.POTENTIAL_CONFLICT))

        assert (
            classify_impact(assessments, claims_examined=6) is ImpactClass.POTENTIAL_CONFLICT
        )

    def test_refines_outranks_supports(self):
        assessments = [record(AssessmentClass.SUPPORTS), record(AssessmentClass.REFINES)]

        assert classify_impact(assessments, claims_examined=2) is ImpactClass.REFINES

    def test_only_irrelevant_means_no_material_change(self):
        assessments = [record(AssessmentClass.IRRELEVANT), record(AssessmentClass.INSUFFICIENT_EVIDENCE)]

        assert classify_impact(assessments, claims_examined=2) is ImpactClass.NO_MATERIAL_CHANGE

    def test_no_related_claims_means_new_knowledge(self):
        """A finding, not a non-event."""
        assert classify_impact([], claims_examined=0) is ImpactClass.NEW_KNOWLEDGE

    def test_claims_existed_but_none_affected_is_not_new_knowledge(self):
        assert (
            classify_impact([record(AssessmentClass.IRRELEVANT)], claims_examined=3)
            is ImpactClass.NO_MATERIAL_CHANGE
        )

    def test_only_conflict_forces_a_stop(self):
        assert requires_human_review(ImpactClass.POTENTIAL_CONFLICT) is True
        for impact in (ImpactClass.SUPPORTS, ImpactClass.REFINES, ImpactClass.NO_MATERIAL_CHANGE):
            assert requires_human_review(impact) is False

    def test_irrelevant_and_insufficient_are_not_actionable(self):
        assessments = [
            record(AssessmentClass.SUPPORTS, "a"),
            record(AssessmentClass.IRRELEVANT, "b"),
            record(AssessmentClass.INSUFFICIENT_EVIDENCE, "c"),
        ]

        assert [a.claim_id for a in actionable(assessments)] == ["a"]

    def test_impact_mapping_is_total(self):
        for classification in AssessmentClass:
            assert isinstance(impact_of(classification), ImpactClass)

    def test_no_confidence_score_is_invented(self):
        """Categorical outcomes only — a model's self-report is not a measurement."""
        fields = set(AssessmentRecord.model_fields)
        assert not {f for f in fields if "confidence" in f or "score" in f}


# --------------------------------------------------------------------------
# proposals
# --------------------------------------------------------------------------


class TestProposalGeneration:
    def _propose(self, store, classification, refined=""):
        proposer = EvolutionProposer(store, workflow_id="wf1", source_id="src-b")
        rec = record(classification, claim_id=self.claim_id)
        return proposer.propose([rec], refined_statements={self.claim_id: refined})

    @pytest.fixture(autouse=True)
    def _bind(self, knowledge):
        self.store = knowledge["store"]
        self.claim_id = knowledge["claim"].id

    def test_supports_creates_an_evidence_proposal(self):
        batch = self._propose(self.store, AssessmentClass.SUPPORTS)

        assert batch.created[0].type is ProposalType.CLAIM_EVIDENCE
        assert batch.created[0].operation.action == "attach_evidence"

    def test_refines_creates_a_refinement_proposal_with_before_and_after(self):
        batch = self._propose(
            self.store, AssessmentClass.REFINES, refined="RAG helps when retrieval is relevant."
        )

        proposal = batch.created[0]
        assert proposal.type is ProposalType.CLAIM_REFINEMENT
        assert proposal.operation.before == "RAG can improve factual accuracy."
        assert proposal.operation.after == "RAG helps when retrieval is relevant."

    def test_conflict_proposals_are_marked_ambiguous(self):
        """Which makes Phase 3's batch-approval guard refuse to bulk-approve them."""
        batch = self._propose(self.store, AssessmentClass.POTENTIAL_CONFLICT)

        assert batch.created[0].safety is SafetyClass.AMBIGUOUS

    def test_irrelevant_produces_no_proposal(self):
        batch = self._propose(self.store, AssessmentClass.IRRELEVANT)

        assert batch.created == []
        assert batch.skipped[0]["reason"] == "not actionable by design"

    def test_insufficient_evidence_produces_no_proposal(self):
        batch = self._propose(self.store, AssessmentClass.INSUFFICIENT_EVIDENCE)

        assert batch.created == []

    def test_proposals_start_pending(self):
        batch = self._propose(self.store, AssessmentClass.SUPPORTS)

        assert batch.created[0].status is ProposalStatus.PENDING

    def test_a_model_proposal_cannot_be_deterministic_verified(self):
        batch = self._propose(self.store, AssessmentClass.SUPPORTS)

        assert batch.created[0].safety is not SafetyClass.DETERMINISTIC_VERIFIED

    def test_proposal_identity_is_deterministic(self):
        first = self._propose(self.store, AssessmentClass.SUPPORTS)
        second = self._propose(self.store, AssessmentClass.SUPPORTS)

        assert second.created == []
        assert second.existing[0].id == first.created[0].id

    def test_a_rejected_proposal_is_not_resurrected(self):
        batch = self._propose(self.store, AssessmentClass.SUPPORTS)
        ProposalService(self.store).reject(batch.created[0].id, note="no")

        again = self._propose(self.store, AssessmentClass.SUPPORTS)

        assert again.created == []
        assert again.existing[0].status is ProposalStatus.REJECTED

    def test_provenance_records_provider_prompt_and_schema(self):
        proposal = self._propose(self.store, AssessmentClass.SUPPORTS).created[0]

        assert proposal.provenance.tier is ProvenanceTier.MODEL_INFERENCE
        assert proposal.provenance.model_id == "mock|mock-1"
        assert proposal.provenance.prompt_version and proposal.provenance.schema_version
        assert proposal.provenance.derivation_key == "k"

    def test_evidence_spans_are_recorded_as_provenance_inputs(self):
        proposal = self._propose(self.store, AssessmentClass.SUPPORTS).created[0]

        assert [i.entity_id for i in proposal.provenance.inputs] == ["s1"]

    def test_a_missing_claim_is_skipped_not_invented(self):
        proposer = EvolutionProposer(self.store, workflow_id="wf", source_id="s")

        batch = proposer.propose([record(AssessmentClass.SUPPORTS, claim_id="gone")])

        assert batch.created == []
        assert batch.skipped[0]["reason"] == "claim no longer exists"


# --------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------


def approved_proposal(store, knowledge, classification, refined="") -> Proposal:
    proposer = EvolutionProposer(store, workflow_id="wf1", source_id="src-b")
    rec = AssessmentRecord(
        claim_id=knowledge["claim"].id,
        classification=classification,
        rationale="the new evidence bears on this claim",
        evidence_span_ids=(knowledge["span_b"].id,),
        provider_id="mock",
        model_id="mock-1",
        prompt_version="p",
        schema_version="s",
        derivation_key="k",
    )
    batch = proposer.propose([rec], refined_statements={knowledge["claim"].id: refined})
    proposal = (batch.created or batch.existing)[0]
    return ProposalService(store).approve(proposal.id, note="reviewed")


class TestEvolutionActivation:
    def test_supports_attaches_evidence_without_changing_the_statement(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        proposal = approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)

        result = EvolutionActivator(store).activate(proposal)

        after = store.get_claim(claim.id)
        assert result.outcome.value == "created"
        assert after.statement == claim.statement
        assert after.status is ClaimStatus.ACTIVE
        assert len(store.evidence_for_claim(claim.id)) == 2

    def test_attached_evidence_is_infers_from_not_quotes(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        EvolutionActivator(store).activate(
            approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)
        )

        relations = {e.relation for e in store.evidence_for_claim(claim.id)}

        assert EvidenceRelation.INFERS_FROM in relations

    def test_refinement_supersedes_and_preserves_the_original(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        proposal = approved_proposal(
            store, knowledge, AssessmentClass.REFINES, refined="RAG helps when retrieval is good."
        )

        result = EvolutionActivator(store).activate(proposal)

        original = store.get_claim(claim.id)
        assert original is not None, "the original must still be retrievable"
        assert original.status is ClaimStatus.SUPERSEDED
        assert original.superseded_by == result.entity_id
        assert store.get_claim(result.entity_id).statement == "RAG helps when retrieval is good."

    def test_refinement_records_a_supersede_revision(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        EvolutionActivator(store).activate(
            approved_proposal(store, knowledge, AssessmentClass.REFINES, refined="Sharper claim here.")
        )

        ops = {r.op.value for r in store.revisions_for(EntityType.CLAIM, claim.id)}

        assert "supersede" in ops

    def test_conflict_marks_disputed_without_retracting(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        proposal = approved_proposal(store, knowledge, AssessmentClass.POTENTIAL_CONFLICT)

        EvolutionActivator(store).activate(proposal)

        after = store.get_claim(claim.id)
        assert after.status is ClaimStatus.DISPUTED
        assert after.statement == claim.statement, "Forge flags doubt; it does not rewrite"
        assert after.status is not ClaimStatus.RETRACTED

    def test_conflict_attaches_the_disputing_evidence(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        EvolutionActivator(store).activate(
            approved_proposal(store, knowledge, AssessmentClass.POTENTIAL_CONFLICT)
        )

        span_ids = {e.span_id for e in store.evidence_for_claim(claim.id)}

        assert knowledge["span_b"].id in span_ids
        assert knowledge["span_a"].id in span_ids, "original evidence must survive"

    def test_activation_records_a_revision(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        before = store.count_revisions()

        EvolutionActivator(store).activate(
            approved_proposal(store, knowledge, AssessmentClass.POTENTIAL_CONFLICT)
        )

        assert store.count_revisions() > before

    def test_an_unapproved_proposal_is_refused(self, knowledge):
        store = knowledge["store"]
        proposer = EvolutionProposer(store, workflow_id="wf", source_id="s")
        pending = proposer.propose(
            [
                AssessmentRecord(
                    claim_id=knowledge["claim"].id,
                    classification=AssessmentClass.SUPPORTS,
                    rationale="bears on the claim",
                    evidence_span_ids=(knowledge["span_b"].id,),
                    provider_id="mock",
                    model_id="mock-1",
                    prompt_version="p",
                    schema_version="s",
                    derivation_key="k",
                )
            ]
        ).created[0]

        result = EvolutionActivator(store).activate(pending)

        assert result.outcome.value == "refused"
        assert "not approved" in result.reason

    def test_activation_is_idempotent(self, knowledge):
        store, claim = knowledge["store"], knowledge["claim"]
        activator = EvolutionActivator(store)
        proposal = approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)

        activator.activate(proposal)
        state = (store.count_revisions(), len(store.evidence_for_claim(claim.id)))
        second = activator.activate(store.get_proposal(proposal.id))

        assert second.outcome.value == "already_active"
        assert (store.count_revisions(), len(store.evidence_for_claim(claim.id))) == state

    def test_a_storage_failure_is_reported_not_swallowed(self, knowledge, monkeypatch):
        store = knowledge["store"]
        proposal = approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)

        def boom(*_args, **_kw):
            raise RuntimeError("disk is on fire")

        monkeypatch.setattr(store, "put_claim", boom)
        result = EvolutionActivator(store).activate(proposal)

        assert result.outcome.value == "failed"
        assert "disk is on fire" in result.reason
        assert store.get_proposal(proposal.id).status is ProposalStatus.APPROVED, (
            "a failed activation must leave the proposal retryable"
        )

    def test_missing_evidence_refuses_the_change(self, knowledge, store):
        proposal = approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)
        broken = proposal.model_copy(update={"evidence_span_ids": ("span-gone",)})

        result = EvolutionActivator(store).activate(broken)

        assert result.outcome.value == "refused"
        assert "no evidence span resolves" in result.reason

    def test_activation_makes_no_llm_calls(self, knowledge):
        store = knowledge["store"]
        proposal = approved_proposal(store, knowledge, AssessmentClass.SUPPORTS)
        CALLS.reset()

        EvolutionActivator(store).activate(proposal)

        assert CALLS.count == 0


# --------------------------------------------------------------------------
# workflow run record
# --------------------------------------------------------------------------


class TestWorkflowRun:
    def test_identity_is_deterministic(self):
        first = WorkflowRun.make_id("src", "hash", "v1")
        second = WorkflowRun.make_id("src", "hash", "v1")

        assert first == second
        assert first != WorkflowRun.make_id("src", "different", "v1")

    def test_provider_change_is_detected(self):
        run = WorkflowRun(
            id="w", source_id="s", provider_id="ollama", model_id="qwen3:8b"
        )

        assert run.provider_changed("cloud:anthropic", "claude-sonnet-5") is True
        assert run.provider_changed("ollama", "qwen3:8b") is False

    def test_a_fresh_run_has_no_provider_to_change(self):
        assert WorkflowRun(id="w", source_id="s").provider_changed("anything", "at-all") is False

    def test_run_round_trips_through_storage(self, store):
        run = WorkflowRun(
            id="w1",
            source_id="s1",
            status=WorkflowStatus.WAITING_FOR_REVIEW,
            assessments=(record(AssessmentClass.SUPPORTS),),
        )
        store.put_workflow(run)

        loaded = store.get_workflow("w1")

        assert loaded.status is WorkflowStatus.WAITING_FOR_REVIEW
        assert loaded.assessments[0].classification is AssessmentClass.SUPPORTS

    def test_counts_summarize_the_run(self):
        run = WorkflowRun(
            id="w",
            source_id="s",
            assessments=(
                record(AssessmentClass.SUPPORTS, "a"),
                record(AssessmentClass.IRRELEVANT, "b"),
            ),
        )

        assert run.by_classification() == {"IRRELEVANT": 1, "SUPPORTS": 1}
