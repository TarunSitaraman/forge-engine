"""Phase 4 end-to-end: the agentic evolution loop, through LangGraph and the CLI.

The properties under test are the ones that distinguish this from a script with
extra steps:

* **It stops itself.** The workflow pauses at ``await_human_review`` and
  persists, rather than proceeding on its own judgement.
* **It resumes rather than restarts.** A resumed run continues from the
  checkpoint and does not re-pay for semantic work already done — asserted by
  counting model calls, not by inspection.
* **It routes.** Different evidence takes different paths through the graph,
  and the cheap paths never reach the model.
* **It changes nothing without approval.** Every test that reaches activation
  also checks the prior state survived.

CI is fully offline: every model call goes through a scripted provider over the
real ``LLMProvider`` interface.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.domain import (
    AssessmentClass,
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
    ProposalStatus,
    ProposalType,
    Provenance,
    ProvenanceTier,
    Source,
    SourceKind,
    Span,
    WorkflowStatus,
    deterministic_provenance,
)
from forge.evolution.service import EvolutionService, ProviderMismatch
from forge.evolution.state import APPROVAL_APPROVED, APPROVAL_REJECTED
from forge.llm import MockProvider
from forge.llm.base import CALLS, LLMError, ProviderUnavailable
from forge.proposals import ProposalService
from forge.storage import SqliteStore

runner = CliRunner()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def model_provenance() -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent="CandidateExtractor",
        model_id="mock|mock-1",
    )


@pytest.fixture
def evolving(settings):
    """A store holding one claim from paper A and new evidence from paper B."""
    store = SqliteStore(settings.db_path)
    store.initialize()

    concept = Concept(
        id=Concept.make_id("Retrieval Augmented Generation"),
        canonical_name="Retrieval Augmented Generation",
        kind=ConceptKind.TECHNOLOGY,
        provenance=model_provenance(),
        origin_proposal_id="p-seed",
    )
    store.put_concept(concept)

    def add(name: str, text: str, heading: tuple[str, ...], page: int):
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

    _, span_a = add(
        "paper-a",
        "RAG can improve factual accuracy on knowledge-intensive tasks.",
        ("Retrieval Augmented Generation",),
        1,
    )
    source_b, span_b = add(
        "paper-b",
        "Retrieval Augmented Generation can introduce errors when the retrieved "
        "context is irrelevant.",
        ("Retrieval Augmented Generation", "Failure Modes"),
        2,
    )
    source_c, _ = add(
        "paper-c",
        "Dijkstra's algorithm computes shortest paths over weighted graphs.",
        ("Shortest Paths",),
        1,
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
    yield {
        "store": store,
        "settings": settings,
        "claim": claim,
        "concept": concept,
        "span_b": span_b,
        "source_b": source_b,
        "source_c": source_c,
    }
    store.close()


def provider_for(classification: str = "POTENTIAL_CONFLICT", refined: str = "") -> MockProvider:
    """Scripted analysis model that echoes the ids Forge actually showed it."""

    def respond(request):
        text = request.messages[-1].content
        claims = re.findall(r"\[claim_id: ([^\]]+)\]", text)
        spans = re.findall(r"\[span_id: ([^\]]+)\]", text)
        if not claims or not spans:
            return json.dumps({"assessments": []})
        return json.dumps(
            {
                "assessments": [
                    {
                        "claim_id": claims[0],
                        "classification": classification,
                        "rationale": "The new evidence qualifies the conditions of this claim.",
                        "evidence_span_ids": [spans[0]],
                        "refined_statement": refined,
                    }
                ]
            }
        )

    return MockProvider(responder=respond)


def service_for(evolving, provider=None, **kw) -> EvolutionService:
    return EvolutionService(
        evolving["store"],
        evolving["settings"],
        provider=provider if provider is not None else provider_for(),
        provider_id=kw.pop("provider_id", "mock"),
        model_id=kw.pop("model_id", "mock-1"),
        **kw,
    )


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


class TestWorkflowExecution:
    def test_the_full_arc_reaches_a_pause(self, evolving):
        service = service_for(evolving)

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.WAITING_FOR_REVIEW
        assert outcome.interrupted is True
        assert outcome.run.impact is ImpactClass.POTENTIAL_CONFLICT
        assert len(outcome.run.proposal_ids) == 1
        service.close()

    def test_nodes_execute_in_order(self, evolving):
        service = service_for(evolving)

        run = service.start(evolving["source_b"].id).run

        assert [n.node for n in run.nodes] == [
            "register_evidence",
            "identify_affected_concepts",
            "retrieve_related_claims",
            "assess_evidence",
            "classify_impact",
            "generate_proposals",
        ]
        service.close()

    def test_knowledge_is_untouched_while_awaiting_review(self, evolving):
        store, claim = evolving["store"], evolving["claim"]
        service = service_for(evolving)

        service.start(evolving["source_b"].id)

        after = store.get_claim(claim.id)
        assert after.status is ClaimStatus.ACTIVE
        assert after.statement == claim.statement
        assert len(store.evidence_for_claim(claim.id)) == 1
        service.close()

    def test_the_interrupt_explains_how_to_decide(self, evolving):
        service = service_for(evolving)

        payload = service.start(evolving["source_b"].id).interrupt_payload

        assert "forge proposals approve" in payload["how_to_decide"]
        assert payload["proposals"][0]["type"] == ProposalType.CLAIM_CONFLICT.value
        service.close()

    def test_unrelated_evidence_never_reaches_the_model(self, evolving):
        """Deterministic narrowing is what makes the common case free."""
        service = service_for(evolving)
        CALLS.reset()

        outcome = service.start(evolving["source_c"].id)

        assert outcome.status is WorkflowStatus.COMPLETED
        assert outcome.run.impact is ImpactClass.NEW_KNOWLEDGE
        assert CALLS.count == 0
        assert [n.node for n in outcome.run.nodes][-1] == "finalize_workflow"
        service.close()

    def test_a_run_records_provider_and_model(self, evolving):
        service = service_for(evolving, provider_id="cloud:anthropic", model_id="claude-sonnet-5")

        run = service.start(evolving["source_b"].id).run

        assert run.provider_id == "cloud:anthropic"
        assert run.assessments[0].model_id == "claude-sonnet-5"
        service.close()

    def test_workflow_id_is_deterministic(self, evolving):
        service = service_for(evolving)

        first = service.workflow_id_for(evolving["source_b"].id)
        second = service.workflow_id_for(evolving["source_b"].id)

        assert first == second
        service.close()


class TestHumanInTheLoop:
    def test_approval_then_resume_applies_the_change(self, evolving):
        store, claim = evolving["store"], evolving["claim"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id, note="reviewed")

        resumed = service.resume(outcome.run.id)

        assert resumed.status is WorkflowStatus.COMPLETED
        assert store.get_claim(claim.id).status is ClaimStatus.DISPUTED
        assert len(resumed.run.revision_ids) >= 1
        service.close()

    def test_rejection_ends_the_run_without_changing_knowledge(self, evolving):
        store, claim = evolving["store"], evolving["claim"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).reject(proposal_id, note="not persuasive")

        resumed = service.resume(outcome.run.id)

        assert resumed.status is WorkflowStatus.COMPLETED
        assert resumed.run.activated_entity_ids == ()
        assert store.get_claim(claim.id).status is ClaimStatus.ACTIVE
        service.close()

    def test_resuming_without_deciding_pauses_again(self, evolving):
        """An undecided proposal must not be treated as consent."""
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)

        again = service.resume(outcome.run.id)

        assert again.status is WorkflowStatus.WAITING_FOR_REVIEW
        assert again.interrupted is True
        service.close()

    def test_the_stored_decision_is_what_counts(self, evolving):
        """Approval flows through the proposal system, not the resume payload."""
        store = evolving["store"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        proposal_id = outcome.run.proposal_ids[0]
        ProposalService(store).approve(proposal_id)

        resumed = service.resume(outcome.run.id)

        assert store.get_proposal(proposal_id).status is ProposalStatus.ACTIVATED
        assert resumed.run.activated_entity_ids
        service.close()


class TestResumeAcrossProcessRestart:
    def test_a_paused_run_survives_losing_every_object(self, evolving, settings):
        """The one feature checkpointing exists for."""
        store = evolving["store"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        workflow_id = outcome.run.id
        proposal_ids = list(outcome.run.proposal_ids)

        # Tear everything down, as a process exit would.
        service.close()
        store.close()
        del service, outcome, store

        reopened = SqliteStore(settings.db_path)
        reopened.initialize()
        for proposal_id in proposal_ids:
            ProposalService(reopened).approve(proposal_id, note="reviewed after restart")

        revived = EvolutionService(
            reopened,
            settings,
            provider=provider_for(),
            provider_id="mock",
            model_id="mock-1",
        )
        resumed = revived.resume(workflow_id)

        assert resumed.status is WorkflowStatus.COMPLETED
        assert resumed.run.activated_entity_ids
        revived.close()
        reopened.close()

    def test_resume_does_not_repeat_the_semantic_work(self, evolving):
        store = evolving["store"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        CALLS.reset()

        service.resume(outcome.run.id)

        assert CALLS.count == 0, "a resumed run must not re-pay for assessment"
        service.close()

    def test_a_paused_workflow_is_listed_as_waiting(self, evolving):
        store = evolving["store"]
        service = service_for(evolving)
        service.start(evolving["source_b"].id)

        waiting = store.list_workflows(status=WorkflowStatus.WAITING_FOR_REVIEW)

        assert len(waiting) == 1
        assert waiting[0].awaiting_review is True
        service.close()

    def test_starting_a_paused_workflow_resumes_it_rather_than_restarting(self, evolving):
        store = evolving["store"]
        service = service_for(evolving)
        first = service.start(evolving["source_b"].id)
        for proposal_id in first.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        CALLS.reset()

        second = service.start(evolving["source_b"].id)

        assert second.status is WorkflowStatus.COMPLETED
        assert CALLS.count == 0
        service.close()


class TestProviderRules:
    def test_an_unavailable_provider_halts_explicitly(self, evolving):
        service = service_for(evolving, provider=MockProvider(fail_with=ProviderUnavailable("off")))

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE
        assert outcome.run.proposal_ids == ()
        assert any("unavailable" in e for e in outcome.run.errors)
        service.close()

    def test_no_provider_at_all_is_reported_not_worked_around(self, evolving):
        service = EvolutionService(evolving["store"], evolving["settings"])

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.SEMANTIC_ANALYSIS_UNAVAILABLE
        service.close()

    def test_an_unavailable_run_leaves_knowledge_untouched(self, evolving):
        store, claim = evolving["store"], evolving["claim"]
        service = service_for(evolving, provider=MockProvider(fail_with=ProviderUnavailable("off")))

        service.start(evolving["source_b"].id)

        assert store.get_claim(claim.id).status is ClaimStatus.ACTIVE
        service.close()

    def test_resuming_under_a_different_model_is_refused(self, evolving):
        store = evolving["store"]
        service = service_for(evolving, provider_id="ollama", model_id="qwen3:8b")
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        service.close()

        other = service_for(evolving, provider_id="cloud:anthropic", model_id="claude-sonnet-5")
        with pytest.raises(ProviderMismatch, match="two models"):
            other.resume(outcome.run.id)
        other.close()

    def test_a_provider_change_can_be_accepted_explicitly(self, evolving):
        store = evolving["store"]
        service = service_for(evolving, provider_id="ollama", model_id="qwen3:8b")
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        service.close()

        other = service_for(evolving, provider_id="cloud:anthropic", model_id="claude-sonnet-5")
        resumed = other.resume(outcome.run.id, allow_provider_change=True)

        assert resumed.status is WorkflowStatus.COMPLETED
        assert any("two models" in w for w in resumed.run.warnings), (
            "the change must be recorded, not hidden"
        )
        other.close()

    def test_malformed_output_fails_the_run_without_proposing(self, evolving):
        service = service_for(evolving, provider=MockProvider(default_response="not json"))

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.FAILED
        assert outcome.run.proposal_ids == ()
        service.close()

    def test_a_transport_failure_fails_the_run(self, evolving):
        service = service_for(evolving, provider=MockProvider(fail_with=LLMError("timeout")))

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.FAILED
        service.close()


class TestClassificationRouting:
    def test_supports_produces_an_evidence_proposal(self, evolving):
        store = evolving["store"]
        service = service_for(evolving, provider=provider_for("SUPPORTS"))

        outcome = service.start(evolving["source_b"].id)

        proposal = store.get_proposal(outcome.run.proposal_ids[0])
        assert proposal.type is ProposalType.CLAIM_EVIDENCE
        assert outcome.run.impact is ImpactClass.SUPPORTS
        service.close()

    def test_refines_supersedes_on_approval(self, evolving):
        store, claim = evolving["store"], evolving["claim"]
        service = service_for(
            evolving,
            provider=provider_for("REFINES", refined="RAG improves accuracy when retrieval is relevant."),
        )
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)

        service.resume(outcome.run.id)

        original = store.get_claim(claim.id)
        assert original.status is ClaimStatus.SUPERSEDED
        assert original.superseded_by is not None
        assert store.get_claim(original.superseded_by).statement.startswith("RAG improves accuracy")
        service.close()

    def test_irrelevant_produces_no_proposal_and_no_pause(self, evolving):
        service = service_for(evolving, provider=provider_for("IRRELEVANT"))

        outcome = service.start(evolving["source_b"].id)

        assert outcome.status is WorkflowStatus.COMPLETED
        assert outcome.interrupted is False
        assert outcome.run.impact is ImpactClass.NO_MATERIAL_CHANGE
        assert outcome.run.proposal_ids == ()
        service.close()

    def test_insufficient_evidence_produces_no_proposal(self, evolving):
        service = service_for(evolving, provider=provider_for("INSUFFICIENT_EVIDENCE"))

        outcome = service.start(evolving["source_b"].id)

        assert outcome.run.proposal_ids == ()
        assert outcome.status is WorkflowStatus.COMPLETED
        service.close()


class TestIdempotency:
    def test_the_whole_workflow_is_safe_to_repeat(self, evolving):
        store = evolving["store"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        service.resume(outcome.run.id)

        before = (
            store.count_revisions(),
            len(store.list_claims()),
            store.counts()["evidence_links"],
            store.counts()["proposals"],
            store.counts()["workflows"],
        )
        CALLS.reset()
        service.start(evolving["source_b"].id)
        after = (
            store.count_revisions(),
            len(store.list_claims()),
            store.counts()["evidence_links"],
            store.counts()["proposals"],
            store.counts()["workflows"],
        )

        assert before == after, "a repeated workflow must create nothing new"
        assert CALLS.count == 0, "and must spend nothing"
        service.close()

    def test_repeating_uses_the_assessment_cache(self, evolving):
        store = evolving["store"]
        service = service_for(evolving)
        first = service.start(evolving["source_b"].id)
        for proposal_id in first.run.proposal_ids:
            ProposalService(store).reject(proposal_id)
        service.resume(first.run.id)

        second = service.start(evolving["source_b"].id)

        assert second.run.cache_hits >= 1
        assert second.run.llm_calls == 0
        service.close()


class TestObservability:
    def test_every_node_records_timing(self, evolving):
        service = service_for(evolving)

        run = service.start(evolving["source_b"].id).run

        assert all(n.duration_ms >= 0 for n in run.nodes)
        assert run.duration_ms > 0
        service.close()

    def test_only_the_assessment_node_spends_calls(self, evolving):
        service = service_for(evolving)

        run = service.start(evolving["source_b"].id).run

        spending = {n.node for n in run.nodes if n.llm_calls > 0}
        assert spending == {"assess_evidence"}
        service.close()

    def test_explain_answers_why_forge_proposed_this(self, evolving):
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)

        detail = service.explain(outcome.run.id)

        assert detail["candidates_detail"][0]["detail"], "must say why the concept was considered"
        assert detail["assessments_detail"][0]["rationale"]
        assert detail["proposals"][0]["reason"]
        assert detail["evidence"][0]["citation"]
        service.close()


class TestNoWriteBack:
    def test_the_workflow_never_writes_to_the_vault(self, evolving, real_vault, tmp_path):
        """D2 still holds: evolution writes to the store, never to Markdown."""
        before = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout

        store = evolving["store"]
        service = service_for(evolving)
        outcome = service.start(evolving["source_b"].id)
        for proposal_id in outcome.run.proposal_ids:
            ProposalService(store).approve(proposal_id)
        service.resume(outcome.run.id)
        service.close()

        after = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout
        assert before == after


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def env_for(settings) -> dict[str, str]:
    return {
        "FORGE_VAULT_PATH": str(settings.vault_path),
        "FORGE_STATE_DIR": str(settings.state_dir),
        "FORGE_LLM_PROVIDER": "mock",
    }


class TestPhase4Cli:
    def test_evolve_reports_a_pause(self, evolving, settings, monkeypatch):
        import forge.cli.phase4 as phase4

        monkeypatch.setattr(
            phase4, "_service", lambda s, store, **kw: service_for(evolving)
        )
        result = runner.invoke(
            app, ["evolve", "papers/paper-b.pdf"], env=env_for(settings)
        )

        assert result.exit_code == 0, result.stdout
        assert "WAITING_FOR_REVIEW" in result.stdout
        assert "Retrieval Augmented Generation" in result.stdout

    def test_evolve_accepts_a_bare_filename(self, evolving, settings, monkeypatch):
        import forge.cli.phase4 as phase4

        monkeypatch.setattr(phase4, "_service", lambda s, store, **kw: service_for(evolving))
        result = runner.invoke(app, ["evolve", "paper-b.pdf", "--json"], env=env_for(settings))

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "waiting_for_review"

    def test_evolve_on_an_unknown_source_exits_nonzero(self, evolving, settings):
        result = runner.invoke(app, ["evolve", "never-ingested.pdf"], env=env_for(settings))

        assert result.exit_code == 1
        assert "no ingested source" in result.stdout + str(result.stderr or "")

    def test_workflow_list_shows_runs(self, evolving, settings, monkeypatch):
        import forge.cli.phase4 as phase4

        monkeypatch.setattr(phase4, "_service", lambda s, store, **kw: service_for(evolving))
        runner.invoke(app, ["evolve", "paper-b.pdf"], env=env_for(settings))

        result = runner.invoke(app, ["workflow", "list", "--json"], env=env_for(settings))

        payload = json.loads(result.stdout)
        assert payload["counts"] == {"waiting_for_review": 1}

    def test_workflow_status_and_inspect(self, evolving, settings, monkeypatch):
        import forge.cli.phase4 as phase4

        monkeypatch.setattr(phase4, "_service", lambda s, store, **kw: service_for(evolving))
        started = runner.invoke(app, ["evolve", "paper-b.pdf", "--json"], env=env_for(settings))
        workflow_id = json.loads(started.stdout)["id"]

        status = runner.invoke(
            app, ["workflow", "status", workflow_id, "--json"], env=env_for(settings)
        )
        inspect = runner.invoke(
            app, ["workflow", "inspect", workflow_id[:12]], env=env_for(settings)
        )

        assert json.loads(status.stdout)["status"] == "waiting_for_review"
        assert inspect.exit_code == 0
        assert "why" not in inspect.stdout.lower() or "considered" in inspect.stdout
        assert "POTENTIAL_CONFLICT" in inspect.stdout

    def test_resume_through_the_cli_after_approving(self, evolving, settings, monkeypatch):
        import forge.cli.phase4 as phase4

        monkeypatch.setattr(phase4, "_service", lambda s, store, **kw: service_for(evolving))
        store = evolving["store"]
        started = runner.invoke(app, ["evolve", "paper-b.pdf", "--json"], env=env_for(settings))
        payload = json.loads(started.stdout)
        for proposal_id in payload["proposal_ids"]:
            runner.invoke(app, ["proposals", "approve", proposal_id], env=env_for(settings))

        result = runner.invoke(
            app, ["workflow", "resume", payload["id"], "--json"], env=env_for(settings)
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "completed"
        assert store.get_claim(evolving["claim"].id).status is ClaimStatus.DISPUTED

    def test_unknown_workflow_exits_nonzero(self, evolving, settings):
        result = runner.invoke(
            app, ["workflow", "status", "no-such-workflow"], env=env_for(settings)
        )

        assert result.exit_code == 1
