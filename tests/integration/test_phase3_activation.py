"""Phase 3 end-to-end: the proposal → approval → canonical knowledge loop.

These tests exercise the CLI as a user would, because the loop is only closed
if it is closed *through the commands* — an activator that works when driven
from Python but not from `forge activate` has not shipped.

Two properties recur and are asserted repeatedly:

* **Batch operations are guarded.** Bulk approval never touches ambiguous
  proposals unless explicitly told to, and defaults to a dry run.
* **The vault is never written.** Every command here reads Markdown and writes
  only to the derived store. The final test asserts that against `git status`
  on the real repository.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from typer.testing import CliRunner

from forge.activation import ProposalActivator
from forge.cli.main import app
from forge.domain import (
    Concept,
    ConceptKind,
    Derivation,
    Document,
    IdentityState,
    Proposal,
    ProposalStatus,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    SafetyClass,
    Source,
    SourceKind,
    Span,
)
from forge.identity import IdentityConfig, IdentityService
from forge.llm.base import CALLS
from forge.proposals import ProposalService
from forge.storage import SqliteStore

runner = CliRunner()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def env_for(settings) -> dict[str, str]:
    return {
        "FORGE_VAULT_PATH": str(settings.vault_path),
        "FORGE_STATE_DIR": str(settings.state_dir),
    }


def model_provenance() -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent="CandidateExtractor",
        model_id="llama3.1:8b",
    )


@pytest.fixture
def cli_store(settings):
    """A store at the path the CLI will use for these settings."""
    store = SqliteStore(settings.db_path)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def evidence_span(cli_store):
    source = Source.for_path("papers/phase3.pdf", kind=SourceKind.PDF, content_hash="p3")
    cli_store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "p3"),
        source_id=source.id,
        parser="forge.pdf",
        parser_version="1",
        content_hash="p3",
    )
    cli_store.put_document(document)
    span = Span(
        id=Span.make_id(document.id, 0, "p.2"),
        document_id=document.id,
        ordinal=0,
        locator="p.2 L1-L4",
        heading_path=("Activation",),
        start_line=1,
        end_line=4,
        text=(
            "Proposal activation turns an approved proposal into a canonical concept "
            "without discarding the evidence that produced it."
        ),
        content_hash="p3s1",
        page=2,
    )
    cli_store.put_spans([span])
    return span


def concept_proposal(span_id: str, name: str, pid: str, safety=SafetyClass.MODEL_GENERATED):
    return Proposal(
        id=pid,
        type=ProposalType.NEW_CONCEPT,
        safety=safety,
        operation=ProposedOperation(
            action="create_concept", target=name, details={"kind": "technology"}
        ),
        reason="extracted from an ingested source",
        evidence_span_ids=(span_id,),
        provenance=model_provenance(),
    )


# --------------------------------------------------------------------------
# the loop, through the CLI
# --------------------------------------------------------------------------


class TestActivationCli:
    def test_approved_proposal_becomes_a_canonical_concept(
        self, settings, cli_store, evidence_span
    ):
        service = ProposalService(cli_store)
        service.create(concept_proposal(evidence_span.id, "Proposal Activation", "p-act-1"))
        service.approve("p-act-1", note="reviewed")

        result = runner.invoke(app, ["activate", "--json"], env=env_for(settings))

        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["counts"]["created"] == 1
        assert cli_store.get_concept_by_name("Proposal Activation") is not None

    def test_activation_is_idempotent_across_cli_runs(
        self, settings, cli_store, evidence_span
    ):
        """Exit criterion: re-running activation must not duplicate knowledge."""
        service = ProposalService(cli_store)
        service.create(concept_proposal(evidence_span.id, "Idempotent Concept", "p-act-2"))
        service.approve("p-act-2")
        env = env_for(settings)

        runner.invoke(app, ["activate"], env=env)
        before = cli_store.counts()["concepts"]
        second = runner.invoke(app, ["activate", "--json"], env=env)

        assert cli_store.counts()["concepts"] == before
        payload = json.loads(second.stdout)
        assert payload["counts"].get("created", 0) == 0

    def test_pending_proposals_are_not_activated(self, settings, cli_store, evidence_span):
        ProposalService(cli_store).create(
            concept_proposal(evidence_span.id, "Never Approved", "p-act-3")
        )

        runner.invoke(app, ["activate"], env=env_for(settings))

        assert cli_store.get_concept_by_name("Never Approved") is None, (
            "activation must require an approval decision, not merely a proposal"
        )

    def test_activation_makes_no_llm_calls(self, settings, cli_store, evidence_span):
        service = ProposalService(cli_store)
        service.create(concept_proposal(evidence_span.id, "Deterministic Activation", "p-act-4"))
        service.approve("p-act-4")
        CALLS.reset()

        runner.invoke(app, ["activate"], env=env_for(settings))

        assert CALLS.count == 0, "activation is deterministic work and must not call a model"

    def test_unknown_proposal_id_exits_nonzero(self, settings, cli_store):
        result = runner.invoke(app, ["activate", "no-such-proposal"], env=env_for(settings))

        assert result.exit_code == 1

    def test_concept_command_shows_origin_and_evidence(
        self, settings, cli_store, evidence_span
    ):
        service = ProposalService(cli_store)
        service.create(concept_proposal(evidence_span.id, "Traceable Concept", "p-act-5"))
        service.approve("p-act-5", note="reviewed by hand")
        env = env_for(settings)
        runner.invoke(app, ["activate"], env=env)

        result = runner.invoke(app, ["concept", "Traceable Concept", "--json"], env=env)

        assert result.exit_code == 0
        detail = json.loads(result.stdout)
        assert detail["origin_proposal"]["id"] == "p-act-5"
        assert detail["origin_spans"], "a concept must be able to show what evidenced it"
        assert "p.2" in detail["origin_spans"][0]["citation"]

    def test_a_model_supplied_namespace_is_ignored(self, settings, cli_store, evidence_span):
        """Namespacing is a user decision, not something extraction may assert.

        A proposal carrying ``namespace`` in its details must not be able to
        create a namespaced concept — otherwise the model would be choosing
        the vocabulary the identity config exists to let the user choose.
        """
        service = ProposalService(cli_store)
        proposal = concept_proposal(evidence_span.id, "Heap", "p-ns-1").model_copy(
            update={
                "operation": ProposedOperation(
                    action="create_concept",
                    target="Heap",
                    details={"kind": "concept", "namespace": "invented-by-the-model"},
                )
            }
        )
        service.create(proposal)
        service.approve("p-ns-1")

        runner.invoke(app, ["activate"], env=env_for(settings))

        created = cli_store.concepts_named("Heap")
        assert [c.namespace for c in created] == [None]

    def test_concept_command_reports_ambiguity_instead_of_guessing(
        self, settings, cli_store, evidence_span
    ):
        for namespace in ("data-structure", "memory"):
            cli_store.put_concept(
                Concept(
                    id=Concept.make_id("Heap", namespace),
                    canonical_name="Heap",
                    namespace=namespace,
                    kind=ConceptKind.CONCEPT,
                    origin_proposal_id="p-seed",
                    provenance=model_provenance(),
                )
            )

        result = runner.invoke(app, ["concept", "Heap", "--json"], env=env_for(settings))

        payload = json.loads(result.stdout)
        assert payload["ambiguous"] is True
        assert sorted(payload["candidates"]) == ["data-structure/Heap", "memory/Heap"]


class TestGraphCli:
    def test_stats_reports_a_measured_graph(self, settings, cli_store, evidence_span):
        service = ProposalService(cli_store)
        for i, name in enumerate(("Alpha Concept", "Bravo Concept")):
            pid = f"p-graph-{i}"
            service.create(concept_proposal(evidence_span.id, name, pid))
            service.approve(pid)
        env = env_for(settings)
        runner.invoke(app, ["activate"], env=env)

        result = runner.invoke(app, ["graph", "stats", "--json"], env=env)

        assert result.exit_code == 0
        stats = json.loads(result.stdout)
        assert stats["nodes"] == 2
        assert stats["neighbor_query_ms"] >= 0.0

    def test_relationships_is_dry_run_by_default(self, settings, cli_store, evidence_span):
        service = ProposalService(cli_store)
        for i, name in enumerate(("Proposal Activation", "Canonical Concept")):
            pid = f"p-rel-{i}"
            service.create(concept_proposal(evidence_span.id, name, pid))
            service.approve(pid)
        env = env_for(settings)
        runner.invoke(app, ["activate"], env=env)

        result = runner.invoke(app, ["relationships", "--json"], env=env)

        assert json.loads(result.stdout)["dry_run"] is True
        assert cli_store.all_links() == [], "a dry run must create nothing"

    def test_path_reports_absence_without_claiming_none_exists(
        self, settings, cli_store, evidence_span
    ):
        service = ProposalService(cli_store)
        for i, name in enumerate(("Lonely One", "Lonely Two")):
            pid = f"p-path-{i}"
            service.create(concept_proposal(evidence_span.id, name, pid))
            service.approve(pid)
        env = env_for(settings)
        runner.invoke(app, ["activate"], env=env)

        result = runner.invoke(app, ["graph", "path", "Lonely One", "Lonely Two"], env=env)

        assert result.exit_code == 0
        assert "does not prove none exists" in result.stdout


class TestIdentityCli:
    def test_scaffold_documents_collisions_without_deciding_them(self, settings):
        env = env_for(settings)

        result = runner.invoke(app, ["identity", "scaffold", "--json"], env=env)

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        config = IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml")
        assert config.collisions, "the fixture vault contains a deliberate stem collision"
        assert all(r.default is None for r in config.collisions.values())
        assert payload["unresolved"]

    def test_decide_then_list_shows_the_decision(self, settings):
        env = env_for(settings)
        runner.invoke(app, ["identity", "scaffold"], env=env)
        config = IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml")
        name, resolution = next(iter(config.collisions.items()))
        chosen = resolution.identities[0].qualified_name

        decided = runner.invoke(app, ["identity", "decide", resolution.name, chosen], env=env)
        listed = runner.invoke(app, ["identity", "list", "--json"], env=env)

        assert decided.exit_code == 0
        by_name = {c["name"]: c for c in json.loads(listed.stdout)["collisions"]}
        assert by_name[resolution.name]["default"] == chosen

    def test_deciding_an_identity_that_does_not_exist_is_refused(self, settings):
        env = env_for(settings)
        runner.invoke(app, ["identity", "scaffold"], env=env)
        config = IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml")
        name = next(iter(config.collisions.values())).name

        result = runner.invoke(app, ["identity", "decide", name, "invented/Nonsense"], env=env)

        assert result.exit_code == 1

    def test_clear_returns_a_collision_to_undecided(self, settings):
        env = env_for(settings)
        runner.invoke(app, ["identity", "scaffold"], env=env)
        config = IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml")
        name, resolution = next(iter(config.collisions.items()))
        runner.invoke(
            app,
            ["identity", "decide", resolution.name, resolution.identities[0].qualified_name],
            env=env,
        )

        runner.invoke(app, ["identity", "clear", resolution.name], env=env)

        reloaded = IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml")
        assert reloaded.collisions[name].default is None

    def test_a_recorded_decision_resolves_the_bare_name(self, settings):
        env = env_for(settings)
        runner.invoke(app, ["identity", "scaffold"], env=env)
        path = settings.vault_path / "config" / "concept-identity.yaml"
        config = IdentityConfig.load(path)
        name, resolution = next(iter(config.collisions.items()))
        chosen = resolution.identities[0].qualified_name
        runner.invoke(app, ["identity", "decide", resolution.name, chosen], env=env)

        service = IdentityService(IdentityConfig.load(path))
        resolved = service.resolve(resolution.name)

        assert resolved.state is IdentityState.RESOLVED_BY_USER
        assert resolved.identity is not None
        assert resolved.identity.qualified_name == chosen


# --------------------------------------------------------------------------
# batch operations
# --------------------------------------------------------------------------


class TestBatchProposalOperations:
    def _seed(self, cli_store, span_id, count: int, safety: SafetyClass) -> ProposalService:
        service = ProposalService(cli_store)
        for i in range(count):
            service.create(
                concept_proposal(span_id, f"Batch Concept {i}", f"p-batch-{safety.value}-{i}", safety)
            )
        return service

    def test_approve_all_defaults_to_a_dry_run(self, settings, cli_store, evidence_span):
        self._seed(cli_store, evidence_span.id, 3, SafetyClass.MODEL_GENERATED)

        result = runner.invoke(
            app,
            ["proposals", "approve-all", "--safety", "model_generated", "--json"],
            env=env_for(settings),
        )

        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True and payload["matched"] == 3
        assert all(
            p.status is ProposalStatus.PENDING for p in ProposalService(cli_store).list()
        ), "a dry run must decide nothing"

    def test_no_dry_run_approves_the_matched_set(self, settings, cli_store, evidence_span):
        self._seed(cli_store, evidence_span.id, 3, SafetyClass.MODEL_GENERATED)

        result = runner.invoke(
            app,
            [
                "proposals",
                "approve-all",
                "--safety",
                "model_generated",
                "--no-dry-run",
                "--json",
            ],
            env=env_for(settings),
        )

        assert json.loads(result.stdout)["approved"] == 3
        statuses = {p.status for p in ProposalService(cli_store).list()}
        assert statuses == {ProposalStatus.APPROVED}

    def test_ambiguous_proposals_require_an_explicit_flag(
        self, settings, cli_store, evidence_span
    ):
        self._seed(cli_store, evidence_span.id, 2, SafetyClass.AMBIGUOUS)

        refused = runner.invoke(
            app, ["proposals", "approve-all", "--safety", "ambiguous"], env=env_for(settings)
        )

        assert refused.exit_code == 2
        assert "refusing to bulk-approve" in refused.stdout + str(refused.stderr or "")
        assert all(
            p.status is ProposalStatus.PENDING for p in ProposalService(cli_store).list()
        )

    def test_ambiguous_batch_proceeds_when_explicitly_allowed(
        self, settings, cli_store, evidence_span
    ):
        self._seed(cli_store, evidence_span.id, 2, SafetyClass.AMBIGUOUS)

        result = runner.invoke(
            app,
            [
                "proposals",
                "approve-all",
                "--safety",
                "ambiguous",
                "--include-ambiguous",
                "--no-dry-run",
                "--json",
            ],
            env=env_for(settings),
        )

        assert json.loads(result.stdout)["approved"] == 2

    def test_batch_is_filtered_by_safety_class(self, settings, cli_store, evidence_span):
        self._seed(cli_store, evidence_span.id, 2, SafetyClass.MODEL_GENERATED)
        self._seed(cli_store, evidence_span.id, 3, SafetyClass.AMBIGUOUS)

        result = runner.invoke(
            app,
            ["proposals", "approve-all", "--safety", "model_generated", "--json"],
            env=env_for(settings),
        )

        assert json.loads(result.stdout)["matched"] == 2, (
            "safety class comes from provenance and must partition the batch"
        )

    def test_unknown_safety_class_is_rejected(self, settings, cli_store):
        result = runner.invoke(
            app, ["proposals", "approve-all", "--safety", "totally_safe"], env=env_for(settings)
        )

        assert result.exit_code == 2

    def test_batch_approval_activates_cleanly_afterwards(
        self, settings, cli_store, evidence_span
    ):
        self._seed(cli_store, evidence_span.id, 3, SafetyClass.MODEL_GENERATED)
        env = env_for(settings)
        runner.invoke(
            app,
            ["proposals", "approve-all", "--safety", "model_generated", "--no-dry-run"],
            env=env,
        )

        result = runner.invoke(app, ["activate", "--json"], env=env)

        assert json.loads(result.stdout)["counts"]["created"] == 3


# --------------------------------------------------------------------------
# embeddings and evaluation, through the CLI
# --------------------------------------------------------------------------


class TestEmbeddingsCli:
    def test_status_reports_the_null_provider_as_unavailable(self, settings, cli_store):
        result = runner.invoke(app, ["embeddings", "status", "--json"], env=env_for(settings))

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["available"] is False
        assert payload["stored_vectors"] == 0

    def test_build_with_the_hashing_provider_stores_vectors(
        self, settings, cli_store, evidence_span
    ):
        env = env_for(settings)

        result = runner.invoke(
            app, ["embeddings", "build", "--provider", "hashing", "--json"], env=env
        )

        assert result.exit_code == 0, result.stdout
        assert cli_store.count_embeddings() > 0

    def test_hashing_embeddings_need_no_network(self, settings, cli_store, evidence_span):
        """The whole point of the hashing provider: measurable without a download."""
        CALLS.reset()

        runner.invoke(
            app,
            ["embeddings", "build", "--provider", "hashing"],
            env=env_for(settings),
        )

        assert CALLS.count == 0


class TestRetrievalEvalCli:
    def test_eval_runs_the_labelled_set_and_reports_metrics(self, real_vault, tmp_path):
        """The labels point at the real vault, so the evaluation runs there."""
        env = {
            "FORGE_VAULT_PATH": str(real_vault),
            "FORGE_STATE_DIR": str(tmp_path / "eval-state"),
        }
        result = runner.invoke(app, ["retrieval-eval", "--json"], env=env)

        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["queries"] >= 20
        summary = payload["summaries"][0]
        assert summary["method"] == "lexical"
        assert set(summary) >= {"recall@5", "recall@10", "precision@5", "mrr"}


# --------------------------------------------------------------------------
# the vault stays untouched
# --------------------------------------------------------------------------


class TestNoWriteBack:
    def test_the_whole_phase_3_loop_leaves_the_real_vault_clean(
        self, real_vault, real_settings, tmp_path
    ):
        """D2, enforced end to end: activation writes to the store, never to Markdown."""
        before = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout

        env = {
            "FORGE_VAULT_PATH": str(real_vault),
            "FORGE_STATE_DIR": str(tmp_path / "state"),
        }
        runner.invoke(app, ["activate"], env=env)
        runner.invoke(app, ["relationships"], env=env)
        runner.invoke(app, ["graph", "stats"], env=env)

        after = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout
        assert before == after
