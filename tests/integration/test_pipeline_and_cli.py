"""End-to-end pipeline and CLI behaviour on the fixture vault.

Uses the small synthetic vault so files can be created, edited, and deleted —
the real corpus is never written to.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.corpus import IndexPipeline
from forge.llm.base import CALLS
from forge.storage import SqliteStore

runner = CliRunner()


@pytest.fixture
def pipeline(settings, tmp_path):
    store = SqliteStore(tmp_path / "p.db")
    store.initialize()
    yield IndexPipeline(settings, store), store
    store.close()


class TestPipelineLifecycle:
    def test_first_run_persists_everything(self, pipeline):
        pipe, store = pipeline
        result = pipe.run(write_reports=False)
        assert result.persisted_sources == result.index.file_count
        assert result.persisted_spans > 0
        assert store.counts()["sources"] == result.index.file_count

    def test_second_run_persists_nothing_and_calls_no_model(self, pipeline):
        pipe, _ = pipeline
        pipe.run(write_reports=False)
        CALLS.reset()
        second = pipe.run(write_reports=False)
        assert second.persisted_sources == 0
        assert CALLS.count == 0

    def test_editing_a_file_reprocesses_only_it(self, pipeline, fixture_vault):
        pipe, _ = pipeline
        pipe.run(write_reports=False)

        target = fixture_vault / "DSA" / "01_Patterns" / "DFS.md"
        target.write_text(target.read_text() + "\n\nAn appended line.\n", encoding="utf-8")

        result = pipe.run(write_reports=False)
        assert [c.path for c in result.changes.modified] == ["DSA/01_Patterns/DFS.md"]
        assert result.persisted_sources == 1

    def test_new_file_is_detected(self, pipeline, fixture_vault):
        pipe, _ = pipeline
        pipe.run(write_reports=False)
        (fixture_vault / "Notes" / "brand-new.md").write_text("# New\n", encoding="utf-8")
        result = pipe.run(write_reports=False)
        assert [c.path for c in result.changes.new] == ["Notes/brand-new.md"]

    def test_deleted_file_invalidates_derived_state_only(self, pipeline, fixture_vault):
        pipe, store = pipeline
        pipe.run(write_reports=False)
        before = store.counts()["sources"]

        (fixture_vault / "Notes" / "plain-note.md").unlink()
        result = pipe.run(write_reports=False)

        assert [c.path for c in result.changes.deleted] == ["Notes/plain-note.md"]
        assert store.counts()["sources"] == before - 1
        # History of the removed source survives.
        assert store.count_revisions() > 0

    def test_reset_then_reindex_reproduces_state(self, pipeline):
        """Derived state is rebuildable — the core architectural promise."""
        pipe, store = pipeline
        first = pipe.run(write_reports=False)
        counts_before = {k: v for k, v in store.counts().items() if k != "revisions"}

        store.reset()
        assert store.counts()["sources"] == 0

        second = pipe.run(write_reports=False)
        counts_after = {k: v for k, v in store.counts().items() if k != "revisions"}

        assert counts_before == counts_after
        assert first.index.fingerprint() == second.index.fingerprint()

    def test_reports_are_written_as_valid_json(self, pipeline, settings):
        pipe, _ = pipeline
        result = pipe.run()
        assert len(result.reports_written) == 4
        for path in result.reports_written:
            payload = json.loads(open(path).read())
            assert "report" in payload and "fingerprint" in payload

    def test_duplicate_content_is_detected(self, pipeline):
        pipe, _ = pipeline
        result = pipe.run(write_reports=False)
        dupes = result.index.duplicate_hashes
        assert len(dupes) == 1
        assert sorted(next(iter(dupes.values()))) == [
            "Notes/duplicate-a.md",
            "Notes/duplicate-b.md",
        ]


class TestCli:
    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def test_index_reports_zero_llm_calls(self, settings):
        result = runner.invoke(app, ["index", "--json"], env=self._env(settings))
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["llm_calls"] == 0
        assert payload["files_indexed"] > 0

    def test_index_is_idempotent(self, settings):
        env = self._env(settings)
        runner.invoke(app, ["index", "--json"], env=env)
        second = json.loads(runner.invoke(app, ["index", "--json"], env=env).stdout)
        assert second["changes"]["unchanged"] == second["files_indexed"]
        assert second["persisted"]["sources"] == 0

    def test_status_runs_without_a_model(self, settings):
        result = runner.invoke(app, ["status", "--json"], env=self._env(settings))
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["llm"]["reachable"] is False  # no Ollama in CI
        assert payload["markdown_files_on_disk"] > 0

    def test_corpus_stats(self, settings):
        result = runner.invoke(app, ["corpus-stats", "--json"], env=self._env(settings))
        assert result.exit_code == 0
        stats = json.loads(result.stdout)
        assert stats["file_count"] > 0
        assert "by_folder" in stats

    @pytest.mark.parametrize("target", ["all", "frontmatter", "links", "conventions"])
    def test_diagnostics_targets(self, settings, target):
        result = runner.invoke(app, ["diagnostics", target, "--json"], env=self._env(settings))
        assert result.exit_code == 0
        assert json.loads(result.stdout)

    def test_diagnostics_rejects_unknown_target(self, settings):
        result = runner.invoke(app, ["diagnostics", "nonsense"], env=self._env(settings))
        assert result.exit_code == 2

    def test_inspect_shows_repairs_without_applying_them(self, settings, fixture_vault):
        target = "DSA/01_Patterns/DFS.md"
        before = (fixture_vault / target).read_text()

        result = runner.invoke(app, ["inspect", target, "--json"], env=self._env(settings))
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["repairs"], "the fixture has a malformed related: field"

        assert (fixture_vault / target).read_text() == before, "inspect must not write"

    def test_inspect_with_spans(self, settings):
        result = runner.invoke(
            app,
            ["inspect", "DSA/01_Patterns/Graph Traversal.md", "--spans", "--json"],
            env=self._env(settings),
        )
        payload = json.loads(result.stdout)
        assert payload["spans"]
        assert payload["spans"][0]["ordinal"] == 0

    def test_inspect_missing_file_fails_cleanly(self, settings):
        result = runner.invoke(app, ["inspect", "nope.md"], env=self._env(settings))
        assert result.exit_code == 1

    def test_model_test_exits_nonzero_without_a_model(self, settings):
        result = runner.invoke(
            app, ["model-test", "--no-write"], env=self._env(settings)
        )
        assert result.exit_code == 1
        assert "did not run" in result.stdout

    def test_model_test_with_mock_provider(self, settings):
        env = {**self._env(settings), "FORGE_LLM_PROVIDER": "mock"}
        result = runner.invoke(app, ["model-test", "--no-write", "--json"], env=env)
        assert result.exit_code == 0
        assert json.loads(result.stdout)["reachable"] is True

    def test_cli_never_writes_to_the_vault(self, settings, fixture_vault):
        """Every Phase 1 command is read-only with respect to Markdown."""
        snapshot = {
            p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))
        }
        env = self._env(settings)
        for args in (
            ["index"],
            ["status"],
            ["corpus-stats"],
            ["diagnostics", "all"],
            ["inspect", "DSA/01_Patterns/DFS.md"],
        ):
            runner.invoke(app, args, env=env)

        assert {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))} == snapshot


class TestProposalFilterParsing:
    """`proposals list` filters are user input, so a bad one must not traceback.

    The enum's names are upper case and its values lower case, so
    ``--status PENDING`` — which is what `docs/research/extraction-cost.md`
    §4 told people to type — used to raise a raw ``ValueError`` out of
    ``enum`` and print a stack trace over the review workflow.
    """

    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    @pytest.mark.parametrize("status", ["pending", "PENDING", "Pending", " pending "])
    def test_status_filter_is_case_insensitive(self, settings, status):
        result = runner.invoke(app, ["proposals", "list", "--status", status], env=self._env(settings))
        assert result.exit_code == 0

    def test_status_all_is_not_an_enum_member(self, settings):
        result = runner.invoke(app, ["proposals", "list", "--status", "ALL"], env=self._env(settings))
        assert result.exit_code == 0

    def test_unknown_status_exits_two_and_lists_the_valid_set(self, settings):
        result = runner.invoke(app, ["proposals", "list", "--status", "bogus"], env=self._env(settings))
        assert result.exit_code == 2
        assert "Traceback" not in result.output
        for expected in ("pending", "approved", "activated", "rejected", "superseded"):
            assert expected in result.output

    def test_unknown_type_exits_two_and_lists_the_valid_set(self, settings):
        result = runner.invoke(app, ["proposals", "list", "--type", "bogus"], env=self._env(settings))
        assert result.exit_code == 2
        assert "Traceback" not in result.output
        assert "new_concept" in result.output

    @pytest.mark.parametrize("type_", ["new_concept", "NEW_CONCEPT", "claim_conflict"])
    def test_type_filter_accepts_every_documented_value(self, settings, type_):
        result = runner.invoke(app, ["proposals", "list", "--type", type_], env=self._env(settings))
        assert result.exit_code == 0


class TestProposalSampling:
    """`--sample` must draw randomly, not take the first N.

    Proposals come back grouped by source, so the first N describe one or two
    documents. Judging 2,000 proposals from that is judging one file.
    """

    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def _seed_proposals(self, settings, count):
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
        from forge.storage import SqliteStore

        store = SqliteStore(settings.db_path)
        store.initialize()
        for i in range(count):
            store.put_proposal(
                Proposal(
                    id=f"p{i:04d}",
                    type=ProposalType.NEW_CONCEPT,
                    safety=SafetyClass.MODEL_GENERATED,
                    target_entity_type=EntityType.CONCEPT,
                    operation=ProposedOperation(
                        action="create_concept", target=f"concept-{i:04d}"
                    ),
                    reason="test",
                    evidence_span_ids=(f"sp{i:04d}",),
                    provenance=Provenance(
                        tier=ProvenanceTier.EXTRACTED_CLAIM,
                        derivation=Derivation.MODEL,
                        confidence=0.5,
                        agent="test/1",
                        model_id="test-model",
                    ),
                )
            )
        store.close()

    def test_sample_is_not_the_first_n(self, settings):
        self._seed_proposals(settings, 60)
        result = runner.invoke(
            app, ["proposals", "list", "--sample", "10"], env=self._env(settings)
        )
        assert result.exit_code == 0
        assert "random sample of 10 from 60" in result.output
        shown = {line.split()[0] for line in result.output.splitlines() if line.startswith("p0")}
        first_ten = {f"p{i:04d}" for i in range(10)}
        assert shown != first_ten, "a random sample that equals the first ten is not random"

    def test_sample_is_reproducible_for_a_seed(self, settings):
        self._seed_proposals(settings, 60)
        env = self._env(settings)
        args = ["proposals", "list", "--sample", "10", "--seed", "7"]
        first = runner.invoke(app, args, env=env).output
        second = runner.invoke(app, args, env=env).output
        assert first == second

    def test_a_different_seed_draws_differently(self, settings):
        self._seed_proposals(settings, 60)
        env = self._env(settings)
        a = runner.invoke(app, ["proposals", "list", "--sample", "10", "--seed", "1"], env=env)
        b = runner.invoke(app, ["proposals", "list", "--sample", "10", "--seed", "2"], env=env)
        assert a.output != b.output

    def test_sample_larger_than_the_population_is_not_an_error(self, settings):
        self._seed_proposals(settings, 5)
        result = runner.invoke(
            app, ["proposals", "list", "--sample", "50"], env=self._env(settings)
        )
        assert result.exit_code == 0
        assert "random sample of 5 from 5" in result.output


class TestExtractPlanCommand:
    """A cost preview is only useful if it is free and refuses to invent numbers."""

    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def test_it_prices_an_ingested_vault_without_calling_the_model(self, settings, tmp_path):
        from forge.ingestion import IngestionPipeline, IngestOptions

        store = SqliteStore(settings.db_path)
        store.initialize()
        IngestionPipeline(settings, store).ingest_path(settings.vault_path, IngestOptions())
        store.close()

        CALLS.reset()
        result = runner.invoke(
            app,
            ["extract-plan", str(settings.vault_path), "--json"],
            env=self._env(settings),
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["calls"] > 0
        assert CALLS.count == 0, "a plan must cost nothing"

    def test_it_refuses_a_wall_clock_estimate_without_a_measured_rate(self, settings):
        result = runner.invoke(
            app,
            ["extract-plan", str(settings.vault_path), "--json"],
            env=self._env(settings),
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["estimated_hours"] is None
        assert payload["seconds_per_call"] is None

    def test_a_supplied_rate_produces_an_estimate(self, settings, tmp_path):
        from forge.ingestion import IngestionPipeline, IngestOptions

        store = SqliteStore(settings.db_path)
        store.initialize()
        IngestionPipeline(settings, store).ingest_path(settings.vault_path, IngestOptions())
        store.close()

        result = runner.invoke(
            app,
            [
                "extract-plan",
                str(settings.vault_path),
                "--seconds-per-call",
                "49",
                "--json",
            ],
            env=self._env(settings),
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["estimated_hours"] == pytest.approx(payload["calls"] * 49 / 3600)

    def test_an_unpriced_vault_reports_unknown_rather_than_free(self, settings):
        """Reporting an uningested corpus as a zero-cost run is the dangerous bug."""
        result = runner.invoke(
            app, ["extract-plan", str(settings.vault_path)], env=self._env(settings)
        )
        assert result.exit_code == 0
        assert "not ingested" in result.output
        assert "unknown" in result.output


class TestProposalDuplicatesCommand:
    """Dedup has to be reachable from the `forge` command, not just importable.

    Recorded rule from 2026-08-19: an operational tool a user runs belongs on
    the CLI. The engine is installed with pipx, so a module the user cannot
    invoke is a module the user does not have.
    """

    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def _seed(self, settings, names):
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
        from forge.storage import SqliteStore

        store = SqliteStore(settings.db_path)
        store.initialize()
        for i, name in enumerate(names):
            store.put_proposal(
                Proposal(
                    id=f"d{i:04d}",
                    type=ProposalType.NEW_CONCEPT,
                    safety=SafetyClass.MODEL_GENERATED,
                    target_entity_type=EntityType.CONCEPT,
                    operation=ProposedOperation(action="create_concept", target=name),
                    reason="test",
                    evidence_span_ids=(f"sp{i:04d}",),
                    provenance=Provenance(
                        tier=ProvenanceTier.EXTRACTED_CLAIM,
                        derivation=Derivation.MODEL,
                        confidence=0.5,
                        agent="test/1",
                        model_id="test-model",
                    ),
                )
            )
        store.close()

    def test_an_empty_store_is_not_an_error(self, settings):
        result = runner.invoke(app, ["proposals", "duplicates"], env=self._env(settings))
        assert result.exit_code == 0
        assert "no duplicate clusters" in result.output

    def test_it_reports_an_alias_pair_and_suggests_a_survivor(self, settings):
        self._seed(settings, ["Reranking", "Reranker", "Binary Search"])
        result = runner.invoke(app, ["proposals", "duplicates"], env=self._env(settings))
        assert result.exit_code == 0
        assert "Reranking" in result.output
        assert "Binary Search" not in result.output, "a lone concept is not a cluster"

    def test_it_reports_but_never_decides(self, settings):
        """The command must not change any proposal's status."""
        from forge.domain import ProposalStatus
        from forge.storage import SqliteStore

        self._seed(settings, ["Reranking", "Reranker"])
        runner.invoke(app, ["proposals", "duplicates"], env=self._env(settings))
        store = SqliteStore(settings.db_path)
        store.initialize()
        statuses = {p.status for p in store.list_proposals()}
        store.close()
        assert statuses == {ProposalStatus.PENDING}

    def test_a_bad_status_filter_exits_two_without_a_traceback(self, settings):
        result = runner.invoke(
            app, ["proposals", "duplicates", "--status", "bogus"], env=self._env(settings)
        )
        assert result.exit_code == 2
        assert "Traceback" not in result.output

    def test_json_output_carries_the_dedup_version(self, settings):
        self._seed(settings, ["Reranking", "Reranker"])
        result = runner.invoke(
            app, ["proposals", "duplicates", "--json"], env=self._env(settings)
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["version"].startswith("dedup/")
        assert payload["redundant"] == 1


class TestStatusReportsModelIdentity:
    """`forge status` must show the identity extraction will actually cache under.

    FORGE_OLLAMA_THINK=0 appends "+nothink" to the model id, which is part of
    the derivation key. It was left exported after the 2026-08-19 experiment
    and silently governed a 5.7-hour extraction run, because nothing in the CLI
    displayed it.
    """

    def _env(self, settings, **extra):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
            **extra,
        }

    def test_identity_is_plain_when_no_mode_is_set(self, settings):
        env = self._env(settings, FORGE_LLM_PROVIDER="ollama")
        result = runner.invoke(app, ["status"], env=env)
        assert result.exit_code == 0
        assert "model identity :" in result.stdout
        assert "+nothink" not in result.stdout
        assert "reasoning is OFF" not in result.stdout

    def test_reasoning_off_is_shown_and_flagged(self, settings):
        env = self._env(settings, FORGE_LLM_PROVIDER="ollama", FORGE_OLLAMA_THINK="0")
        result = runner.invoke(app, ["status"], env=env)
        assert result.exit_code == 0
        assert "+nothink" in result.stdout
        assert "reasoning is OFF" in result.stdout

    def test_identity_is_in_the_json_payload(self, settings):
        env = self._env(settings, FORGE_LLM_PROVIDER="ollama", FORGE_OLLAMA_THINK="0")
        result = runner.invoke(app, ["status", "--json"], env=env)
        assert result.exit_code == 0
        assert json.loads(result.stdout)["llm"]["model_identity"].endswith("+nothink")
