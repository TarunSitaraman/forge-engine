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
