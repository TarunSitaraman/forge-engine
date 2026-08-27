"""Proposal lifecycle, safety classification, matching, and write-back.

These tests protect the property Phase 2 exists to guarantee: **nothing the
engine believes reaches the user's files without an explicit human decision.**
"""

from __future__ import annotations

import pytest

from forge.domain import (
    Concept,
    ConceptKind,
    Derivation,
    EntityType,
    MatchKind,
    Proposal,
    ProposalStatus,
    ProposalTransitionError,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    ProvenanceViolation,
    RevisionOp,
    SafetyClass,
    deterministic_provenance,
)
from forge.matching import ConceptMatcher, build_ambiguity_index
from forge.proposals import ProposalApplier, ProposalService, build_repair_proposals


def det_proposal(pid: str = "p1", **kw) -> Proposal:
    defaults = dict(
        id=pid,
        type=ProposalType.METADATA_REPAIR,
        safety=SafetyClass.DETERMINISTIC_VERIFIED,
        operation=ProposedOperation(
            action="replace_frontmatter_line",
            target="a.md",
            before="related: [[A]], [[B]]",
            after='related: ["A", "B"]',
            details={"line": 3},
        ),
        reason="quoting preserves the links",
        provenance=deterministic_provenance("test", ProvenanceTier.USER_ASSERTION),
    )
    defaults.update(kw)
    return Proposal(**defaults)


def model_provenance() -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE, derivation=Derivation.MODEL, agent="t", model_id="m1"
    )


class TestLifecycle:
    def test_starts_pending(self):
        assert det_proposal().status is ProposalStatus.PENDING

    def test_approve_records_a_decision(self):
        approved = det_proposal().approve(by="alice", note="checked")
        assert approved.status is ProposalStatus.APPROVED
        assert approved.decided_at is not None
        assert approved.decided_by == "alice" and approved.decision_note == "checked"

    def test_reject_records_a_decision(self):
        rejected = det_proposal().reject(note="not wanted")
        assert rejected.status is ProposalStatus.REJECTED
        assert rejected.decided_at is not None

    @pytest.mark.parametrize("first", ["approve", "reject"])
    def test_cannot_decide_twice(self, first):
        decided = getattr(det_proposal(), first)()
        with pytest.raises(ProposalTransitionError):
            getattr(decided, first)()

    def test_approved_can_be_superseded(self):
        superseded = det_proposal().approve().supersede("p2")
        assert superseded.status is ProposalStatus.SUPERSEDED
        assert superseded.superseded_by == "p2"

    def test_superseded_is_terminal(self):
        superseded = det_proposal().supersede("p2")
        with pytest.raises(ProposalTransitionError):
            superseded.approve()

    def test_identity_is_deterministic(self):
        a = Proposal.make_id(ProposalType.METADATA_REPAIR, "a.md", "fp")
        b = Proposal.make_id(ProposalType.METADATA_REPAIR, "a.md", "fp")
        assert a == b
        assert a != Proposal.make_id(ProposalType.NEW_CONCEPT, "a.md", "fp")


class TestSafetyClassification:
    def test_model_cannot_claim_deterministic_verified(self):
        with pytest.raises(ProvenanceViolation, match="not verification"):
            det_proposal(
                safety=SafetyClass.DETERMINISTIC_VERIFIED,
                provenance=model_provenance(),
                evidence_span_ids=("s1",),
            )

    def test_model_proposal_must_cite_evidence(self):
        with pytest.raises(ProvenanceViolation, match="cites no evidence"):
            det_proposal(safety=SafetyClass.MODEL_GENERATED, provenance=model_provenance())

    def test_only_approved_verified_proposals_are_auto_applicable(self):
        assert det_proposal().auto_applicable is False
        assert det_proposal().approve().auto_applicable is True
        assert (
            det_proposal(
                safety=SafetyClass.MODEL_GENERATED,
                provenance=model_provenance(),
                evidence_span_ids=("s1",),
            )
            .approve()
            .auto_applicable
            is False
        )


class TestService:
    def test_create_and_retrieve(self, store):
        service = ProposalService(store)
        proposal, created = service.create(det_proposal())
        assert created is True
        assert service.get(proposal.id).id == proposal.id

    def test_create_is_idempotent(self, store):
        service = ProposalService(store)
        service.create(det_proposal())
        _, created = service.create(det_proposal())
        assert created is False

    def test_rejected_proposals_are_not_resurrected(self, store):
        """Re-running ingestion must not reopen a decision the user made."""
        service = ProposalService(store)
        service.create(det_proposal())
        service.reject("p1", note="no thanks")

        again, created = service.create(det_proposal())
        assert created is False
        assert again.status is ProposalStatus.REJECTED

    def test_status_change_writes_a_revision(self, store):
        service = ProposalService(store)
        service.create(det_proposal())
        service.approve("p1")
        revisions = store.revisions_for(EntityType.PROPOSAL, "p1")
        assert [r.op for r in revisions] == [RevisionOp.CREATE, RevisionOp.CHANGE]
        assert "pending -> approved" in (revisions[1].note or "")

    def test_filtering(self, store):
        service = ProposalService(store)
        service.create(det_proposal("p1"))
        service.create(
            det_proposal(
                "p2",
                type=ProposalType.NEW_CONCEPT,
                safety=SafetyClass.MODEL_GENERATED,
                provenance=model_provenance(),
                evidence_span_ids=("s1",),
            )
        )
        service.approve("p1")
        assert len(service.list(status=ProposalStatus.APPROVED)) == 1
        assert len(service.list(type=ProposalType.NEW_CONCEPT)) == 1
        assert service.counts() == {"approved": 1, "pending": 1}

    def test_abbreviated_id_resolves(self, store):
        service = ProposalService(store)
        service.create(det_proposal("abcdef123456"))
        assert service.get("abcdef") is not None

    def test_ambiguous_abbreviation_resolves_to_nothing(self, store):
        service = ProposalService(store)
        service.create(det_proposal("abc111"))
        service.create(
            det_proposal(
                "abc222",
                operation=ProposedOperation(action="replace_frontmatter_line", target="b.md", details={"line": 1}),
            )
        )
        resolved, ambiguous = service.resolve("abc")
        assert resolved is None and len(ambiguous) == 2
        with pytest.raises(KeyError, match="matches 2 proposals"):
            service.approve("abc")

    def test_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            ProposalService(store).approve("nonexistent")


class TestMetadataRepairBridge:
    def test_builds_one_proposal_per_repairable_line(self, indexer):
        index = indexer.build_index()
        proposals = list(build_repair_proposals(index.files))
        expected = sum(len(f.repairs) for f in index.files)
        assert len(proposals) == expected > 0

    def test_verified_repairs_are_classified_verified(self, indexer):
        proposals = list(build_repair_proposals(indexer.build_index().files))
        assert all(p.safety is SafetyClass.DETERMINISTIC_VERIFIED for p in proposals)

    def test_repairs_are_deterministic_not_model_generated(self, indexer):
        proposals = list(build_repair_proposals(indexer.build_index().files))
        assert all(p.provenance.derivation is Derivation.DETERMINISTIC for p in proposals)
        assert all(p.provenance.model_id is None for p in proposals)

    def test_proposal_shows_affected_links(self, indexer):
        proposals = list(build_repair_proposals(indexer.build_index().files))
        related = [p for p in proposals if p.operation.details["key"] == "related"]
        assert related and related[0].operation.details["affected_links"]

    def test_generation_is_stable_across_runs(self, indexer):
        first = [p.id for p in build_repair_proposals(indexer.build_index().files)]
        second = [p.id for p in build_repair_proposals(indexer.build_index().files)]
        assert first == second


class TestConceptMatching:
    def concept(self, name: str, **kw) -> Concept:
        return Concept(
            id=Concept.make_id(name),
            canonical_name=name,
            kind=ConceptKind.CONCEPT,
            provenance=deterministic_provenance("t", ProvenanceTier.USER_ASSERTION),
            **kw,
        )

    @pytest.mark.parametrize("name", ["Heap", "Binary Search", "Trie"])
    def test_known_collisions_are_never_merged(self, name):
        """The three real vault collisions must always stay ambiguous."""
        index = build_ambiguity_index(
            [
                "DSA/01_Patterns/Heap.md",
                "DSA/03_DataStructures/Heap.md",
                "DSA/01_Patterns/Binary Search.md",
                "DSA/02_Algorithms/Binary Search.md",
                "DSA/01_Patterns/Trie.md",
                "DSA/03_DataStructures/Trie.md",
            ]
        )
        result = ConceptMatcher([], ambiguity_index=index).match(name)
        assert result.kind is MatchKind.AMBIGUOUS
        assert len(result.candidates) == 2
        assert result.best is None, "an ambiguous match must expose no winner"

    def test_ambiguity_beats_an_exact_concept_match(self):
        """Even a stored concept cannot resolve a vault-level collision."""
        index = build_ambiguity_index(
            ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]
        )
        matcher = ConceptMatcher([self.concept("Heap")], ambiguity_index=index)
        assert matcher.match("Heap").kind is MatchKind.AMBIGUOUS

    def test_structural_filenames_are_not_collisions(self):
        index = build_ambiguity_index(["a/_index.md", "b/_index.md", "c/README.md", "d/README.md"])
        assert index == {}

    def test_exact_match(self):
        matcher = ConceptMatcher([self.concept("Retrieval Augmented Generation")])
        result = matcher.match("Retrieval Augmented Generation")
        assert result.kind is MatchKind.MATCH_CANDIDATE
        assert result.best.signal == "exact"

    def test_alias_match(self):
        matcher = ConceptMatcher([self.concept("Retrieval Augmented Generation", aliases=("RAG",))])
        assert matcher.match("RAG").best.signal == "alias"

    def test_normalized_match(self):
        matcher = ConceptMatcher([self.concept("Graph Traversal")])
        assert matcher.match("graph-traversal").best.signal == "normalized"

    def test_lexical_match(self):
        matcher = ConceptMatcher([self.concept("Sliding Window")])
        result = matcher.match("Sliding Windows")
        assert result.kind is MatchKind.MATCH_CANDIDATE
        assert result.best.signal == "lexical"

    def test_unrelated_name_is_new(self):
        matcher = ConceptMatcher([self.concept("Sliding Window")])
        assert matcher.match("Byzantine Fault Tolerance").kind is MatchKind.NEW_CONCEPT

    def test_previously_proposed_name_is_not_new(self):
        """Without this, every source rediscovers the same concept as brand new."""
        matcher = ConceptMatcher([], proposed_concepts=[("Hybrid Search", "prop123")])
        result = matcher.match("Hybrid Search")
        assert result.kind is MatchKind.MATCH_CANDIDATE
        assert result.best.signal == "proposed"

    def test_near_ties_are_ambiguous_not_a_coin_flip(self):
        matcher = ConceptMatcher([self.concept("Concept Alpha"), self.concept("Concept Alpah")])
        result = matcher.match("Concept Alpza")
        assert result.kind is MatchKind.AMBIGUOUS
        assert result.best is None

    def test_works_without_embeddings(self):
        matcher = ConceptMatcher([self.concept("Sliding Window")])
        assert matcher.embeddings_available is False
        assert matcher.match("Sliding Window").kind is MatchKind.MATCH_CANDIDATE

    def test_empty_name_is_new(self):
        assert ConceptMatcher([]).match("   ").kind is MatchKind.NEW_CONCEPT


class TestWriteBack:
    def _setup(self, fixture_vault, store, tmp_path):
        from forge.corpus.indexer import CorpusIndexer
        from forge.config import Settings

        settings = Settings(vault_path=fixture_vault, state_dir=tmp_path / "st")
        index = CorpusIndexer(settings).build_index()
        service = ProposalService(store)
        proposals = list(build_repair_proposals(index.files))
        service.create_many(proposals)
        applier = ProposalApplier(fixture_vault, store, tmp_path / "backups")
        return service, applier, proposals

    def test_dry_run_writes_nothing(self, fixture_vault, store, tmp_path):
        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        target = fixture_vault / proposals[0].operation.target
        before = target.read_bytes()

        approved = service.approve(proposals[0].id)
        report = applier.apply([approved], apply=False)

        assert report.dry_run is True
        assert target.read_bytes() == before
        assert report.applied == []

    def test_apply_writes_and_backs_up(self, fixture_vault, store, tmp_path):
        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        approved = service.approve(proposals[0].id)
        target = fixture_vault / approved.operation.target
        original = target.read_text()

        report = applier.apply([approved], apply=True)

        assert len(report.applied) == 1
        assert target.read_text() != original
        assert approved.operation.after in target.read_text()
        backup = tmp_path / "backups"
        assert list(backup.rglob("*.md")), "a backup must exist before any write"

    def test_apply_records_a_revision(self, fixture_vault, store, tmp_path):
        from forge.config import Settings
        from forge.ingestion import IngestionPipeline

        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        settings = Settings(vault_path=fixture_vault, state_dir=tmp_path / "st")
        IngestionPipeline(settings, store).ingest_path(fixture_vault)

        approved = service.approve(proposals[0].id)
        applier.apply([approved], apply=True)

        source = store.get_source_by_locator(approved.operation.target)
        revisions = store.revisions_for(EntityType.SOURCE, source.id)
        applied = [r for r in revisions if r.cause == approved.id]
        assert applied and applied[0].before != applied[0].after

    def test_unapproved_proposals_are_refused(self, fixture_vault, store, tmp_path):
        _, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        report = applier.apply([proposals[0]], apply=True)
        assert report.applied == []
        assert "not approved" in report.refused[0].refused_reason

    def test_model_generated_proposals_are_refused(self, fixture_vault, store, tmp_path):
        _, applier, _ = self._setup(fixture_vault, store, tmp_path)
        risky = det_proposal(
            safety=SafetyClass.MODEL_GENERATED,
            provenance=model_provenance(),
            evidence_span_ids=("s1",),
        ).approve()
        report = applier.apply([risky], apply=True)
        assert report.applied == []
        assert "not automatically applicable" in report.refused[0].refused_reason

    def test_stale_proposal_is_refused(self, fixture_vault, store, tmp_path):
        """If the file changed since the proposal, applying could clobber an edit."""
        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        approved = service.approve(proposals[0].id)
        target = fixture_vault / approved.operation.target
        lines = target.read_text().split("\n")
        lines[approved.operation.details["line"] - 1] = "related: [[Something Else]]"
        target.write_text("\n".join(lines))

        report = applier.apply([approved], apply=True)
        assert report.applied == []
        assert "has changed since" in report.refused[0].refused_reason

    def test_only_the_named_file_is_touched(self, fixture_vault, store, tmp_path):
        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        approved = service.approve(proposals[0].id)
        others = {
            p: p.read_bytes()
            for p in fixture_vault.rglob("*.md")
            if p != fixture_vault / approved.operation.target
        }

        applier.apply([approved], apply=True)

        assert {p: p.read_bytes() for p in others} == others

    def test_missing_file_is_refused(self, fixture_vault, store, tmp_path):
        service, applier, proposals = self._setup(fixture_vault, store, tmp_path)
        approved = service.approve(proposals[0].id)
        (fixture_vault / approved.operation.target).unlink()
        report = applier.apply([approved], apply=True)
        assert "no longer exists" in report.refused[0].refused_reason


class TestBulkApply:
    """`approve-all --apply` must go through the same gate as single apply.

    Phase 1 of the direction plan is 283 identical repairs. Without a batched
    path that is 283 invocations; with one that skips the applier it would be
    283 unguarded writes. It uses ProposalApplier either way, so refusals and
    per-file backups behave the same in bulk as they do singly.
    """

    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def test_approve_all_without_apply_leaves_the_vault_untouched(self, settings, fixture_vault):
        from typer.testing import CliRunner

        from forge.cli.main import app

        runner = CliRunner()
        env = self._env(settings)
        runner.invoke(app, ["index"], env=env)
        runner.invoke(app, ["proposals", "generate"], env=env)

        before = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}
        result = runner.invoke(
            app,
            ["proposals", "approve-all", "--no-dry-run"],
            env=env,
        )
        assert result.exit_code == 0
        assert {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))} == before

    def test_approve_all_with_apply_writes_and_backs_up(self, settings, fixture_vault):
        from typer.testing import CliRunner

        from forge.cli.main import app

        runner = CliRunner()
        env = self._env(settings)
        runner.invoke(app, ["index"], env=env)
        gen = runner.invoke(app, ["proposals", "generate", "--json"], env=env)
        import json as _json

        if _json.loads(gen.stdout)["built"] == 0:
            pytest.skip("fixture vault has no repairable frontmatter")

        before = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}
        result = runner.invoke(
            app,
            ["proposals", "approve-all", "--no-dry-run", "--apply"],
            env=env,
        )
        assert result.exit_code == 0

        after = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}
        assert after != before, "--apply must actually write"
        assert (settings.state_dir / "backups").exists(), "every write needs a backup"

    def test_applying_twice_is_a_no_op(self, settings, fixture_vault):
        """The repaired form is a fixed point, so a second pass finds nothing."""
        from typer.testing import CliRunner

        from forge.cli.main import app

        runner = CliRunner()
        env = self._env(settings)
        runner.invoke(app, ["index"], env=env)
        runner.invoke(app, ["proposals", "generate"], env=env)
        runner.invoke(app, ["proposals", "approve-all", "--no-dry-run", "--apply"], env=env)

        settled = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}
        runner.invoke(app, ["index"], env=env)
        second = runner.invoke(app, ["proposals", "generate", "--json"], env=env)
        import json as _json

        assert _json.loads(second.stdout)["built"] == 0
        assert {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))} == settled
