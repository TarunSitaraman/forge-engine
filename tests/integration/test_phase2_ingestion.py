"""Phase 2 end-to-end: ingestion, change detection, cost control, retrieval, CLI.

These tests prove the Phase 2 exit criteria. Where a criterion is about cost or
about *not* doing something, it is asserted by counting, not by inspection.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.domain import (
    ExtractionStatus,
    IngestionStatus,
    MatchKind,
    ProposalStatus,
    ProposalType,
    SafetyClass,
    TrustTier,
)
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.llm.base import CALLS
from forge.proposals import ProposalService
from forge.retrieval import SearchQuery, SearchService

runner = CliRunner()


class TestDeterministicIngestion:
    def test_pdf_ingests_offline(self, pipeline, pdf_dir, store):
        """Exit criterion 2: PDF ingestion works with no model at all."""
        CALLS.reset()
        report = pipeline.ingest_path(pdf_dir / "multipage.pdf")
        source = report.sources[0]

        assert source.status is IngestionStatus.INGESTED
        assert source.spans == 3 and source.pages == 3
        assert CALLS.count == 0
        assert store.counts()["spans"] == 3

    def test_markdown_ingests_and_reuses_phase1_parsing(self, pipeline, fixture_vault, store):
        """Exit criterion 3."""
        report = pipeline.ingest_path(fixture_vault / "DSA/01_Patterns/Graph Traversal.md")
        assert report.sources[0].status is IngestionStatus.INGESTED
        assert report.sources[0].spans > 0
        assert report.sources[0].pages is None

    def test_directory_ingest_survives_bad_files(self, pipeline, pdf_dir):
        report = pipeline.ingest_path(pdf_dir)
        statuses = {s.locator.split("/")[-1]: s.status for s in report.sources}
        assert statuses["multipage.pdf"] is IngestionStatus.INGESTED
        assert statuses["image-only.pdf"] is IngestionStatus.OCR_REQUIRED
        assert statuses["malformed.pdf"] is IngestionStatus.PARSE_FAILED
        assert statuses["not-a-pdf.pdf"] is IngestionStatus.PARSE_FAILED

    def test_ocr_required_stores_nothing(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir / "image-only.pdf")
        assert store.counts()["sources"] == 0

    def test_external_pdfs_are_not_user_authored(self, pipeline, pdf_dir, store):
        """A downloaded PDF is not evidence of the same standing as your own note."""
        pipeline.ingest_path(pdf_dir / "simple.pdf")
        assert store.list_sources()[0].trust_tier is TrustTier.UNVERIFIED

    def test_vault_markdown_is_user_authored(self, pipeline, fixture_vault, store):
        pipeline.ingest_path(fixture_vault / "Notes/plain-note.md")
        assert store.list_sources()[0].trust_tier is TrustTier.USER_AUTHORED


class TestSpanTraceability:
    def test_every_span_resolves_to_its_source(self, pipeline, pdf_dir, store):
        """Exit criterion 4 + 10: spans point at real documents and sources."""
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        service = SearchService(store)
        for source in store.list_sources():
            for document in store.documents_for_source(source.id):
                for span in store.spans_for_document(document.id):
                    hit = service.span(span.id)
                    assert hit.document is not None and hit.source is not None
                    assert hit.citation.startswith(source.locator)

    def test_spans_carry_page_and_section(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        spans = SearchService(store).spans_for_source(
            store.list_sources()[0].locator
        )
        chunking = next(s for s in spans if "Chunking" in s.text)
        assert chunking.page == 2
        assert "Chunking Strategy" in chunking.heading_path
        assert "p.2" in chunking.citation()

    def test_span_ids_are_deterministic(self, settings, store, pdf_dir, tmp_path):
        from forge.storage import SqliteStore

        first = IngestionPipeline(settings, store)
        first.ingest_path(pdf_dir / "multipage.pdf")
        ids_a = sorted(s.id for s in _all_spans(store))

        other = SqliteStore(tmp_path / "second.db")
        other.initialize()
        IngestionPipeline(settings, other).ingest_path(pdf_dir / "multipage.pdf")
        ids_b = sorted(s.id for s in _all_spans(other))
        other.close()

        assert ids_a == ids_b


class TestChangeDetection:
    def test_unchanged_source_is_a_noop(self, pipeline, pdf_dir, store):
        """Exit criteria 5 + 6, and the headline cost-control property."""
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        before = store.counts()

        CALLS.reset()
        report = pipeline.ingest_path(pdf_dir / "multipage.pdf")

        assert report.sources[0].status is IngestionStatus.UNCHANGED
        assert CALLS.count == 0
        assert store.counts() == before, "no duplicate documents or spans"

    def test_unchanged_source_skips_extraction_entirely(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        extractor = scripted_extractor()
        pipe = IngestionPipeline(settings, store, extractor=extractor)
        options = IngestOptions(extract=True)

        first = pipe.ingest_path(pdf_dir / "multipage.pdf", options)
        assert first.llm_calls > 0

        CALLS.reset()
        second = pipe.ingest_path(pdf_dir / "multipage.pdf", options)
        assert second.llm_calls == 0
        assert CALLS.count == 0
        assert second.sources[0].extraction_status is ExtractionStatus.SKIPPED_CACHED

    def test_modified_source_is_reprocessed(self, pipeline, fixture_vault, store):
        target = fixture_vault / "Notes/plain-note.md"
        pipeline.ingest_path(target)
        original_hash = store.list_sources()[0].content_hash

        target.write_text(target.read_text() + "\n\n## New Section\n\nAdded content here.\n")
        report = pipeline.ingest_path(target)

        assert report.sources[0].status is IngestionStatus.INGESTED
        assert store.list_sources()[0].content_hash != original_hash

    def test_modification_preserves_history(self, pipeline, fixture_vault, store):
        """Exit criterion: modified sources keep their prior revision."""
        from forge.domain import EntityType, RevisionOp

        target = fixture_vault / "Notes/plain-note.md"
        pipeline.ingest_path(target)
        source_id = store.list_sources()[0].id

        target.write_text(target.read_text() + "\nmore\n")
        pipeline.ingest_path(target)

        revisions = store.revisions_for(EntityType.SOURCE, source_id)
        assert [r.op for r in revisions] == [RevisionOp.CREATE, RevisionOp.CHANGE]
        assert revisions[1].before["content_hash"] != revisions[1].after["content_hash"]

    def test_modification_creates_a_new_document_version(self, pipeline, fixture_vault, store):
        target = fixture_vault / "Notes/plain-note.md"
        pipeline.ingest_path(target)
        target.write_text(target.read_text() + "\nmore\n")
        pipeline.ingest_path(target)

        source = store.list_sources()[0]
        documents = store.documents_for_source(source.id)
        assert len(documents) == 2
        assert sorted(d.version for d in documents) == [1, 2]

    def test_force_reprocesses_unchanged_content(self, pipeline, pdf_dir):
        pipeline.ingest_path(pdf_dir / "simple.pdf")
        report = pipeline.ingest_path(pdf_dir / "simple.pdf", IngestOptions(force=True))
        assert report.sources[0].status is IngestionStatus.INGESTED

    def test_derivation_cache_invalidated_on_change(
        self, settings, store, fixture_vault, scripted_extractor
    ):
        pipe = IngestionPipeline(settings, store, extractor=scripted_extractor())
        target = fixture_vault / "Notes/plain-note.md"
        pipe.ingest_path(target, IngestOptions(extract=True))
        assert store.counts()["derivations"] >= 1

        target.write_text(target.read_text() + "\nchanged\n")
        report = pipe.ingest_path(target, IngestOptions(extract=True))
        assert report.llm_calls > 0, "changed content must be re-extracted"


class TestExtractionIntegration:
    def test_ingestion_succeeds_without_a_model(self, pipeline, pdf_dir):
        """Exit criterion 8: deterministic ingestion survives having no LLM.

        The extraction status is SKIPPED_NO_PROVIDER rather than SUCCEEDED —
        reporting "succeeded" when no model ran would overstate what happened.
        """
        report = pipeline.ingest_path(pdf_dir / "simple.pdf", IngestOptions(extract=True))
        assert report.sources[0].status is IngestionStatus.INGESTED
        assert report.sources[0].spans > 0
        assert report.sources[0].extraction_status is ExtractionStatus.SKIPPED_NO_PROVIDER
        assert report.sources[0].llm_calls == 0

    def test_extraction_produces_evidenced_proposals(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        """Exit criteria 9 + 10."""
        extractor = scripted_extractor(
            concepts=[{"name": "Chunking Strategy", "kind": "concept", "mention": "Chunk size"}],
            claims=[
                {
                    "statement": "Chunk size affects retrieval quality",
                    "evidence_quote": "Chunk size materially affects retrieval quality.",
                    "concept": "Chunking Strategy",
                }
            ],
        )
        pipe = IngestionPipeline(settings, store, extractor=extractor)
        pipe.ingest_path(pdf_dir / "multipage.pdf", IngestOptions(extract=True))

        service = ProposalService(store)
        claims = service.list(type=ProposalType.NEW_CLAIM)
        assert claims

        search = SearchService(store)
        for proposal in claims:
            assert proposal.evidence_span_ids
            assert proposal.provenance.model_id
            assert proposal.provenance.derivation.value == "model"
            assert proposal.safety is SafetyClass.MODEL_GENERATED
            # Evidence must resolve all the way back to a source.
            hit = search.span(proposal.evidence_span_ids[0])
            assert hit is not None and hit.source is not None

    def test_extraction_failure_does_not_lose_the_ingest(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        pipe = IngestionPipeline(settings, store, extractor=scripted_extractor(fail=True))
        report = pipe.ingest_path(pdf_dir / "multipage.pdf", IngestOptions(extract=True))

        assert report.sources[0].status is IngestionStatus.INGESTED
        assert report.sources[0].spans == 3, "spans survive an extraction failure"
        assert report.sources[0].extraction_status is ExtractionStatus.FAILED
        assert report.sources[0].concepts_proposed == 0

    def test_failed_extraction_is_not_cached(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        """Caching a failure would make the run permanently model-free."""
        pipe = IngestionPipeline(settings, store, extractor=scripted_extractor(fail=True))
        pipe.ingest_path(pdf_dir / "simple.pdf", IngestOptions(extract=True))
        assert store.counts()["derivations"] == 0


class TestOverlapAndAmbiguity:
    def test_second_overlapping_document_detects_existing_concepts(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        """Exit criterion 11, the Demo 3 scenario."""
        extractor = scripted_extractor(
            concepts=[{"name": "Hybrid Search", "kind": "concept", "mention": "Hybrid"}],
            claims=[],
        )
        pipe = IngestionPipeline(settings, store, extractor=extractor)
        options = IngestOptions(extract=True)

        pipe.ingest_path(pdf_dir / "multipage.pdf", options)
        pipe.ingest_path(pdf_dir / "overlapping.pdf", options)

        service = ProposalService(store)
        matches = service.list(type=ProposalType.CONCEPT_MATCH)
        assert matches, "the second document must recognise the first document's concepts"
        assert any(
            m.operation.details["match_kind"] == MatchKind.MATCH_CANDIDATE.value for m in matches
        )

    def test_ambiguous_concept_is_never_auto_selected(
        self, settings, store, pdf_dir, scripted_extractor
    ):
        """Exit criterion 11, the Demo 4 scenario."""
        extractor = scripted_extractor(
            concepts=[{"name": "Heap", "kind": "data_structure", "mention": "A heap"}], claims=[]
        )
        pipe = IngestionPipeline(settings, store, extractor=extractor)
        pipe.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))

        ambiguous = [
            p
            for p in ProposalService(store).list(limit=100)
            if p.operation.details.get("match_kind") == MatchKind.AMBIGUOUS.value
        ]
        assert ambiguous, "Heap collides in the vault and must surface as ambiguous"
        proposal = ambiguous[0]
        assert proposal.safety is SafetyClass.AMBIGUOUS
        assert proposal.operation.after is None, "no automatic selection"
        assert len(proposal.operation.details["candidates"]) == 2
        assert proposal.status is ProposalStatus.PENDING


class TestRetrievalIntegration:
    def test_lexical_search_returns_provenance(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir)
        hits = SearchService(store).search(SearchQuery(text="retrieval passages"))
        assert hits
        for hit in hits:
            assert hit.source is not None and hit.document is not None
            assert "::" in hit.citation

    def test_filters_narrow_results(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir)
        service = SearchService(store)
        assert service.search(SearchQuery(text="chunk", page=2))
        assert not service.search(SearchQuery(text="chunk", page=99))
        assert not service.search(SearchQuery(text="chunk", source_contains="nonexistent"))
        assert service.search(SearchQuery(text="chunk", source_kinds={"pdf"}))
        assert not service.search(SearchQuery(text="chunk", source_kinds={"markdown"}))

    def test_search_works_without_embeddings(self, pipeline, pdf_dir, store):
        """Exit criterion: documented degradation mode."""
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        service = SearchService(store)
        assert service.semantic_available is False
        assert "lexical retrieval only" in service.degradation_note()
        assert service.search(SearchQuery(text="chunking", semantic=True))

    def test_fts_special_characters_do_not_crash(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        service = SearchService(store)
        for query in ["RAG (retrieval)", 'quote"inside', "AND OR NOT", "*", ""]:
            service.search(SearchQuery(text=query))  # must not raise

    def test_search_index_survives_reingest(self, pipeline, pdf_dir, store):
        pipeline.ingest_path(pdf_dir / "multipage.pdf")
        pipeline.ingest_path(pdf_dir / "multipage.pdf", IngestOptions(force=True))
        hits = SearchService(store).search(SearchQuery(text="chunking"))
        assert len({h.span.id for h in hits}) == len(hits), "no duplicate index rows"


class TestVaultSafety:
    def test_ingestion_never_modifies_the_vault(self, pipeline, fixture_vault):
        """Exit criterion 14."""
        before = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}
        pipeline.ingest_path(fixture_vault, IngestOptions(extract=True))
        assert {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))} == before

    def test_real_corpus_untouched_by_phase2(self, real_settings, real_vault, tmp_path):
        """The real 621-file corpus must survive Phase 2 exactly as Phase 1 left it."""
        import subprocess

        from forge.storage import SqliteStore

        before = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout

        store = SqliteStore(tmp_path / "real.db")
        store.initialize()
        IngestionPipeline(real_settings, store).ingest_path(
            real_vault / "DSA" / "01_Patterns" / "DFS.md"
        )
        store.close()

        after = subprocess.run(
            ["git", "status", "--porcelain"], cwd=real_vault, capture_output=True, text=True
        ).stdout
        assert before == after


class TestPhase2Cli:
    def _env(self, settings):
        return {
            "FORGE_VAULT_PATH": str(settings.vault_path),
            "FORGE_STATE_DIR": str(settings.state_dir),
        }

    def test_ingest_reports_zero_llm_calls(self, settings, pdf_dir):
        result = runner.invoke(
            app, ["ingest", str(pdf_dir / "simple.pdf"), "--json"], env=self._env(settings)
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["totals"]["llm_calls"] == 0
        assert payload["totals"]["spans"] > 0

    def test_ingest_twice_is_unchanged(self, settings, pdf_dir):
        env = self._env(settings)
        runner.invoke(app, ["ingest", str(pdf_dir / "simple.pdf")], env=env)
        result = runner.invoke(app, ["ingest", str(pdf_dir / "simple.pdf"), "--json"], env=env)
        payload = json.loads(result.stdout)
        assert payload["by_status"] == {"unchanged": 1}
        assert payload["totals"]["llm_calls"] == 0

    def test_ingest_ocr_required_is_reported(self, settings, pdf_dir):
        result = runner.invoke(
            app, ["ingest", str(pdf_dir / "image-only.pdf"), "--json"], env=self._env(settings)
        )
        payload = json.loads(result.stdout)
        assert payload["sources"][0]["status"] == "ocr_required"

    def test_ingest_malformed_exits_nonzero(self, settings, pdf_dir):
        result = runner.invoke(
            app, ["ingest", str(pdf_dir / "malformed.pdf")], env=self._env(settings)
        )
        assert result.exit_code == 1

    def test_search_command(self, settings, pdf_dir):
        env = self._env(settings)
        runner.invoke(app, ["ingest", str(pdf_dir / "multipage.pdf")], env=env)
        result = runner.invoke(app, ["search", "chunking", "--json"], env=env)
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["hits"]
        assert payload["hits"][0]["citation"]
        assert payload["degradation"]

    def test_documents_command(self, settings, pdf_dir):
        env = self._env(settings)
        runner.invoke(app, ["ingest", str(pdf_dir / "multipage.pdf")], env=env)
        payload = json.loads(runner.invoke(app, ["documents", "--json"], env=env).stdout)
        assert payload[0]["spans"] == 3
        assert payload[0]["kind"] == "pdf"

    def test_concepts_command_explains_empty_state(self, settings):
        result = runner.invoke(app, ["concepts"], env=self._env(settings))
        assert result.exit_code == 0
        assert "proposes concepts" in result.stdout

    def test_proposals_workflow(self, settings):
        env = self._env(settings)
        generated = runner.invoke(app, ["proposals", "generate", "--json"], env=env)
        assert generated.exit_code == 0
        assert json.loads(generated.stdout)["created"] > 0

        listed = json.loads(
            runner.invoke(app, ["proposals", "list", "--json"], env=env).stdout
        )
        proposal_id = listed["proposals"][0]["id"]

        shown = runner.invoke(app, ["proposals", "show", proposal_id], env=env)
        assert shown.exit_code == 0 and "change:" in shown.stdout

        approved = runner.invoke(app, ["proposals", "approve", proposal_id], env=env)
        assert approved.exit_code == 0
        assert "NOT performed" in approved.stdout, "approval must not write by default"

    def test_proposals_approve_without_apply_leaves_vault_alone(self, settings, fixture_vault):
        env = self._env(settings)
        before = {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))}

        runner.invoke(app, ["proposals", "generate"], env=env)
        listed = json.loads(runner.invoke(app, ["proposals", "list", "--json"], env=env).stdout)
        runner.invoke(app, ["proposals", "approve", listed["proposals"][0]["id"]], env=env)

        assert {p: p.read_bytes() for p in sorted(fixture_vault.rglob("*.md"))} == before

    def test_proposals_apply_writes_with_backup(self, settings, fixture_vault):
        env = self._env(settings)
        runner.invoke(app, ["proposals", "generate"], env=env)
        listed = json.loads(runner.invoke(app, ["proposals", "list", "--json"], env=env).stdout)
        proposal = listed["proposals"][0]

        result = runner.invoke(
            app, ["proposals", "approve", proposal["id"], "--apply", "--json"], env=env
        )
        payload = json.loads(result.stdout)
        assert payload["apply"]["applied"] == 1

        target = fixture_vault / proposal["operation"]["target"]
        assert proposal["operation"]["after"] in target.read_text()
        assert list((settings.state_dir / "backups").rglob("*.md"))

    def test_proposals_reject(self, settings):
        env = self._env(settings)
        runner.invoke(app, ["proposals", "generate"], env=env)
        listed = json.loads(runner.invoke(app, ["proposals", "list", "--json"], env=env).stdout)
        pid = listed["proposals"][0]["id"]
        result = runner.invoke(app, ["proposals", "reject", pid, "--json"], env=env)
        assert json.loads(result.stdout)["status"] == "rejected"

    def test_unknown_proposal_exits_nonzero(self, settings):
        result = runner.invoke(app, ["proposals", "show", "nope"], env=self._env(settings))
        assert result.exit_code == 1

    def test_phase1_commands_still_work(self, settings):
        """Exit criterion 1, at the CLI level."""
        env = self._env(settings)
        for args in (["index", "--json"], ["status", "--json"], ["corpus-stats", "--json"]):
            result = runner.invoke(app, args, env=env)
            assert result.exit_code == 0, f"{args} regressed"


def _all_spans(store):
    spans = []
    for source in store.list_sources():
        for document in store.documents_for_source(source.id):
            spans.extend(store.spans_for_document(document.id))
    return spans
