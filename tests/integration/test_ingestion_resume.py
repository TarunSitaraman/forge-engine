"""Phase 5 gate: interrupting mid-ingestion and resuming does not duplicate work.

Re-running a *completed* ingest is already known idempotent
(`test_phase2_ingestion.py`). The untested case is the interrupted one: a run
that got partway through a directory, died, and was started again.

The pipeline has been wrong here before. `_ingest_one` carries a long comment
about a version that short-circuited on the content hash alone, which made
`--extract` a silent no-op and "skipped every source it had already *ingested*
rather than every source it had already *extracted*". That defect is fixed;
nothing was asserting it stays fixed.

The property these tests pin is the strong one: **a partial run followed by a
resume must leave the same store as one uninterrupted run.** Weaker phrasings
("no duplicate spans", "no crash") pass for systems that silently drop the
work they were interrupted during.
"""

from __future__ import annotations

import shutil

import pytest

from forge.config import Settings
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.llm.base import CALLS
from forge.proposals import ProposalService
from forge.storage import SqliteStore

DOCUMENTS = ("simple.pdf", "multipage.pdf", "overlapping.pdf")


@pytest.fixture
def corpus(tmp_path, pdf_dir):
    """A directory of three real PDFs, ingested as a unit."""
    root = tmp_path / "corpus"
    root.mkdir()
    for name in DOCUMENTS:
        shutil.copy(pdf_dir / name, root / name)
    return root


def fresh(tmp_path, fixture_vault, name):
    settings = Settings(vault_path=fixture_vault, state_dir=tmp_path / f"state-{name}")
    store = SqliteStore(tmp_path / f"{name}.db")
    store.initialize()
    return settings, store


class TestResumeLeavesTheSameStore:
    def test_partial_then_resume_matches_an_uninterrupted_run(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        """The gate, stated as an equality rather than an absence."""
        options = IngestOptions(extract=True)

        settings_a, store_a = fresh(tmp_path, fixture_vault, "whole")
        IngestionPipeline(
            settings_a, store_a, extractor=scripted_extractor()
        ).ingest_path(corpus, options)
        uninterrupted = store_a.counts()

        # The interrupted run: one document lands, then the process "dies".
        settings_b, store_b = fresh(tmp_path, fixture_vault, "resumed")
        IngestionPipeline(
            settings_b, store_b, extractor=scripted_extractor()
        ).ingest_path(corpus / DOCUMENTS[0], options)
        partial = store_b.counts()
        assert partial["sources"] == 1, "the partial run must really be partial"

        # A new process, a new pipeline, the same directory.
        IngestionPipeline(
            settings_b, store_b, extractor=scripted_extractor()
        ).ingest_path(corpus, options)

        assert store_b.counts() == uninterrupted, (
            "resuming produced a different store than running straight through"
        )

    def test_the_resumed_document_costs_no_model_calls(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        """Not duplicating *work*, not merely not duplicating rows."""
        options = IngestOptions(extract=True)
        settings, store = fresh(tmp_path, fixture_vault, "calls")

        IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus / DOCUMENTS[0], options)

        CALLS.reset()
        report = IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus / DOCUMENTS[0], options)

        assert report.llm_calls == 0, "re-ingesting an extracted document re-ran the model"

    def test_resuming_does_not_duplicate_proposals(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        """A duplicated proposal is a duplicate a human has to reject by hand."""
        options = IngestOptions(extract=True)
        settings, store = fresh(tmp_path, fixture_vault, "proposals")

        IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus, options)
        before = len(ProposalService(store).list(limit=500))

        IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus, options)

        assert len(ProposalService(store).list(limit=500)) == before


class TestResumeDoesNotSkipUnfinishedWork:
    def test_a_source_ingested_without_extraction_is_still_extracted_later(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        """The defect `_ingest_one` documents, pinned.

        Ingesting deterministically first and extracting later is the normal
        order, since extraction is opt-in. A resume that treats "source stored"
        as "source done" makes the later `--extract` a silent no-op.
        """
        settings, store = fresh(tmp_path, fixture_vault, "twophase")

        IngestionPipeline(settings, store).ingest_path(corpus, IngestOptions(extract=False))
        assert store.counts()["sources"] == len(DOCUMENTS)
        assert len(ProposalService(store).list(limit=500)) == 0, "nothing extracted yet"

        report = IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus, IngestOptions(extract=True))

        assert report.llm_calls > 0, "the extract pass was skipped as already done"
        assert ProposalService(store).list(limit=500), "extraction produced nothing"

    def test_extraction_over_stored_spans_does_not_rechunk(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        """Re-deriving the document would bump its version and duplicate spans."""
        settings, store = fresh(tmp_path, fixture_vault, "spans")

        IngestionPipeline(settings, store).ingest_path(corpus, IngestOptions(extract=False))
        spans_before = store.counts()["spans"]
        documents_before = store.counts()["documents"]

        IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus, IngestOptions(extract=True))

        assert store.counts()["spans"] == spans_before, "extraction re-chunked the corpus"
        assert store.counts()["documents"] == documents_before, "extraction bumped the version"


class TestASourceNeverMatchesItself:
    """The defect the equality test caught, pinned directly.

    Extraction is cached, so a resumed run re-derives nothing, but it still
    reaches `_propose` — and by then the matcher has learned about the
    proposals the *first* run made. A source that proposed `Test Concept` as
    NEW_CONCEPT came back and raised a CONCEPT_MATCH against its own earlier
    proposal, leaving one source holding two live proposals for one name.

    The store-equality test above would catch this again, but only as a number.
    This says what went wrong.
    """

    def test_one_source_never_holds_two_live_proposals_for_one_name(
        self, tmp_path, fixture_vault, corpus, scripted_extractor
    ):
        from forge.domain import ProposalStatus, ProposalType

        options = IngestOptions(extract=True)
        settings, store = fresh(tmp_path, fixture_vault, "selfmatch")

        pipeline = IngestionPipeline(settings, store, extractor=scripted_extractor())
        pipeline.ingest_path(corpus / DOCUMENTS[0], options)
        IngestionPipeline(
            settings, store, extractor=scripted_extractor()
        ).ingest_path(corpus, options)

        by_source: dict[tuple[str, str], list[str]] = {}
        for proposal in store.list_proposals(limit=500):
            if proposal.type not in (ProposalType.NEW_CONCEPT, ProposalType.CONCEPT_MATCH):
                continue
            if proposal.status is ProposalStatus.REJECTED:
                continue
            key = (proposal.source_id or "", proposal.operation.target.casefold())
            by_source.setdefault(key, []).append(proposal.type.value)

        collisions = {k: v for k, v in by_source.items() if len(v) > 1}
        assert not collisions, (
            f"a source proposed the same name twice: {collisions}"
        )
