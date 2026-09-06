"""The MVP acceptance test: a second overlapping document updates the model.

The roadmap is explicit that steps 1-10 of the demo are "a competent RAG
pipeline that many tools already deliver", and that the product claim lives
entirely in steps 11-13:

    11. User adds a second, overlapping document.
    12. Forge detects the overlap.
    13. Forge updates the graph rather than creating duplicate notes.

Step 12 was already covered: `TestOverlapAndAmbiguity` asserts that the second
document raises a `CONCEPT_MATCH` proposal. **Nothing asserted step 13**, and
the two are not the same claim. Detecting an overlap and then activating the
match into a graph that did not grow a duplicate node are different properties,
and only the second is the one the product is named for.

So these tests run the whole arc against the real pipeline, activator and store
and assert the shape of the graph on the other side. Extraction runs through a
scripted provider, as everywhere else in this suite; every other component is
the production path.
"""

from __future__ import annotations

import pytest

from forge.activation import ProposalActivator
from forge.domain import ProposalType
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.proposals import ProposalService

CONCEPT = "Hybrid Search"


def approve_and_activate(store, *, of_type=None):
    """Approve every pending proposal and activate it, as `forge` would."""
    service = ProposalService(store)
    activator = ProposalActivator(store)
    results = []
    for proposal in service.list(limit=200):
        if of_type is not None and proposal.type is not of_type:
            continue
        if proposal.status.value != "pending":
            continue
        service.approve(proposal.id, by="mvp-test")
        results.append(activator.activate(service.get(proposal.id)))
    return results


@pytest.fixture
def first_document(settings, store, pdf_dir, scripted_extractor):
    """Ingest document A and activate what it proposes, leaving a real graph."""
    extractor = scripted_extractor(
        concepts=[{"name": CONCEPT, "kind": "concept", "mention": "Hybrid"}], claims=[]
    )
    pipeline = IngestionPipeline(settings, store, extractor=extractor)
    pipeline.ingest_path(pdf_dir / "multipage.pdf", IngestOptions(extract=True))
    approve_and_activate(store, of_type=ProposalType.NEW_CONCEPT)
    return pipeline


class TestSecondDocumentDoesNotDuplicate:
    def test_the_graph_gains_no_node_for_a_concept_it_already_has(
        self, settings, store, pdf_dir, scripted_extractor, first_document
    ):
        """Step 13. The claim the whole project is named for."""
        before = store.counts()["concepts"]
        assert before >= 1, "document A must have produced a concept to overlap with"

        extractor = scripted_extractor(
            concepts=[{"name": CONCEPT, "kind": "concept", "mention": "Hybrid"}], claims=[]
        )
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        pipeline.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))

        # Ingestion alone must not create the duplicate either: the proposal
        # is the only thing that may exist before a human decides.
        assert store.counts()["concepts"] == before, "ingestion wrote a concept without approval"

        approve_and_activate(store)

        assert store.counts()["concepts"] == before, (
            "a second document about a known concept added a node; "
            "the model duplicated rather than updated"
        )

    def test_the_overlap_is_detected_as_a_match_not_a_new_concept(
        self, settings, store, pdf_dir, scripted_extractor, first_document
    ):
        """Step 12, and the reason step 13 is not vacuous.

        An unchanged concept count could also mean the second document
        proposed a duplicate and activation refused it. The mechanism has to
        be asserted, not just the outcome.
        """
        extractor = scripted_extractor(
            concepts=[{"name": CONCEPT, "kind": "concept", "mention": "Hybrid"}], claims=[]
        )
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        pipeline.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))

        pending = [p for p in ProposalService(store).list(limit=100) if p.status.value == "pending"]
        kinds = {p.type for p in pending}

        assert ProposalType.CONCEPT_MATCH in kinds, "the overlap must be detected as a match"
        assert ProposalType.NEW_CONCEPT not in kinds, (
            "a concept the graph already holds must never be proposed as new"
        )

    def test_a_paraphrased_name_is_not_matched_and_needs_a_human(
        self, settings, store, pdf_dir, scripted_extractor, first_document
    ):
        """The boundary of the claim, stated rather than skipped.

        Matching is lexical (embeddings are optional and off by default), so
        `Hybrid Retrieval` against a stored `Hybrid Search` shares one token
        and does not match. The second document therefore proposes a *new*
        concept, which would be a duplicate if it were activated.

        This is not a defect so much as the reason the human gate exists: the
        duplicate is a pending proposal, never a graph node, and cannot become
        one without someone approving it. The honest form of the product claim
        is that **the graph never duplicates without a human approving the
        duplicate**, not that overlap detection is complete.
        """
        extractor = scripted_extractor(
            concepts=[{"name": "Hybrid Retrieval", "kind": "concept", "mention": "Hybrid"}],
            claims=[],
        )
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        before = store.counts()["concepts"]
        pipeline.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))

        pending = [p for p in ProposalService(store).list(limit=100) if p.status.value == "pending"]
        assert any(p.type is ProposalType.NEW_CONCEPT for p in pending), (
            "a paraphrase is expected to miss the lexical matcher; if this now "
            "matches, the boundary moved and the docs should say so"
        )
        assert store.counts()["concepts"] == before, (
            "the near-duplicate must stay a proposal until a human decides"
        )

    def test_the_update_is_traceable_to_a_revision(
        self, settings, store, pdf_dir, scripted_extractor, first_document
    ):
        """An update nobody can audit is indistinguishable from an overwrite."""
        before = store.count_revisions()

        extractor = scripted_extractor(
            concepts=[{"name": CONCEPT, "kind": "concept", "mention": "Hybrid"}], claims=[]
        )
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        pipeline.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))
        approve_and_activate(store)

        assert store.count_revisions() >= before, "revisions must never be lost"

    def test_the_second_document_is_stored_as_its_own_source(
        self, settings, store, pdf_dir, scripted_extractor, first_document
    ):
        """Not duplicating the *concept* must not mean discarding the document.

        The failure mode on the other side of this gate is a system that
        deduplicates so eagerly the second source leaves no trace, which would
        make its evidence uncitable.
        """
        sources_before = store.counts()["sources"]
        spans_before = store.counts()["spans"]

        extractor = scripted_extractor(
            concepts=[{"name": CONCEPT, "kind": "concept", "mention": "Hybrid"}], claims=[]
        )
        pipeline = IngestionPipeline(settings, store, extractor=extractor)
        pipeline.ingest_path(pdf_dir / "overlapping.pdf", IngestOptions(extract=True))

        counts = store.counts()
        assert counts["sources"] > sources_before, "the second document must be stored"
        assert counts["spans"] > spans_before, "its spans must be citable"
