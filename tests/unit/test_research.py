"""Phase 9: the six vision questions, gap detection, and synthesis staleness.

The vision calls its six questions "the product's real acceptance criteria",
and says they are "deliberately not answerable by a search box". Each has a
test named for it here.

Two of the three phase gates are closable in code and are closed here. The
third, "gap detection produces findings a human agrees are real gaps", needs a
human: what these tests can establish is that each rule fires on exactly the
structure it claims to detect and stays silent otherwise. Whether the findings
matter is not a property of the code.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from forge.domain import (
    Claim,
    ClaimLink,
    ClaimStatus,
    Concept,
    ConceptKind,
    Derivation,
    Document,
    EvidenceLink,
    EvidenceRelation,
    GapKind,
    LinkType,
    Provenance,
    ProvenanceTier,
    ProvenanceViolation,
    Question,
    QuestionStatus,
    Source,
    SourceKind,
    Span,
    Synthesis,
    SynthesisScope,
    TrustTier,
    utc_now,
)
from forge.research import (
    belief_for_concept,
    changes_since,
    detect_gaps,
    evidence_for_question,
    gap_report,
    open_questions,
)
from forge.storage import SqliteStore
from forge.storage.sqlite_store import SCHEMA_VERSION, claim_fingerprint

HUMAN = Provenance(
    tier=ProvenanceTier.USER_ASSERTION, derivation=Derivation.HUMAN, agent="test"
)
DETERMINISTIC = Provenance(
    tier=ProvenanceTier.USER_ASSERTION, derivation=Derivation.DETERMINISTIC, agent="bootstrap"
)
MODEL = Provenance(
    tier=ProvenanceTier.EXTRACTED_CLAIM,
    derivation=Derivation.MODEL,
    agent="extractor",
    model_id="test-model",
)
SYNTH = Provenance(
    tier=ProvenanceTier.SYNTHESIS, derivation=Derivation.MODEL, agent="synth", model_id="m"
)


class Vault:
    """A tiny corpus builder, so each test says only what it is about."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store
        self._spans = 0

    def concept(self, name: str, *, path: str | None = None) -> Concept:
        concept = Concept(
            id=Concept.make_id(name),
            canonical_name=name,
            kind=ConceptKind.TECHNOLOGY,
            vault_path=path,
            provenance=DETERMINISTIC,
        )
        self.store.put_concept(concept)
        return concept

    def source(self, locator: str) -> tuple[Source, Document]:
        source = Source.for_path(
            locator,
            kind=SourceKind.MARKDOWN,
            content_hash=locator,
            trust_tier=TrustTier.USER_AUTHORED,
        )
        self.store.put_source(source)
        document = Document(
            id=Document.make_id(source.id, locator),
            source_id=source.id,
            parser="test",
            parser_version="1",
            content_hash=locator,
        )
        self.store.put_document(document)
        return source, document

    def span(self, document: Document, text: str) -> Span:
        self._spans += 1
        span = Span(
            id=Span.make_id(document.id, self._spans, f"L{self._spans}"),
            document_id=document.id,
            ordinal=self._spans,
            locator=f"L{self._spans}",
            start_line=self._spans,
            end_line=self._spans,
            text=text,
            content_hash=f"span-{self._spans}",
        )
        self.store.put_spans([span])
        self.store.rebuild_search_index()
        return span

    def dispute(self, claim: Claim) -> Claim:
        """Mark a claim disputed *and* record the revision, as activation does.

        `put_claim` records a revision only on create: the activation layer
        owns the "why" and writes the CHANGE itself, so having the store write
        one too would double-record every approved change. A test that wants
        the change report to see a dispute therefore has to do what the real
        path does. See `forge.research.changes` for the caveat this implies.
        """
        from forge.domain import EntityType, record_change

        disputed = self.restate(claim, status=ClaimStatus.DISPUTED)
        self.store.append_revision(
            record_change(
                EntityType.CLAIM,
                claim.id,
                {"status": claim.status.value},
                {"status": disputed.status.value},
                cause="test-dispute",
            )
        )
        return disputed

    def restate(self, claim: Claim, **updates) -> Claim:
        """Rewrite a stored claim, re-supplying its evidence.

        `validate_claim` refuses to store an evidence-requiring claim without
        evidence, whatever the caller is doing, so changing a status means
        handing the links back. That is the real contract and the tests use it
        rather than a shortcut around it.
        """
        updated = claim.model_copy(update=updates)
        self.store.put_claim(updated, list(self.store.evidence_for_claim(claim.id)))
        return updated

    def claim(
        self,
        statement: str,
        concept: Concept,
        spans: list[Span],
        *,
        provenance: Provenance = MODEL,
        status: ClaimStatus = ClaimStatus.ACTIVE,
        superseded_by: str | None = None,
    ) -> Claim:
        claim = Claim(
            id=Claim.make_id(statement, spans[0].id if spans else statement),
            statement=statement,
            subject_concept_id=concept.id,
            provenance=provenance,
            status=status,
            # The domain refuses a SUPERSEDED claim that does not say what
            # replaced it, which is Principle 11: nothing is retired without a
            # record of what took its place.
            superseded_by=superseded_by or ("replacement" if status is ClaimStatus.SUPERSEDED else None),
        )
        self.store.put_claim(
            claim,
            [
                EvidenceLink(
                    id=EvidenceLink.make_id(claim.id, s.id, EvidenceRelation.PARAPHRASES),
                    claim_id=claim.id,
                    span_id=s.id,
                    relation=EvidenceRelation.PARAPHRASES,
                    provenance=provenance,
                )
                for s in spans
            ],
        )
        return claim

    def question(self, text: str, *, concepts: tuple[str, ...] = ()) -> Question:
        question = Question(
            id=Question.make_id(text), text=text, provenance=HUMAN, concept_ids=concepts
        )
        self.store.put_question(question)
        return question


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    store = SqliteStore(tmp_path / "forge.db")
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def vault(store: SqliteStore) -> Vault:
    return Vault(store)


# -- the six vision questions ----------------------------------------------


def test_question_one_what_do_i_currently_believe_about_x(vault: Vault):
    """Held claims, separated from what is no longer held."""
    rag = vault.concept("RAG", path="Technologies/Docs/rag.md")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds generation in retrieved passages.")
    held = vault.claim("RAG reduces hallucination.", rag, [span])
    vault.claim("RAG is only for chatbots.", rag, [span], status=ClaimStatus.SUPERSEDED,
                superseded_by="whatever-replaced-it")

    belief = belief_for_concept(vault.store, rag.id)

    assert [c.id for c in belief.held] == [held.id]
    assert len(belief.superseded) == 1
    assert belief.supporting_sources == ["rag.md"]


def test_question_two_which_sources_support_and_which_disagree(vault: Vault):
    """Dissent is reported as the three real things it can be.

    Forge has no `CONTRADICTS` edge on purpose: Phase 4 produces
    `POTENTIAL_CONFLICT` and routes it to a human rather than asserting a
    contradiction a model detected. So dissent is a disputed claim, a
    superseded one, or a conflict awaiting review, and never a verdict.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds generation.")
    vault.claim("RAG reduces hallucination.", rag, [span])
    vault.claim("RAG always helps.", rag, [span], status=ClaimStatus.DISPUTED)
    vault.claim("RAG is only for chatbots.", rag, [span], status=ClaimStatus.SUPERSEDED,
                superseded_by="whatever-replaced-it")

    belief = belief_for_concept(vault.store, rag.id)

    kinds = {d.kind for d in belief.dissent}
    assert kinds == {"disputed_claim", "superseded_claim"}
    assert not belief.is_settled
    assert belief.supporting_sources == ["rag.md"], "only held claims contribute sources"


def test_a_belief_with_nothing_against_it_is_settled(vault: Vault):
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])

    assert belief_for_concept(vault.store, rag.id).is_settled


def test_a_held_claim_with_no_evidence_is_reported(vault: Vault):
    """Only USER_ASSERTION may stand without evidence; anything else is a defect."""
    rag = vault.concept("RAG")
    asserted = vault.claim("I think RAG helps.", rag, [], provenance=HUMAN)

    belief = belief_for_concept(vault.store, rag.id)

    assert belief.unevidenced == [asserted.id]


def test_question_three_what_changed_in_my_understanding_this_month(vault: Vault):
    """Read from the revision log, so movement is visible, not just endpoints.

    A claim created and then disputed inside the window is two facts about how
    understanding moved. A diff of the endpoints would show one.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds generation.")
    claim = vault.claim("RAG reduces hallucination.", rag, [span])
    vault.dispute(claim)

    report = changes_since(vault.store, days=30)

    assert claim.id in report.claims_created
    assert claim.id in report.claims_disputed
    assert rag.id in report.concepts_created
    assert report.by_entity["Claim"]["create"] == 1


def test_a_change_window_excludes_what_falls_outside_it(vault: Vault):
    vault.concept("RAG")
    future = utc_now() + timedelta(days=2)
    report = changes_since(vault.store, since=future, until=future + timedelta(days=1))
    assert report.total == 0


def test_question_four_what_questions_remain_unanswered(vault: Vault):
    open_one = vault.question("What is agent memory?")
    answered = vault.question("What is RAG?")
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG retrieves.", rag, [vault.span(doc, "Retrieval grounds.")])
    vault.store.link_answer(answered.id, claim.id)
    vault.store.put_question(
        answered.model_copy(
            update={"status": QuestionStatus.ANSWERED, "resolved_at": utc_now()}
        )
    )

    assert [q.id for q in open_questions(vault.store)] == [open_one.id]


def test_question_five_what_concepts_am_i_missing(vault: Vault):
    """Gap detection, on a corpus built to contain exactly one of each finding."""
    barren = vault.concept("Vector Databases", path="Technologies/Docs/vdb.md")
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds generation.")
    vault.claim("RAG reduces hallucination.", rag, [span])
    vault.question("What is agent memory?")

    gaps = detect_gaps(vault.store, limit=100)
    by_kind = {g.kind: g for g in gaps}

    assert by_kind[GapKind.CONCEPT_WITHOUT_CLAIMS].subject_id == barren.id
    assert GapKind.QUESTION_WITHOUT_ANSWERS in by_kind
    assert GapKind.CLAIM_WITH_SINGLE_SOURCE in by_kind
    assert GapKind.ISOLATED_CONCEPT in by_kind


def test_question_six_which_evidence_is_relevant_to_an_unresolved_question(vault: Vault):
    """Scoped by the question, and excluding what the answer already cites.

    Returning evidence already read into the answer would make this a search
    box with extra steps.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    cited = vault.span(doc, "Retrieval grounds generation in retrieved passages.")
    uncited = vault.span(doc, "Hallucination falls when generation is grounded.")
    question = vault.question("Does retrieval reduce hallucination?", concepts=(rag.id,))

    before = evidence_for_question(vault.store, question.id, limit=10)
    assert {span_id for span_id, _, _ in before} == {cited.id, uncited.id}

    claim = vault.claim("Retrieval grounds generation.", rag, [cited])
    vault.store.link_answer(question.id, claim.id)

    after = evidence_for_question(vault.store, question.id, limit=10)
    assert [span_id for span_id, _, _ in after] == [uncited.id]


def test_question_text_with_punctuation_does_not_break_the_search(vault: Vault):
    """FTS5 reads punctuation as syntax, so a real question would be a query error.

    "Does retrieval reduce hallucination?" ends in a character FTS5 treats as
    an operator. Users write questions with punctuation; this must not be a
    trap they discover.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    vault.span(doc, "Retrieval grounds generation in retrieved passages.")
    question = vault.question(
        'Does "retrieval" reduce hallucination: really? (2026)', concepts=(rag.id,)
    )

    assert evidence_for_question(vault.store, question.id, limit=5)


# -- gate three: syntheses auto-mark stale ---------------------------------


def _synthesis(vault: Vault, claim: Claim, concept: Concept) -> Synthesis:
    synthesis = Synthesis(
        id="syn-1",
        scope=SynthesisScope.CONCEPT,
        scope_id=concept.id,
        body="RAG is generally held to reduce hallucination.",
        source_claim_ids=(claim.id,),
        provenance=SYNTH,
    )
    vault.store.put_synthesis(synthesis)
    return synthesis


def test_gate_three_a_synthesis_goes_stale_when_a_constituent_claim_changes(vault: Vault):
    """The invariant, on the write that causes it. No model call anywhere.

    "Does this make an existing synthesis outdated?" is answered by software:
    a status string and a fingerprint, compared.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])
    _synthesis(vault, claim, rag)
    assert vault.store.get_synthesis("syn-1").stale is False

    vault.restate(claim, status=ClaimStatus.DISPUTED)

    after = vault.store.get_synthesis("syn-1")
    assert after.stale is True
    assert "status" in (after.stale_reason or "")
    assert claim.id in (after.stale_reason or "")


def test_a_synthesis_goes_stale_when_a_claim_is_reworded(vault: Vault):
    """Status is not the only thing that changes what a claim asserts."""
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])
    _synthesis(vault, claim, rag)

    vault.restate(claim, statement="RAG eliminates hallucination.")

    after = vault.store.get_synthesis("syn-1")
    assert after.stale is True
    assert "statement" in (after.stale_reason or "")


def test_rewriting_a_claim_unchanged_does_not_stale_a_synthesis(vault: Vault):
    """Staleness that cries wolf is staleness nobody reads.

    Re-ingestion rewrites claims constantly. Only a change to what the claim
    asserts, or how strongly, may mark work written from it as outdated.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds.")
    claim = vault.claim("RAG reduces hallucination.", rag, [span])
    _synthesis(vault, claim, rag)

    vault.claim("RAG reduces hallucination.", rag, [span])  # identical rewrite

    assert vault.store.get_synthesis("syn-1").stale is False


def test_staleness_can_be_re_derived_from_the_data_alone(vault: Vault):
    """The write hook is not the only guarantee.

    A claim deleted out from under a synthesis, or a database restored from a
    backup, never passes through `put_claim`. The invariant is worth being able
    to re-establish, so `recheck_synthesis_staleness` recomputes it.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])
    _synthesis(vault, claim, rag)

    # Bypass the hook entirely, the way a restore or an external edit would.
    vault.store._conn.execute(
        "UPDATE synthesis_claims SET fingerprint = 'something else' WHERE claim_id = ?",
        (claim.id,),
    )
    vault.store._conn.commit()
    assert vault.store.get_synthesis("syn-1").stale is False

    assert vault.store.recheck_synthesis_staleness() == ["syn-1"]
    assert vault.store.get_synthesis("syn-1").stale is True


def test_a_staling_synthesis_records_a_revision(vault: Vault):
    """So "what changed this month" can report it."""
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])
    _synthesis(vault, claim, rag)
    vault.restate(claim, status=ClaimStatus.DISPUTED)

    assert "syn-1" in changes_since(vault.store, days=1).syntheses_staled


def test_the_fingerprint_ignores_what_does_not_change_the_assertion(vault: Vault):
    """Named directly, because the temptation is to hash the whole claim.

    `created_at` and provenance bookkeeping move without changing what a claim
    says. A fingerprint over the whole object would fire on those.
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])

    moved = claim.model_copy(
        update={"provenance": claim.provenance.model_copy(update={"agent_version": "9.9.9"})}
    )
    assert claim_fingerprint(moved) == claim_fingerprint(claim)

    reworded = claim.model_copy(update={"statement": "RAG eliminates hallucination."})
    assert claim_fingerprint(reworded) != claim_fingerprint(claim)


# -- gap rules fire on exactly what they claim ------------------------------


def test_a_gap_report_summarises_a_kind_that_describes_the_whole_corpus(vault: Vault):
    """Measured on the real vault: 544 of 545 concepts had no claims.

    Printing that 544 times buries the findings that name one thing. It is one
    fact about the state of extraction, and the report says it once.
    """
    for i in range(10):
        vault.concept(f"Concept {i}")

    report = gap_report(vault.store, limit=50)

    saturated = {s["kind"] for s in report.saturated}
    assert "concept_without_claims" in saturated
    assert not [g for g in report.gaps if g.kind is GapKind.CONCEPT_WITHOUT_CLAIMS]
    assert report.by_kind["concept_without_claims"] == 10, "still counted, just not listed"


def test_asking_for_a_saturated_kind_by_name_still_lists_it(vault: Vault):
    for i in range(10):
        vault.concept(f"Concept {i}")

    report = gap_report(vault.store, kinds=[GapKind.CONCEPT_WITHOUT_CLAIMS], limit=50)

    assert report.saturated == []
    assert report.returned == 10


def test_a_small_population_is_never_treated_as_saturated(vault: Vault):
    """One finding out of one subject is specific, not corpus-wide.

    Found on the first real run: the vault's single claim rested on a single
    source, and that entirely legitimate finding was summarised away as "1 of 1
    describes the corpus".
    """
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])

    report = gap_report(vault.store, limit=50)

    assert "claim_with_single_source" not in {s["kind"] for s in report.saturated}
    assert any(g.kind is GapKind.CLAIM_WITH_SINGLE_SOURCE for g in report.gaps)


def test_a_claim_with_two_sources_is_not_a_single_source_gap(vault: Vault):
    rag = vault.concept("RAG")
    _, doc_a = vault.source("a.md")
    _, doc_b = vault.source("b.md")
    vault.claim(
        "RAG reduces hallucination.",
        rag,
        [vault.span(doc_a, "Retrieval grounds."), vault.span(doc_b, "Grounding helps.")],
    )

    gaps = detect_gaps(vault.store, kinds=[GapKind.CLAIM_WITH_SINGLE_SOURCE])

    assert gaps == []


def test_a_linked_concept_is_not_isolated(vault: Vault):
    """With no inbound count recorded, the detector falls back to graph degree.
    See the counted cases below for the behaviour after `forge bootstrap`."""
    a = vault.concept("RAG")
    b = vault.concept("Vector Databases")
    vault.store.put_link(
        ClaimLink(
            id=ClaimLink.make_id(a.id, b.id, LinkType.RELATED_TO),
            from_id=a.id,
            to_id=b.id,
            type=LinkType.RELATED_TO,
            provenance=DETERMINISTIC,
            score=1.0,
            rationale="human-authored link",
        )
    )

    isolated = detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT])

    assert isolated == []


# -- isolation is about arriving, not leaving -------------------------------
#
# Reviewed against the real corpus on 2026-09-07: reporting graph degree 0 as
# isolation produced 72 findings, 71 of them wrong. 115 links pointed at those
# pages, every one from an `_index.md` or another page that is deliberately not
# a concept and so has no node. Counting inbound links over the whole vault
# moved the corpus to 53 findings that are all true.


def test_a_concept_linked_only_from_a_page_that_is_not_a_concept_is_not_isolated(
    vault: Vault,
):
    """The false positive that made the finding useless.

    An `_index.md` hub is not a concept and never becomes a node, so its links
    are not edges. The page it points at is still reachable by a human, which
    is what isolation is supposed to be about.
    """
    orphan_looking = vault.concept("Azure", path="Technologies/Docs/azure.md")
    vault.store.set_concept_inbound(
        {orphan_looking.id: (1, ["Technologies/Docs/_index.md"])}
    )

    assert detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT]) == []


def test_a_concept_nothing_links_to_is_isolated_even_when_it_links_out(vault: Vault):
    """Outgoing links do not make a page reachable. A page with twenty of them
    that nothing points at is exactly as unreachable as one with none."""
    lonely = vault.concept("Handoff", path="Projects/handoff.md")
    other = vault.concept("SmartResQ", path="Projects/smartresq.md")
    vault.store.put_link(
        ClaimLink(
            id=ClaimLink.make_id(lonely.id, other.id, LinkType.RELATED_TO),
            from_id=lonely.id,
            to_id=other.id,
            type=LinkType.RELATED_TO,
            provenance=DETERMINISTIC,
            score=1.0,
            rationale="human-authored link",
        )
    )
    vault.store.set_concept_inbound({lonely.id: (0, []), other.id: (1, ["Projects/handoff.md"])})

    gaps = detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT])

    assert [g.subject_id for g in gaps] == [lonely.id]
    assert "links out to" in gaps[0].detail


def test_a_concept_with_no_links_in_either_direction_says_so(vault: Vault):
    alone = vault.concept("Orphan", path="Projects/orphan.md")
    vault.store.set_concept_inbound({alone.id: (0, [])})

    gap = detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT])[0]

    assert "links to nothing either" in gap.detail


def test_without_counted_inbound_links_the_finding_admits_it(vault: Vault):
    """A store bootstrapped before the count existed cannot tell "nothing links
    here" from "nobody looked". It must not claim the first."""
    vault.concept("RAG", path="Technologies/Docs/rag.md")

    gap = detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT])[0]

    assert "have not been counted" in gap.detail
    assert "forge bootstrap --apply" in gap.detail


def test_a_concept_whose_page_was_deleted_is_not_reported_as_isolated(vault: Vault):
    """The counts are replaced wholesale each bootstrap, so a concept with no
    row was not seen in the last pass; its page is gone. Bootstrap leaves the
    concept alone (it may carry claims) and reports the count separately;
    saying "nothing links to this page" about a page that no longer exists is a
    false statement dressed as a finding.
    """
    live = vault.concept("RAG", path="Technologies/Docs/rag.md")
    deleted = vault.concept("Handoff", path="Projects/gone.md")
    vault.store.set_concept_inbound({live.id: (1, ["Technologies/Docs/_index.md"])})

    gaps = detect_gaps(vault.store, kinds=[GapKind.ISOLATED_CONCEPT])

    assert [g.subject_id for g in gaps] == []
    assert deleted.id not in {g.subject_id for g in gaps}


def test_no_inbound_row_is_not_the_same_answer_as_a_row_of_zero(vault: Vault):
    counted = vault.concept("A", path="a.md")
    uncounted = vault.concept("B", path="b.md")
    vault.store.set_concept_inbound({counted.id: (0, [])})

    assert vault.store.concept_inbound(counted.id) == (0, [])
    assert vault.store.concept_inbound(uncounted.id) is None


def test_counting_inbound_links_replaces_the_previous_count(vault: Vault):
    """The input is one pass over the vault, so a concept whose last inbound
    link was deleted must lose its row, an update that only wrote what it saw
    would leave it claiming to be linked."""
    a = vault.concept("A", path="a.md")
    b = vault.concept("B", path="b.md")
    vault.store.set_concept_inbound({a.id: (1, ["hub.md"]), b.id: (1, ["hub.md"])})
    vault.store.set_concept_inbound({a.id: (1, ["hub.md"]), b.id: (0, [])})

    assert vault.store.concept_inbound(b.id) == (0, [])
    assert vault.store.unreferenced_concepts() == [b.id]


def test_unreferenced_concepts_tells_never_counted_from_all_linked(vault: Vault):
    """`None` and `[]` are different answers and a caller must not collapse
    them: one means nobody has looked."""
    a = vault.concept("A", path="a.md")

    assert vault.store.unreferenced_concepts() is None
    vault.store.set_concept_inbound({a.id: (2, ["hub.md", "other.md"])})
    assert vault.store.unreferenced_concepts() == []


def test_only_a_few_linking_pages_are_kept_per_concept(vault: Vault):
    """Enough to say where a page is reachable from, not a second copy of the
    link graph in a text column."""
    a = vault.concept("A", path="a.md")
    vault.store.set_concept_inbound({a.id: (40, [f"p{i}.md" for i in range(40)])})

    count, sources = vault.store.concept_inbound(a.id)

    assert count == 40, "the count is not capped, only the sample"
    assert len(sources) == 5


def test_a_gap_id_is_stable_across_runs(vault: Vault):
    """So a gap report can be diffed between runs rather than re-read."""
    vault.concept("RAG")
    first = detect_gaps(vault.store)
    second = detect_gaps(vault.store)
    assert [g.id for g in first] == [g.id for g in second]


def test_gap_detection_makes_no_model_calls(vault: Vault):
    """A generated gap list would be fluent, plausible and unfalsifiable."""
    from forge.llm.base import CALLS

    vault.concept("RAG")
    vault.question("What is agent memory?")
    CALLS.reset()

    detect_gaps(vault.store)
    changes_since(vault.store)
    belief_for_concept(vault.store, Concept.make_id("RAG"))

    assert CALLS.count == 0


# -- domain invariants ------------------------------------------------------


def test_a_question_must_be_a_user_assertion():
    """A model filing questions alongside the user's own would quietly change
    what the gap report is measuring."""
    with pytest.raises(ProvenanceViolation):
        Question(id="q1", text="What is RAG?", provenance=MODEL)


def test_a_synthesis_must_declare_the_synthesis_tier():
    with pytest.raises(ProvenanceViolation):
        Synthesis(
            id="s1",
            scope=SynthesisScope.CONCEPT,
            scope_id="c1",
            body="text",
            source_claim_ids=("c",),
            provenance=MODEL,
        )


def test_a_synthesis_with_no_constituent_claims_is_refused():
    """Its one invariant is staleness, and nothing to compare cannot go stale."""
    with pytest.raises(ValueError, match="staleness"):
        Synthesis(
            id="s1",
            scope=SynthesisScope.CONCEPT,
            scope_id="c1",
            body="text",
            source_claim_ids=(),
            provenance=SYNTH,
        )


def test_a_stale_synthesis_must_say_what_made_it_stale():
    with pytest.raises(ValueError, match="what made it stale"):
        Synthesis(
            id="s1",
            scope=SynthesisScope.CONCEPT,
            scope_id="c1",
            body="text",
            source_claim_ids=("c",),
            provenance=SYNTH,
            stale=True,
        )


def test_an_answered_question_must_record_when(vault: Vault):
    with pytest.raises(ValueError, match="when it was resolved"):
        Question(
            id="q1", text="What is RAG?", provenance=HUMAN, status=QuestionStatus.ANSWERED
        )


def test_question_identity_is_the_question_not_its_capitalisation(vault: Vault):
    """Asking the same thing twice records one question, not two."""
    first = vault.question("What is agent memory?")
    second = vault.question("what is AGENT memory?")
    assert first.id == second.id
    assert len(vault.store.list_questions()) == 1


# -- the v4 -> v5 migration -------------------------------------------------


def test_a_v4_database_upgrades_without_losing_anything(tmp_path: Path):
    """Phase 9 adds four tables to a schema people already have data in.

    Simulated the way an upgrade really happens: build a store, drop the new
    tables and roll the recorded version back, then reopen. A migration tested
    only by creating a fresh database tests nothing.
    """
    path = tmp_path / "v4.db"
    store = SqliteStore(path)
    store.initialize()
    vault = Vault(store)
    rag = vault.concept("RAG", path="Technologies/Docs/rag.md")
    _, doc = vault.source("rag.md")
    claim = vault.claim("RAG reduces hallucination.", rag, [vault.span(doc, "Retrieval grounds.")])
    before = store.counts()

    for table in (
        "concept_inbound_links",
        "question_answers",
        "synthesis_claims",
        "syntheses",
        "questions",
    ):
        store._conn.execute(f"DROP TABLE {table}")
    store._conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    store._conn.commit()
    store.close()

    upgraded = SqliteStore(path)
    upgraded.initialize()

    assert upgraded.schema_version == SCHEMA_VERSION
    after = upgraded.counts()
    for table, count in before.items():
        assert after[table] == count, f"{table} lost rows in the upgrade"
    assert upgraded.get_claim(claim.id) is not None
    assert upgraded.list_questions() == []

    # And the new tables work on the upgraded database, not just a fresh one.
    question = Question(id=Question.make_id("What is RAG?"), text="What is RAG?", provenance=HUMAN)
    upgraded.put_question(question)
    upgraded.link_answer(question.id, claim.id)
    assert [c.id for c in upgraded.answers_for_question(question.id)] == [claim.id]

    # v6's table too: an upgraded database must be able to record inbound
    # links, or `forge gaps` keeps reporting the finding it cannot trust.
    assert upgraded.unreferenced_concepts() is None
    upgraded.set_concept_inbound({rag.id: (0, [])})
    assert upgraded.unreferenced_concepts() == [rag.id]
    upgraded.close()


def test_superseding_a_claim_stales_work_written_from_it(tmp_path: Path):
    """`supersede_claim` writes through raw SQL, bypassing `put_claim`.

    That made it the one path where a claim could change under a synthesis
    without the hook noticing, which is the clearest case for staling there is:
    the claim the text was written from is no longer what the model holds.
    Found by writing this test.
    """
    store = SqliteStore(tmp_path / "forge.db")
    store.initialize()
    vault = Vault(store)
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Retrieval grounds generation.")
    original = vault.claim("RAG reduces hallucination.", rag, [span])
    _synthesis(vault, original, rag)

    replacement = Claim(
        id=Claim.make_id("RAG reduces hallucination on open-domain questions.", span.id),
        statement="RAG reduces hallucination on open-domain questions.",
        subject_concept_id=rag.id,
        provenance=MODEL,
    )
    store.supersede_claim(original.id, replacement)

    after = store.get_synthesis("syn-1")
    assert after.stale is True
    assert "status" in (after.stale_reason or "")
    store.close()


def test_a_pipe_in_a_statement_does_not_shift_every_label(tmp_path):
    """`stale_reason` named the wrong fields for any claim containing a `|`.

    The fingerprint is `status|tier|statement|superseded_by`, and the statement
    is the only field that can hold a pipe: a table row, a shell pipeline, an
    `A | B`. Splitting naively gave five parts, so `statement` was reported as
    the text up to the pipe and `superseded_by` as the text after it. The
    reason a human reads to decide whether to rewrite a synthesis was wrong
    exactly for the claims whose text is most structured.

    Found by turning on B905: `zip(labels, old, new)` was silently truncating.
    """
    store = SqliteStore(tmp_path / "forge.db")
    store.initialize()
    vault = Vault(store)
    rag = vault.concept("RAG")
    _, doc = vault.source("rag.md")
    span = vault.span(doc, "Recall and precision trade off.")
    original = vault.claim("recall | precision is the trade-off.", rag, [span])
    _synthesis(vault, original, rag)

    replacement = Claim(
        id=Claim.make_id("recall | precision is not the trade-off.", span.id),
        statement="recall | precision is not the trade-off.",
        subject_concept_id=rag.id,
        provenance=MODEL,
    )
    store.supersede_claim(original.id, replacement)

    reason = store.get_synthesis("syn-1").stale_reason or ""
    # What moved is the original claim's status and its superseded_by, and both
    # must be named with their real values. Before the fix the naive split put
    # the text after the pipe into superseded_by and dropped the real id off
    # the end of the zip, so the one field a reader needs was never reported.
    assert "status 'active' -> 'superseded'" in reason
    assert replacement.id in reason
    assert "precision" not in reason
    store.close()


def test_a_fingerprint_from_another_version_says_only_that_it_differs(tmp_path):
    """A restore, or a fingerprint written by an older schema, is not four fields.

    Guessing at field names it cannot see would be worse than saying less.
    """
    from forge.storage.sqlite_store import _describe_drift

    assert _describe_drift("c1", "something else", "a|b|c|d").endswith("content differs")
    assert _describe_drift("c1", "a|b|c|d", "a|b|c|d").endswith("content differs")
    assert "tier 'b' -> 'z'" in _describe_drift("c1", "a|b|c|d", "a|z|c|d")
