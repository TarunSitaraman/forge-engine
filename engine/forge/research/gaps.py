"""What am I missing, and what remains unanswered.

**Every rule here is a structural fact about the graph, never a judgement.**
"This concept has no claims" is checkable and either true or false. Whether it
*matters* is the user's call, which is why nothing in Forge acts on a gap: it
is reported, ordered by a stated heuristic, and left alone. A model asked to
find "gaps in my understanding" would produce a fluent list nobody could
falsify, which is the failure mode this whole system is built against.

**The weights order a report; they are not probabilities.** Each is a plain
arithmetic rule written down beside the detector that uses it, so a reader can
disagree with the ordering without having to reverse-engineer it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..domain import (
    ClaimStatus,
    GapKind,
    KnowledgeGap,
    ProposalStatus,
    ProposalType,
    Question,
    QuestionStatus,
    utc_now,
)
from ..storage import SqliteStore

#: How long a claim may sit disputed before the dispute is itself the finding.
#: Two weeks is a working default: long enough that a dispute raised and
#: resolved in a normal review cycle does not appear, short enough that one
#: forgotten for a month does.
DISPUTE_STALE_DAYS = 14

#: Cap on gaps returned. A 545-concept vault with no claims yields 545
#: `CONCEPT_WITHOUT_CLAIMS` findings, which is one fact repeated, not a report.
DEFAULT_GAP_LIMIT = 100


#: When a gap kind applies to at least this share of the things it could apply
#: to, the report says so once instead of listing every instance.
#:
#: Measured, not guessed: run over the real vault on 2026-09-07, before any
#: extraction, `CONCEPT_WITHOUT_CLAIMS` matched **544 of 545 concepts**. That is
#: one fact about the state of extraction, and printing it 544 times buries the
#: 72 isolated concepts underneath it, which *are* individually actionable. A
#: report nobody reads to the bottom of has failed whatever it contains.
SATURATION_RATIO = 0.5

#: ...but only once there are enough subjects for a ratio to mean anything.
#: Without this, one finding against a population of one is "100% saturated"
#: and gets collapsed into a summary, which is the opposite of the intent:
#: observed on the first real run, where the vault's single claim rested on a
#: single source and that entirely legitimate finding was summarised away.
SATURATION_MIN_POPULATION = 5


@dataclass
class GapReport:
    """Gap findings, with saturated kinds collapsed to one line each.

    `saturated` names the kinds that describe the corpus rather than any
    particular subject. Their instances are still counted and still reachable
    by asking for that kind specifically; they are kept out of `gaps` so the
    findings that name one thing are visible.
    """

    gaps: list[KnowledgeGap] = field(default_factory=list)
    by_kind: dict[str, int] = field(default_factory=dict)
    saturated: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    returned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "returned": self.returned,
            "by_kind": self.by_kind,
            "saturated": self.saturated,
            "gaps": [
                {
                    "id": g.id,
                    "kind": g.kind.value,
                    "subject_id": g.subject_id,
                    "subject_type": g.subject_type,
                    "subject_label": g.subject_label,
                    "detail": g.detail,
                    "weight": round(g.weight, 3),
                    "evidence": list(g.evidence),
                }
                for g in self.gaps
            ],
        }


def gap_report(
    store: SqliteStore,
    *,
    kinds: Iterable[GapKind] | None = None,
    limit: int = DEFAULT_GAP_LIMIT,
) -> GapReport:
    """`detect_gaps`, with saturated kinds summarised rather than enumerated.

    Asking for a kind explicitly always enumerates it: someone who wants all
    544 claimless concepts can have them, and a caller that did not ask for one
    kind in particular gets a report they can read.
    """
    asked_for = set(kinds) if kinds else None
    everything = detect_gaps(store, kinds=kinds, limit=1_000_000)

    report = GapReport(total=len(everything))
    for gap in everything:
        report.by_kind[gap.kind.value] = report.by_kind.get(gap.kind.value, 0) + 1

    populations = _populations(store)
    collapsed: set[GapKind] = set()
    for kind_value, count in sorted(report.by_kind.items()):
        kind = GapKind(kind_value)
        population = populations.get(kind, 0)
        if asked_for and kind in asked_for:
            continue  # asked for by name: enumerate it
        if population >= SATURATION_MIN_POPULATION and count / population >= SATURATION_RATIO:
            collapsed.add(kind)
            report.saturated.append(
                {
                    "kind": kind_value,
                    "count": count,
                    "population": population,
                    "detail": (
                        f"{count} of {population} describe the corpus rather than "
                        f"any one subject; ask for this kind by name to list them"
                    ),
                }
            )

    report.gaps = [g for g in everything if g.kind not in collapsed][:limit]
    report.returned = len(report.gaps)
    return report


def _populations(store: SqliteStore) -> dict[GapKind, int]:
    """How many subjects each kind *could* apply to. The saturation denominator."""
    concepts = len(store.list_concepts())
    claims = [c for c in store.list_claims() if c.status is ClaimStatus.ACTIVE]
    return {
        GapKind.CONCEPT_WITHOUT_CLAIMS: concepts,
        GapKind.ISOLATED_CONCEPT: concepts,
        GapKind.CLAIM_WITH_SINGLE_SOURCE: len(claims),
        GapKind.QUESTION_WITHOUT_ANSWERS: len(open_questions(store)),
        GapKind.UNRESOLVED_DISPUTE: len(store.list_claims()),
    }


def open_questions(store: SqliteStore) -> list[Question]:
    """Questions not yet answered, oldest first. Question 4 of the six."""
    return [
        q
        for q in store.list_questions()
        if q.status in (QuestionStatus.OPEN, QuestionStatus.PARTIALLY_ANSWERED)
    ]


def detect_gaps(
    store: SqliteStore,
    *,
    kinds: Iterable[GapKind] | None = None,
    limit: int = DEFAULT_GAP_LIMIT,
) -> list[KnowledgeGap]:
    """Run the deterministic gap queries, highest weight first.

    `kinds` restricts which detectors run, which matters on a large vault: the
    concept-level rules dominate a corpus seeded from filenames, and someone
    looking for unresolved disputes should not have to page past 545 of them.
    """
    wanted = set(kinds) if kinds else set(GapKind)
    found: list[KnowledgeGap] = []

    claims = list(store.list_claims())
    claims_by_concept: dict[str, list] = {}
    for claim in claims:
        if claim.subject_concept_id:
            claims_by_concept.setdefault(claim.subject_concept_id, []).append(claim)

    if {GapKind.CONCEPT_WITHOUT_CLAIMS, GapKind.ISOLATED_CONCEPT} & wanted:
        found.extend(_concept_gaps(store, claims_by_concept, wanted))
    if GapKind.QUESTION_WITHOUT_ANSWERS in wanted:
        found.extend(_question_gaps(store))
    if GapKind.CLAIM_WITH_SINGLE_SOURCE in wanted:
        found.extend(_single_source_claims(store, claims))
    if GapKind.UNRESOLVED_DISPUTE in wanted:
        found.extend(_unresolved_disputes(store, claims))

    found.sort(key=lambda g: (-g.weight, g.kind.value, g.subject_label))
    return found[:limit]


def _concept_gaps(store: SqliteStore, claims_by_concept: dict, wanted: set) -> list[KnowledgeGap]:
    """Concepts nothing is claimed about, and concepts nothing links to.

    Weight: a concept with a vault page but no claims is weighted above one
    without, because the user demonstrably cared enough to write the page. An
    isolated concept weighs less than a claimless one: being unlinked is a
    graph-shape observation, while having no claims means the model holds
    nothing about it at all.

    **Isolation is measured by inbound links over the whole vault, not by graph
    degree.** Edges run between concept pages, and the pages that do most of
    the linking in a vault organised hub-and-spoke — `_index.md`, `00_Index/` —
    are deliberately not concepts, so their links never become edges. Reporting
    degree 0 as isolation was measured wrong on the real corpus on 2026-09-07:
    72 findings, 115 inbound links pointing at them, and exactly **one** page
    that nothing in the vault linked to. Worse, a genuinely unreferenced page
    that was then linked from the index its siblings are linked from stayed on
    the list, because that index is navigation — a finding a user cannot clear
    by doing the right thing teaches them to ignore the report.

    Counting inbound links instead moved the corpus from 72 findings, 71 of
    them wrong, to 53 that are all true: 26 cheat sheets and 8 templates that
    nothing links to, 14 problem pages missing from their pattern's index, the
    losing side of three decided name collisions, and one orphaned page. Note
    that outgoing links are irrelevant here: a page with twenty of them that
    nothing points at is exactly as unreachable as one with none.
    """
    out: list[KnowledgeGap] = []
    inbound_known = store.inbound_counted()
    for concept in store.list_concepts():
        has_claims = bool(claims_by_concept.get(concept.id))
        if GapKind.CONCEPT_WITHOUT_CLAIMS in wanted and not has_claims:
            out.append(
                KnowledgeGap(
                    id=KnowledgeGap.make_id(GapKind.CONCEPT_WITHOUT_CLAIMS, concept.id),
                    kind=GapKind.CONCEPT_WITHOUT_CLAIMS,
                    subject_id=concept.id,
                    subject_type="Concept",
                    subject_label=concept.qualified_name,
                    detail=(
                        "the model knows this concept's name and nothing else: "
                        "no claim has it as a subject"
                    ),
                    weight=0.6 if concept.vault_path else 0.4,
                    evidence=(concept.vault_path,) if concept.vault_path else (),
                )
            )
        if GapKind.ISOLATED_CONCEPT in wanted:
            degree = len(store.links_from(concept.id)) + len(store.links_to(concept.id))
            if inbound_known:
                # Isolation is about arriving, not leaving. A page with twenty
                # outgoing links that nothing points at is exactly as
                # unreachable as one with none.
                inbound, _ = store.concept_inbound(concept.id)
                isolated = inbound == 0
                detail = (
                    "no page in the vault links to this one, so nothing but a "
                    "search will reach it"
                    + (
                        f"; it links out to {degree} other concept(s)"
                        if degree
                        else ", and it links to nothing either"
                    )
                )
            else:
                isolated = degree == 0
                detail = (
                    "no edges in either direction — but inbound links have not "
                    "been counted for this store, and links from pages that are "
                    "not concepts do not make edges. Run `forge bootstrap "
                    "--apply` before trusting this finding"
                )
            if not isolated:
                continue
            out.append(
                KnowledgeGap(
                    id=KnowledgeGap.make_id(GapKind.ISOLATED_CONCEPT, concept.id),
                    kind=GapKind.ISOLATED_CONCEPT,
                    subject_id=concept.id,
                    subject_type="Concept",
                    subject_label=concept.qualified_name,
                    detail=detail,
                    weight=0.3,
                    evidence=(concept.vault_path,) if concept.vault_path else (),
                )
            )
    return out


def _question_gaps(store: SqliteStore) -> list[KnowledgeGap]:
    """Open questions no claim answers. Weight rises with age, in whole weeks.

    A question asked yesterday and unanswered is not yet a gap; one asked three
    months ago and untouched is the clearest kind there is.
    """
    out: list[KnowledgeGap] = []
    now = utc_now()
    for question in open_questions(store):
        answers = store.answers_for_question(question.id)
        if answers:
            continue
        weeks = max(0, (now - question.created_at).days // 7)
        out.append(
            KnowledgeGap(
                id=KnowledgeGap.make_id(GapKind.QUESTION_WITHOUT_ANSWERS, question.id),
                kind=GapKind.QUESTION_WITHOUT_ANSWERS,
                subject_id=question.id,
                subject_type="Question",
                subject_label=question.text,
                detail=f"asked {weeks} week(s) ago and no claim answers it",
                weight=min(1.0, 0.7 + 0.05 * weeks),
                evidence=question.tags,
            )
        )
    return out


def _single_source_claims(store: SqliteStore, claims: list) -> list[KnowledgeGap]:
    """Active claims resting on exactly one source.

    Not an error. A single-source claim is unreplicated, which is worth knowing
    before repeating it, and is precisely what "which papers are most relevant"
    is for. Model-derived claims weigh more than user assertions: a human
    asserting something from one source is a decision, a model doing it is an
    extraction nobody has corroborated.
    """
    out: list[KnowledgeGap] = []
    for claim in claims:
        if claim.status is not ClaimStatus.ACTIVE:
            continue
        sources: set[str] = set()
        for link in store.evidence_for_claim(claim.id):
            span = store.get_span(link.span_id)
            document = store.get_document(span.document_id) if span else None
            if document:
                sources.add(document.source_id)
        if len(sources) != 1:
            continue
        from_model = claim.provenance.derivation.value == "model"
        out.append(
            KnowledgeGap(
                id=KnowledgeGap.make_id(GapKind.CLAIM_WITH_SINGLE_SOURCE, claim.id),
                kind=GapKind.CLAIM_WITH_SINGLE_SOURCE,
                subject_id=claim.id,
                subject_type="Claim",
                subject_label=claim.statement,
                detail="rests on one source; nothing corroborates it",
                weight=0.5 if from_model else 0.25,
                evidence=tuple(sorted(sources)),
            )
        )
    return out


def _unresolved_disputes(store: SqliteStore, claims: list) -> list[KnowledgeGap]:
    """Disputes left open, and conflict proposals nobody has ruled on.

    The highest weight in the report, because this is the failure the vision
    names first: two things you believe disagree and nothing says which is
    current.
    """
    out: list[KnowledgeGap] = []
    now = utc_now()
    for claim in claims:
        if claim.status is not ClaimStatus.DISPUTED:
            continue
        out.append(
            KnowledgeGap(
                id=KnowledgeGap.make_id(GapKind.UNRESOLVED_DISPUTE, claim.id),
                kind=GapKind.UNRESOLVED_DISPUTE,
                subject_id=claim.id,
                subject_type="Claim",
                subject_label=claim.statement,
                detail="marked disputed and not resolved",
                weight=0.9,
            )
        )

    for proposal in store.list_proposals(
        status=ProposalStatus.PENDING, type=ProposalType.CLAIM_CONFLICT, limit=500
    ):
        days = (now - proposal.created_at).days
        if days < DISPUTE_STALE_DAYS:
            continue
        subject = proposal.target_entity_id or proposal.id
        out.append(
            KnowledgeGap(
                id=KnowledgeGap.make_id(GapKind.UNRESOLVED_DISPUTE, proposal.id),
                kind=GapKind.UNRESOLVED_DISPUTE,
                subject_id=subject,
                subject_type="Proposal",
                subject_label=proposal.reason,
                detail=f"a conflict has awaited review for {days} days",
                weight=min(1.0, 0.9 + 0.005 * (days - DISPUTE_STALE_DAYS)),
                evidence=proposal.evidence_span_ids,
            )
        )
    return out


def evidence_for_question(
    store: SqliteStore, question_id: str, *, limit: int = 10
) -> list[tuple[str, float, str]]:
    """Spans most relevant to an open question. Question 6 of the six.

    Lexical, deterministic, and **scoped by the question rather than by a
    keyword**: the query is built from the question's own text plus the names
    of any concepts the user attached to it, which is what the vision means by
    "retrieval scoped by an open question".

    Spans already cited by the question's answering claims are excluded. What
    is wanted is what has *not* been read into the answer yet; returning the
    evidence already used would be a search box with extra steps.

    Returns `(span_id, score, citation)`, lower score first, matching the
    store's lexical convention.
    """
    question = store.get_question(question_id)
    if question is None:
        return []

    terms = [question.text]
    for concept_id in question.concept_ids:
        concept = store.get_concept(concept_id)
        if concept is not None:
            terms.append(concept.canonical_name)

    already: set[str] = set()
    for claim in store.answers_for_question(question_id):
        already.update(link.span_id for link in store.evidence_for_claim(claim.id))

    # FTS5 treats punctuation as syntax, so a question mark or a colon in the
    # user's own wording would be a query error rather than a search. Reduced
    # to bare words, OR-ed, which is the loosest useful reading of a question.
    words = sorted({w for term in terms for w in _words(term)})
    if not words:
        return []
    query = " OR ".join(words)

    out: list[tuple[str, float, str]] = []
    for span, score in store.search_spans(query, limit=limit + len(already)):
        if span.id in already:
            continue
        out.append((span.id, score, span.citation()))
        if len(out) >= limit:
            break
    return out


#: Words too common to narrow anything, and question words that appear in
#: almost every question asked. Kept small on purpose: an aggressive stop list
#: would silently drop a real term like "state" or "memory".
_STOPWORDS = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "i", "my", "me", "about", "of", "for", "to", "in", "on", "at", "by", "with", "from", "and", "or", "not", "as", "it", "its", "this", "that", "these", "those", "there", "here", "can", "could", "should", "would", "may", "might", "will", "shall", "have", "has", "had", "be", "been", "being", "if", "then", "than", "so", "such"]
)


def _words(text: str) -> list[str]:
    import re

    return [
        w
        for w in re.sub(r"[^0-9a-zA-Z]+", " ", text.lower()).split()
        if len(w) > 2 and w not in _STOPWORDS
    ]
