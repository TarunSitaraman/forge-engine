"""Proposal service — create, list, decide.

The engine's only route to changing anything a human owns. Approving a proposal
records a decision; it does not enact it. Enactment is a separate, explicitly
flagged step (see :mod:`forge.proposals.apply`), and is off by default.

Proposal identity is deterministic, derived from what the proposal *is*. That
matters more than it sounds: without it, every re-ingest and every re-run of
diagnostics would resurrect proposals the user already rejected, as fresh
PENDING ones. With it, a rejected proposal stays rejected.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..domain import (
    Concept,
    Derivation,
    EntityType,
    MatchKind,
    Proposal,
    ProposalStatus,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    SafetyClass,
    deterministic_provenance,
)
from ..extraction.extractor import ClaimCandidate, ConceptCandidate
from ..ids import text_hash
from ..logging import get_logger
from ..matching.matcher import MatchResult
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)


class ProposalService:
    """Creates and decides proposals."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    # -- creation ----------------------------------------------------------

    def create(self, proposal: Proposal) -> tuple[Proposal, bool]:
        """Store a proposal if unseen. Returns ``(proposal, created)``.

        An existing proposal is returned untouched — including its decision. A
        rejected proposal is never silently reopened.
        """
        existing = self.store.get_proposal(proposal.id)
        if existing is not None:
            return existing, False
        self.store.put_proposal(proposal)
        return proposal, True

    def create_many(self, proposals: Iterable[Proposal]) -> tuple[int, int]:
        created = skipped = 0
        for proposal in proposals:
            _, was_created = self.create(proposal)
            created += int(was_created)
            skipped += int(not was_created)
        return created, skipped

    # -- decisions ---------------------------------------------------------

    def approve(self, proposal_id: str, *, by: str = "cli", note: str | None = None) -> Proposal:
        proposal = self._require(proposal_id)
        decided = proposal.approve(by=by, note=note)
        self.store.put_proposal(decided)
        log.info("proposal_approved", proposal_id=decided.id, type=decided.type.value)
        return decided

    def reject(self, proposal_id: str, *, by: str = "cli", note: str | None = None) -> Proposal:
        proposal = self._require(proposal_id)
        decided = proposal.reject(by=by, note=note)
        self.store.put_proposal(decided)
        log.info("proposal_rejected", proposal_id=decided.id, type=decided.type.value)
        return decided

    def supersede(self, proposal_id: str, by_proposal_id: str) -> Proposal:
        proposal = self._require(proposal_id)
        decided = proposal.supersede(by_proposal_id)
        self.store.put_proposal(decided)
        return decided

    # -- queries -----------------------------------------------------------

    def get(self, proposal_id: str) -> Proposal | None:
        found = self.store.get_proposal(proposal_id)
        if found is not None:
            return found
        matches = self.store.find_proposal(proposal_id)
        return matches[0] if len(matches) == 1 else None

    def resolve(self, proposal_id: str) -> tuple[Proposal | None, list[Proposal]]:
        """Resolve a possibly-abbreviated id.

        Returns ``(proposal, ambiguous_matches)``. An abbreviation matching
        several proposals resolves to none of them — the same discipline the
        concept matcher applies.
        """
        exact = self.store.get_proposal(proposal_id)
        if exact is not None:
            return exact, []
        matches = self.store.find_proposal(proposal_id)
        if len(matches) == 1:
            return matches[0], []
        return None, matches

    def list(
        self,
        *,
        status: ProposalStatus | None = None,
        type: ProposalType | None = None,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Proposal]:
        return self.store.list_proposals(
            status=status, type=type, source_id=source_id, limit=limit, offset=offset
        )

    def counts(self) -> dict[str, int]:
        return self.store.count_proposals()

    def _require(self, proposal_id: str) -> Proposal:
        proposal, ambiguous = self.resolve(proposal_id)
        if proposal is None:
            if ambiguous:
                raise KeyError(
                    f"{proposal_id!r} matches {len(ambiguous)} proposals: "
                    f"{', '.join(p.id[:10] for p in ambiguous)}"
                )
            raise KeyError(f"no proposal {proposal_id!r}")
        return proposal


# --------------------------------------------------------------------------
# Builders — one per proposal type
# --------------------------------------------------------------------------


def concept_proposal(
    candidate: ConceptCandidate,
    match: MatchResult,
    provenance: Provenance,
    *,
    source_id: str | None = None,
) -> Proposal:
    """Build a proposal for an extracted concept.

    The match outcome decides the proposal's type and safety class:

    * ``NEW_CONCEPT``     -> create a concept, ``MODEL_GENERATED``
    * ``MATCH_CANDIDATE`` -> link to an existing concept, ``MODEL_GENERATED``
    * ``AMBIGUOUS``       -> ``AMBIGUOUS`` safety, and the operation deliberately
      names no target, because there is no defensible one to name
    """
    is_new = match.kind is MatchKind.NEW_CONCEPT
    proposal_type = ProposalType.NEW_CONCEPT if is_new else ProposalType.CONCEPT_MATCH

    if match.kind is MatchKind.AMBIGUOUS:
        safety = SafetyClass.AMBIGUOUS
        action = "resolve_ambiguous_concept"
        after = None
    elif is_new:
        safety = SafetyClass.MODEL_GENERATED
        action = "create_concept"
        after = candidate.name
    else:
        safety = SafetyClass.MODEL_GENERATED
        action = "link_to_existing_concept"
        best = match.best
        after = best.canonical_name if best else None

    return Proposal(
        id=Proposal.make_id(
            proposal_type, candidate.name, text_hash(f"{match.kind.value}:{candidate.span_id}")
        ),
        type=proposal_type,
        safety=safety,
        target_entity_type=EntityType.CONCEPT,
        target_entity_id=(match.best.concept_id if match.best else None),
        operation=ProposedOperation(
            action=action,
            target=candidate.name,
            after=after,
            details={
                "kind": candidate.kind,
                "mention": candidate.mention,
                "match_kind": match.kind.value,
                "candidates": [c.to_dict() for c in match.candidates],
            },
        ),
        reason=match.reason,
        evidence_span_ids=(candidate.span_id,),
        source_id=source_id,
        provenance=provenance,
    )


def claim_proposal(
    candidate: ClaimCandidate,
    provenance: Provenance,
    *,
    source_id: str | None = None,
) -> Proposal:
    """Build a proposal for an extracted claim.

    Always evidenced: the candidate's quote was already verified as present in
    its span before this point, so the proposal cannot cite evidence that does
    not exist.
    """
    return Proposal(
        id=Proposal.make_id(
            ProposalType.NEW_CLAIM, candidate.statement, text_hash(candidate.span_id)
        ),
        type=ProposalType.NEW_CLAIM,
        safety=SafetyClass.MODEL_GENERATED,
        target_entity_type=EntityType.CLAIM,
        operation=ProposedOperation(
            action="create_claim",
            target=candidate.statement[:120],
            after=candidate.statement,
            details={"evidence_quote": candidate.evidence_quote, "concept": candidate.concept},
        ),
        reason=f"extracted from span {candidate.span_id[:10]} with a verbatim supporting quote",
        evidence_span_ids=(candidate.span_id,),
        source_id=source_id,
        provenance=provenance,
    )
