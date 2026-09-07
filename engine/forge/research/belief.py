"""What do I currently believe about X, and who disagrees.

**"Believe" is a deliberate word and this module is careful with it.** What
Forge holds is claims with provenance and status, not beliefs with confidence:
`forge.evolution.impact` refuses to invent a confidence, because a number a
model emits about its own certainty is not a measurement. So a belief here is
the *set of active claims* about a concept, each carrying how it was derived
and what evidences it. A reader can see that four claims are model-extracted
from one source and judge accordingly; a single "confidence: 0.82" would have
hidden exactly that.

**Dissent is reported, never resolved.** Forge has no `CONTRADICTS` edge on
purpose: Phase 4 produces `POTENTIAL_CONFLICT` and routes it to a human rather
than asserting a contradiction a model detected. So dissent here is three real
things: claims a human marked disputed, conflict proposals still awaiting
review, and claims that were superseded, which are what the user used to
believe and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import Claim, ClaimStatus, ProposalStatus, ProposalType
from ..storage import SqliteStore


@dataclass
class Dissent:
    """One reason to doubt, with what raised it.

    `kind` is `disputed_claim`, `open_conflict` or `superseded_claim`. All
    three are states a human either created or has not yet resolved; none is a
    model's verdict.
    """

    kind: str
    detail: str
    claim_id: str | None = None
    proposal_id: str | None = None
    raised_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "claim_id": self.claim_id,
            "proposal_id": self.proposal_id,
            "raised_by": self.raised_by,
        }


@dataclass
class Belief:
    """The answer to "what do I believe about X", with its supports and dissent."""

    concept_id: str
    concept_name: str
    held: list[Claim] = field(default_factory=list)
    disputed: list[Claim] = field(default_factory=list)
    superseded: list[Claim] = field(default_factory=list)
    #: Distinct source locators the held claims rest on.
    supporting_sources: list[str] = field(default_factory=list)
    dissent: list[Dissent] = field(default_factory=list)
    #: Held claims resting on no evidence at all. Only USER_ASSERTION may do
    #: that legitimately; anything else here is a defect worth seeing.
    unevidenced: list[str] = field(default_factory=list)

    @property
    def is_settled(self) -> bool:
        """Held claims, and nothing outstanding against them."""
        return bool(self.held) and not self.dissent

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "held": [c.id for c in self.held],
            "disputed": [c.id for c in self.disputed],
            "superseded": [c.id for c in self.superseded],
            "supporting_sources": self.supporting_sources,
            "dissent": [d.to_dict() for d in self.dissent],
            "unevidenced": self.unevidenced,
            "is_settled": self.is_settled,
        }


def belief_for_concept(store: SqliteStore, concept_id: str) -> Belief | None:
    """Assemble what is currently held about one concept. Deterministic."""
    concept = store.get_concept(concept_id)
    if concept is None:
        return None

    belief = Belief(concept_id=concept.id, concept_name=concept.qualified_name)
    claims = [c for c in store.list_claims() if c.subject_concept_id == concept_id]

    sources: dict[str, None] = {}
    for claim in claims:
        if claim.status is ClaimStatus.ACTIVE:
            belief.held.append(claim)
        elif claim.status is ClaimStatus.DISPUTED:
            belief.disputed.append(claim)
            belief.dissent.append(
                Dissent(
                    kind="disputed_claim",
                    detail=f"a claim about this concept is marked disputed: {claim.statement}",
                    claim_id=claim.id,
                )
            )
        elif claim.status is ClaimStatus.SUPERSEDED:
            belief.superseded.append(claim)
            belief.dissent.append(
                Dissent(
                    kind="superseded_claim",
                    detail=(
                        f"this was held and has been superseded: {claim.statement}"
                        + (f" (by {claim.superseded_by})" if claim.superseded_by else "")
                    ),
                    claim_id=claim.id,
                )
            )

        evidence = store.evidence_for_claim(claim.id)
        if claim.status is ClaimStatus.ACTIVE and not evidence:
            belief.unevidenced.append(claim.id)
        for link in evidence:
            span = store.get_span(link.span_id)
            document = store.get_document(span.document_id) if span else None
            source = store.get_source(document.source_id) if document else None
            if source is not None and claim.status is ClaimStatus.ACTIVE:
                sources[source.locator] = None

    belief.supporting_sources = sorted(sources)

    # Conflict proposals still awaiting a human. These are the honest form of
    # "which sources disagree": Forge never asserts a contradiction itself.
    held_ids = {c.id for c in claims}
    for proposal in store.list_proposals(
        status=ProposalStatus.PENDING, type=ProposalType.CLAIM_CONFLICT, limit=500
    ):
        # `target_entity_id` is the claim the conflict is against; the operation's
        # target is a name rather than an id, so it is not what to match on.
        target = proposal.target_entity_id
        if target in held_ids or target == concept_id:
            belief.dissent.append(
                Dissent(
                    kind="open_conflict",
                    detail=proposal.reason or "evidence that appears to disagree, awaiting review",
                    claim_id=target if target in held_ids else None,
                    proposal_id=proposal.id,
                    raised_by=proposal.provenance.agent,
                )
            )

    return belief
