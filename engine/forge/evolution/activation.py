"""Activating evolution proposals — where approved judgement becomes knowledge.

Phase 3 activated proposals that *add* knowledge. These activate proposals that
*change* knowledge, which is a materially riskier operation, so each of the
three is deliberately conservative:

``CLAIM_EVIDENCE``
    Attaches an ``INFERS_FROM`` evidence link. The claim's statement is
    untouched. This is purely additive — the safest possible knowledge change.

``CLAIM_REFINEMENT``
    Creates a **new** claim and supersedes the old one. The original is
    retained, marked ``SUPERSEDED``, and a ``SUPERSEDE`` revision records both
    states. Nothing is overwritten and nothing is deleted; the history remains
    answerable.

``CLAIM_CONFLICT``
    Marks the claim ``DISPUTED`` and attaches the conflicting evidence. It does
    **not** retract, delete, or invalidate the claim, and it never asserts that
    the claim is false. Forge's position is "a human should look at this",
    which is the only position the evidence supports.

The ``INFERS_FROM`` relation is used rather than ``QUOTES`` throughout: a model
concluded that the span bears on the claim, and Phase 1 correctly forbids a
model asserting a verbatim quote. That constraint was written for extraction
and applies here unchanged.
"""

from __future__ import annotations

from typing import Any

from ..activation.activator import ActivationOutcome, ActivationResult
from ..domain import (
    Claim,
    ClaimStatus,
    Derivation,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    Proposal,
    ProposalType,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    Revision,
    record_change,
)
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

EVOLUTION_ACTIVATOR_VERSION = "evolution-activation/0.1.0"

#: Proposal types this module owns. Anything else belongs to Phase 3's
#: activator, which remains untouched.
HANDLED: frozenset[ProposalType] = frozenset(
    {
        ProposalType.CLAIM_EVIDENCE,
        ProposalType.CLAIM_REFINEMENT,
        ProposalType.CLAIM_CONFLICT,
    }
)


class EvolutionActivator:
    """Applies approved evolution proposals to canonical knowledge."""

    version = EVOLUTION_ACTIVATOR_VERSION

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def handles(self, proposal: Proposal) -> bool:
        return proposal.type in HANDLED

    def activate(self, proposal: Proposal) -> ActivationResult:
        """Apply one approved proposal. Never raises; failure is a result."""
        if proposal.status.value == "activated":
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=proposal.activated_entity_type,
                entity_id=proposal.activated_entity_id,
                reason="proposal was already activated",
            )
        if proposal.status.value != "approved":
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason=f"proposal is {proposal.status.value}, not approved",
            )

        claim = self.store.get_claim(proposal.operation.target)
        if claim is None:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason=f"target claim {proposal.operation.target[:12]!r} no longer exists",
            )

        spans = [
            s for s in (self.store.get_span(i) for i in proposal.evidence_span_ids) if s is not None
        ]
        if not spans:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason=(
                    "no evidence span resolves; a knowledge change without resolvable "
                    "evidence must not be applied"
                ),
            )

        try:
            if proposal.type is ProposalType.CLAIM_EVIDENCE:
                return self._attach_evidence(proposal, claim, spans)
            if proposal.type is ProposalType.CLAIM_REFINEMENT:
                return self._refine(proposal, claim, spans)
            return self._flag_conflict(proposal, claim, spans)
        except Exception as exc:
            # A storage failure must never be reported as a successful
            # knowledge change. The proposal stays APPROVED and is retryable.
            log.error("evolution_activation_failed", proposal_id=proposal.id, error=str(exc))
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

    # -- SUPPORTS ----------------------------------------------------------

    def _attach_evidence(
        self, proposal: Proposal, claim: Claim, spans: list[Any]
    ) -> ActivationResult:
        existing = {e.span_id for e in self.store.evidence_for_claim(claim.id)}
        links = [
            self._evidence_link(proposal, claim, span)
            for span in spans
            if span.id not in existing
        ]

        if not links:
            self._mark_activated(proposal, EntityType.CLAIM, claim.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CLAIM,
                entity_id=claim.id,
                reason="every cited span is already evidence for this claim",
                evidence_links=len(existing),
            )

        # put_claim on an existing claim inserts the new links without
        # emitting a second CREATE revision (it only records CREATE when the
        # claim was absent), so the history stays honest.
        self.store.put_claim(claim, links)
        self._record_change(
            claim,
            claim,
            proposal,
            note=f"corroborating evidence added from {len(links)} span(s)",
        )
        self._mark_activated(proposal, EntityType.CLAIM, claim.id)

        log.info("claim_evidence_attached", claim=claim.id[:12], links=len(links))
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CLAIM,
            entity_id=claim.id,
            reason=f"attached {len(links)} corroborating evidence link(s)",
            evidence_links=len(links),
        )

    # -- REFINES -----------------------------------------------------------

    def _refine(self, proposal: Proposal, claim: Claim, spans: list[Any]) -> ActivationResult:
        statement = (proposal.operation.after or "").strip()
        if not statement:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason="refinement proposal carries no replacement statement",
            )

        new_id = Claim.make_id(statement, spans[0].id)
        if new_id == claim.id:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason="refined statement is identical to the existing claim",
            )

        if (existing := self.store.get_claim(new_id)) is not None:
            self._mark_activated(proposal, EntityType.CLAIM, existing.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CLAIM,
                entity_id=existing.id,
                reason="refined claim already exists",
                evidence_links=len(self.store.evidence_for_claim(existing.id)),
            )

        refined = Claim(
            id=new_id,
            statement=statement,
            subject_concept_id=claim.subject_concept_id,
            status=ClaimStatus.ACTIVE,
            origin_proposal_id=proposal.id,
            provenance=self._provenance(proposal, spans),
        )
        links = [self._evidence_link(proposal, refined, span) for span in spans]

        # Supersede first so the SUPERSEDE revision precedes the evidence, then
        # attach evidence with put_claim. Doing it the other way round would
        # make put_claim emit a CREATE revision for a claim that supersede
        # then records again.
        self.store.supersede_claim(claim.id, refined, cause=proposal.id)
        self.store.put_claim(refined, links)
        self._mark_activated(proposal, EntityType.CLAIM, refined.id)

        log.info(
            "claim_refined",
            old=claim.id[:12],
            new=refined.id[:12],
            proposal=proposal.id[:12],
        )
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CLAIM,
            entity_id=refined.id,
            reason=(
                f"refined claim created and {claim.id[:12]} superseded "
                f"(original retained)"
            ),
            evidence_links=len(links),
        )

    # -- POTENTIAL_CONFLICT ------------------------------------------------

    def _flag_conflict(
        self, proposal: Proposal, claim: Claim, spans: list[Any]
    ) -> ActivationResult:
        if claim.status is ClaimStatus.DISPUTED:
            self._mark_activated(proposal, EntityType.CLAIM, claim.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CLAIM,
                entity_id=claim.id,
                reason="claim is already marked disputed",
            )

        disputed = claim.model_copy(update={"status": ClaimStatus.DISPUTED})
        existing = {e.span_id for e in self.store.evidence_for_claim(claim.id)}
        links = [
            self._evidence_link(proposal, disputed, span)
            for span in spans
            if span.id not in existing
        ]

        # The disputing evidence is attached to the claim rather than replacing
        # it: the claim is still what the corpus says, now with a recorded
        # reason to doubt it. Nothing is retracted.
        self.store.put_claim(disputed, links)
        self._record_change(
            claim,
            disputed,
            proposal,
            note="marked DISPUTED by evidence assessment; claim retained, not retracted",
        )
        self._mark_activated(proposal, EntityType.CLAIM, claim.id)

        log.info("claim_disputed", claim=claim.id[:12], spans=len(links))
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CLAIM,
            entity_id=claim.id,
            reason=(
                f"claim marked DISPUTED with {len(links)} conflicting evidence link(s); "
                f"statement retained unchanged"
            ),
            evidence_links=len(links),
        )

    # -- shared ------------------------------------------------------------

    def _evidence_link(self, proposal: Proposal, claim: Claim, span: Any) -> EvidenceLink:
        """``INFERS_FROM``, never ``QUOTES``.

        The model concluded that this span bears on this claim. It did not
        establish that the claim's words appear in the span, and the domain
        layer forbids it from saying so.
        """
        return EvidenceLink(
            id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.INFERS_FROM),
            claim_id=claim.id,
            span_id=span.id,
            relation=EvidenceRelation.INFERS_FROM,
            provenance=self._provenance(proposal, [span]),
        )

    def _provenance(self, proposal: Proposal, spans: list[Any]) -> Provenance:
        source = proposal.provenance
        return Provenance(
            tier=ProvenanceTier.MODEL_INFERENCE,
            derivation=Derivation.MODEL,
            agent="EvolutionActivator",
            model_id=source.model_id or "unknown",
            prompt_version=source.prompt_version,
            schema_version=source.schema_version,
            derivation_key=source.derivation_key,
            workflow_run_id=str(proposal.operation.details.get("workflow_id") or "") or None,
            inputs=tuple(
                ProvenanceInput(
                    entity_type=EntityType.SPAN,
                    entity_id=span.id,
                    tier=ProvenanceTier.SOURCE_FACT,
                )
                for span in spans
            ),
        )

    def _record_change(
        self, before: Claim, after: Claim, proposal: Proposal, *, note: str
    ) -> Revision:
        revision = record_change(
            EntityType.CLAIM,
            after.id,
            before.model_dump(mode="json"),
            after.model_dump(mode="json"),
            cause=proposal.id,
            workflow_run_id=str(proposal.operation.details.get("workflow_id") or "") or None,
            note=note,
        )
        self.store.append_revision(revision)
        return revision

    def _mark_activated(
        self, proposal: Proposal, entity_type: EntityType, entity_id: str
    ) -> Proposal:
        activated = proposal.activate(entity_type=entity_type, entity_id=entity_id)
        self.store.put_proposal(activated)
        return activated
