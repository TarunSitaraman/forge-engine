"""Proposal activation — approved decisions become canonical knowledge.

This closes the loop Phase 2 deliberately left open:

    SOURCE -> DOCUMENT -> SPAN -> EXTRACTION -> PROPOSAL -> APPROVAL
                                                              |
                                                              v
                                                    CANONICAL KNOWLEDGE

Four properties are load-bearing, and each one is enforced rather than
intended:

* **Idempotent.** Entity identity is deterministic, derived from what the
  entity *is*. Approving twice, re-indexing, and approving again converge on
  the same single concept, claim, and evidence link — with no duplicate
  revisions.
* **Never claims more than happened.** The proposal is marked ``ACTIVATED``
  only *after* the entity is persisted, inside the same transaction. A
  persistence failure produces an explicit ``FAILED`` outcome and leaves the
  proposal ``APPROVED``, so it can be retried.
* **Nothing becomes an orphan.** Every activated entity records its origin
  proposal and the spans that evidenced it, so "which proposal created this
  concept?" and "which span caused this claim?" are both answerable.
* **Ambiguity survives approval.** An approved proposal for an unresolved
  collision is refused, not guessed. Approval is a decision to accept a
  proposal, not a decision about which `Heap` was meant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..domain import (
    Claim,
    ClaimStatus,
    Concept,
    ConceptKind,
    Derivation,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    IdentityState,
    MatchKind,
    Proposal,
    ProposalStatus,
    ProposalType,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    SafetyClass,
)
from ..identity.service import IdentityService
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

ACTIVATOR_VERSION = "activation/0.3.0"


class ActivationOutcome(str, Enum):
    """What happened to one proposal."""

    CREATED = "created"
    #: Entity already existed with identical identity — the idempotent path.
    ALREADY_ACTIVE = "already_active"
    #: Refused for a stated reason (not approved, ambiguous, unsupported type).
    REFUSED = "refused"
    #: Attempted and failed. The proposal stays APPROVED for retry.
    FAILED = "failed"


@dataclass
class ActivationResult:
    """Outcome for one proposal."""

    proposal_id: str
    outcome: ActivationOutcome
    entity_type: EntityType | None = None
    entity_id: str | None = None
    reason: str = ""
    evidence_links: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome in (ActivationOutcome.CREATED, ActivationOutcome.ALREADY_ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "outcome": self.outcome.value,
            "entity_type": self.entity_type.value if self.entity_type else None,
            "entity_id": self.entity_id,
            "reason": self.reason,
            "evidence_links": self.evidence_links,
        }


@dataclass
class ActivationReport:
    results: list[ActivationResult] = field(default_factory=list)

    def add(self, result: ActivationResult) -> ActivationResult:
        self.results.append(result)
        return result

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.outcome.value] = out.get(r.outcome.value, 0) + 1
        return dict(sorted(out.items()))

    @property
    def created(self) -> list[ActivationResult]:
        return [r for r in self.results if r.outcome is ActivationOutcome.CREATED]

    @property
    def failed(self) -> list[ActivationResult]:
        return [r for r in self.results if r.outcome is ActivationOutcome.FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "results": [r.to_dict() for r in self.results],
        }


class ProposalActivator:
    """Turns approved proposals into canonical Concepts and Claims."""

    version = ACTIVATOR_VERSION

    def __init__(self, store: SqliteStore, *, identity: IdentityService | None = None) -> None:
        self.store = store
        self.identity = identity or IdentityService()

    # -- entry points ------------------------------------------------------

    def activate(self, proposal: Proposal) -> ActivationResult:
        """Activate one proposal. Never raises for an expected refusal."""
        if (refusal := self._refusal(proposal)) is not None:
            return ActivationResult(
                proposal_id=proposal.id, outcome=ActivationOutcome.REFUSED, reason=refusal
            )

        try:
            if proposal.type is ProposalType.NEW_CONCEPT:
                return self._activate_concept(proposal)
            if proposal.type is ProposalType.NEW_CLAIM:
                return self._activate_claim(proposal)
            if proposal.type is ProposalType.CONCEPT_MATCH:
                return self._activate_concept_match(proposal)
        except Exception as exc:
            # A failure must never be reported as success, and must leave the
            # proposal APPROVED so it can be retried once the cause is fixed.
            log.error("activation_failed", proposal_id=proposal.id, error=str(exc))
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.REFUSED,
            reason=f"no activation path for proposal type {proposal.type.value}",
        )

    def activate_all(self, proposals: Sequence[Proposal]) -> ActivationReport:
        report = ActivationReport()
        for proposal in proposals:
            report.add(self.activate(proposal))
        log.info("activation_run", **report.counts())
        return report

    def activate_approved(self, *, limit: int = 500) -> ActivationReport:
        """Activate every approved proposal awaiting activation."""
        pending = [
            p
            for p in self.store.list_proposals(status=ProposalStatus.APPROVED, limit=limit)
            if p.awaiting_activation
        ]
        return self.activate_all(pending)

    # -- concept -----------------------------------------------------------

    def _activate_concept(self, proposal: Proposal) -> ActivationResult:
        name = proposal.operation.target.strip()
        namespace = self._namespace_for(name)

        concept_id = Concept.make_id(name, namespace)
        if (existing := self.store.get_concept(concept_id)) is not None:
            # Idempotent path: the concept this proposal describes already
            # exists. Record the activation link, but create nothing new.
            self._mark_activated(proposal, EntityType.CONCEPT, existing.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CONCEPT,
                entity_id=existing.id,
                reason=f"concept {existing.qualified_name!r} already exists",
            )

        concept = Concept(
            id=concept_id,
            canonical_name=name,
            kind=_concept_kind(proposal.operation.details.get("kind")),
            namespace=namespace,
            vault_path=self._vault_path_for(name, namespace),
            origin_proposal_id=proposal.id,
            origin_span_ids=proposal.evidence_span_ids,
            provenance=self._provenance(proposal, ProvenanceTier.MODEL_INFERENCE),
        )
        self.store.put_concept(concept)
        self._mark_activated(proposal, EntityType.CONCEPT, concept.id)

        log.info("concept_activated", concept=concept.qualified_name, proposal=proposal.id)
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CONCEPT,
            entity_id=concept.id,
            reason=f"created concept {concept.qualified_name!r}",
        )

    def _activate_concept_match(self, proposal: Proposal) -> ActivationResult:
        """A match proposal links an extracted name to an existing concept.

        It creates no new concept — it registers the extracted name as an
        alias, which is the whole point of having decided they are the same
        thing.

        One exception: a proposal that was *ambiguous* when it was created has
        no match target at all, because the matcher deliberately refused to
        pick one. Once the user resolves the collision, the right action is to
        create the concept they chose — not to look for a match target that
        was never recorded.
        """
        if proposal.safety is SafetyClass.AMBIGUOUS:
            resolution = self.identity.resolve(proposal.operation.target)
            if resolution.state is IdentityState.RESOLVED_BY_USER:
                return self._activate_concept(proposal)

        details = proposal.operation.details
        target_name = proposal.operation.after or details.get("match_kind")
        concept = None
        if proposal.target_entity_id:
            concept = self.store.get_concept(proposal.target_entity_id)
        if concept is None and isinstance(target_name, str):
            concept = self.store.get_concept_by_name(target_name)

        if concept is None:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason=(
                    f"match target {target_name!r} is not a canonical concept yet; "
                    f"activate the concept it matches first"
                ),
            )

        alias = proposal.operation.target.strip()
        if alias.casefold() in {a.casefold() for a in concept.aliases}:
            self._mark_activated(proposal, EntityType.CONCEPT, concept.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CONCEPT,
                entity_id=concept.id,
                reason=f"{alias!r} is already an alias of {concept.qualified_name!r}",
            )

        updated = concept.model_copy(update={"aliases": (*concept.aliases, alias)})
        self.store.put_concept(updated)  # writes a CHANGE revision
        self._mark_activated(proposal, EntityType.CONCEPT, concept.id)
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CONCEPT,
            entity_id=concept.id,
            reason=f"registered {alias!r} as an alias of {concept.qualified_name!r}",
        )

    # -- claim -------------------------------------------------------------

    def _activate_claim(self, proposal: Proposal) -> ActivationResult:
        statement = (proposal.operation.after or proposal.operation.target).strip()
        span_ids = list(proposal.evidence_span_ids)

        # A claim with no resolvable span is not a claim Forge may hold.
        spans = [s for s in (self.store.get_span(sid) for sid in span_ids) if s is not None]
        if not spans:
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.REFUSED,
                reason=(
                    f"no evidence span resolves ({span_ids or 'none cited'}); a claim without "
                    f"resolvable evidence must not become canonical knowledge"
                ),
            )

        claim_id = Claim.make_id(statement, spans[0].id)
        subject = self._subject_concept_id(proposal)

        if (existing := self.store.get_claim(claim_id)) is not None:
            self._mark_activated(proposal, EntityType.CLAIM, existing.id)
            return ActivationResult(
                proposal_id=proposal.id,
                outcome=ActivationOutcome.ALREADY_ACTIVE,
                entity_type=EntityType.CLAIM,
                entity_id=existing.id,
                reason="claim already exists with identical statement and evidence",
                evidence_links=len(self.store.evidence_for_claim(existing.id)),
            )

        provenance = self._provenance(proposal, ProvenanceTier.EXTRACTED_CLAIM, spans=spans)
        claim = Claim(
            id=claim_id,
            statement=statement,
            subject_concept_id=subject,
            status=ClaimStatus.ACTIVE,
            origin_proposal_id=proposal.id,
            provenance=provenance,
        )

        quote = str(proposal.operation.details.get("evidence_quote", ""))
        evidence = [self._evidence_link(claim, span, quote) for span in spans]

        # put_claim validates that an EXTRACTED_CLAIM has evidence, so a claim
        # and its evidence are committed together or not at all.
        self.store.put_claim(claim, evidence)
        self._mark_activated(proposal, EntityType.CLAIM, claim.id)

        log.info("claim_activated", claim=claim.id, spans=len(spans), proposal=proposal.id)
        return ActivationResult(
            proposal_id=proposal.id,
            outcome=ActivationOutcome.CREATED,
            entity_type=EntityType.CLAIM,
            entity_id=claim.id,
            reason=f"created claim with {len(evidence)} evidence link(s)",
            evidence_links=len(evidence),
        )

    # -- shared ------------------------------------------------------------

    def _evidence_link(self, claim: Claim, span: Any, quote: str) -> EvidenceLink:
        """Build the link between a claim and the span that evidences it.

        The link carries **deterministic** provenance, even though the claim
        itself is model-derived. These describe different things: the claim is
        a model's assertion, while the link asserts "this text really is in
        this span" — which the activator establishes by string comparison, in
        code, right here.

        Phase 1 forbids a model asserting ``QUOTES`` precisely because a model
        cannot be trusted to have quoted verbatim. Software checking the same
        thing can, and that check is what this provenance records.
        """
        relation = _evidence_relation(quote, span.text)
        return EvidenceLink(
            id=EvidenceLink.make_id(claim.id, span.id, relation),
            claim_id=claim.id,
            span_id=span.id,
            relation=relation,
            provenance=Provenance(
                tier=ProvenanceTier.EXTRACTED_CLAIM,
                derivation=Derivation.DETERMINISTIC,
                agent="ProposalActivator.evidence",
                agent_version=ACTIVATOR_VERSION,
                inputs=(
                    ProvenanceInput(
                        entity_type=EntityType.SPAN,
                        entity_id=span.id,
                        tier=ProvenanceTier.SOURCE_FACT,
                    ),
                ),
            ),
        )

    def _refusal(self, proposal: Proposal) -> str | None:
        """Reasons a proposal must not be activated."""
        if proposal.status is ProposalStatus.ACTIVATED:
            return None  # handled idempotently below, not a refusal
        if proposal.status is not ProposalStatus.APPROVED:
            return f"not approved (status: {proposal.status.value})"

        if proposal.type is ProposalType.METADATA_REPAIR:
            return (
                "metadata repairs are applied to Markdown, not activated as canonical "
                "knowledge; use `forge proposals approve --apply`"
            )

        # Approving an ambiguous proposal is not a decision about which concept
        # was meant. Activating it anyway would be exactly the silent guess
        # Forge exists to avoid.
        if proposal.safety is SafetyClass.AMBIGUOUS:
            name = proposal.operation.target
            resolution = self.identity.resolve(name)
            if resolution.state is not IdentityState.RESOLVED_BY_USER:
                candidates = [c.qualified_name for c in resolution.candidates] or [
                    c.get("qualified_name") or c.get("canonical_name")
                    for c in proposal.operation.details.get("candidates", [])
                ]
                return (
                    f"{name!r} is an unresolved collision and cannot be activated; "
                    f"decide it first: forge identity decide {name!r} <one of {candidates}>"
                )
        return None

    def _mark_activated(
        self, proposal: Proposal, entity_type: EntityType, entity_id: str
    ) -> None:
        """Record activation, tolerating an already-activated proposal.

        Re-activating is a no-op rather than an error: the idempotency
        guarantee is about the *canonical model*, and a second approval of the
        same proposal must converge rather than fail.
        """
        if proposal.status is ProposalStatus.ACTIVATED:
            return
        self.store.put_proposal(proposal.activate(entity_type, entity_id))

    def _provenance(
        self, proposal: Proposal, tier: ProvenanceTier, *, spans: Sequence[Any] = ()
    ) -> Provenance:
        """Provenance for an activated entity.

        Inherits the proposal's derivation and model, so an entity created from
        a model-extracted proposal is permanently marked as model-derived. The
        activation step adds a decision, not a stronger warrant.
        """
        origin = proposal.provenance
        inputs = tuple(
            ProvenanceInput(
                entity_type=EntityType.SPAN, entity_id=span.id, tier=ProvenanceTier.SOURCE_FACT
            )
            for span in spans
        ) or origin.inputs

        return Provenance(
            tier=tier,
            derivation=origin.derivation,
            agent="ProposalActivator",
            agent_version=ACTIVATOR_VERSION,
            model_id=origin.model_id,
            prompt_version=origin.prompt_version,
            inputs=inputs,
            workflow_run_id=origin.workflow_run_id,
        )

    def _namespace_for(self, name: str) -> str | None:
        resolution = self.identity.resolve(name)
        if resolution.state is IdentityState.RESOLVED_BY_USER and resolution.identity:
            return resolution.identity.namespace
        return None

    def _vault_path_for(self, name: str, namespace: str | None) -> str | None:
        resolution = self.identity.resolve(name)
        if resolution.identity and resolution.identity.namespace == namespace:
            return resolution.identity.vault_path
        return None

    def _subject_concept_id(self, proposal: Proposal) -> str | None:
        """Link a claim to its concept when that concept already exists.

        Deliberately does not *create* the concept: a claim's mention of a
        concept name is not an approval to add that concept to the model.
        """
        name = str(proposal.operation.details.get("concept", "")).strip()
        if not name:
            return None
        namespace = self._namespace_for(name)
        concept = self.store.get_concept(Concept.make_id(name, namespace))
        if concept is None:
            concept = self.store.get_concept_by_name(name)
        return concept.id if concept else None


# -- helpers ---------------------------------------------------------------


def _concept_kind(raw: Any) -> ConceptKind:
    if isinstance(raw, str):
        try:
            return ConceptKind(raw.strip().lower())
        except ValueError:
            pass
    return ConceptKind.CONCEPT


def _evidence_relation(quote: str, span_text: str) -> EvidenceRelation:
    """Verbatim quotes are QUOTES; anything else is a paraphrase.

    Checked deterministically here rather than trusting the model's own claim
    about whether it quoted — the domain layer forbids a model asserting
    ``QUOTES`` for exactly that reason.
    """
    if quote and " ".join(quote.split()).lower() in " ".join(span_text.split()).lower():
        return EvidenceRelation.QUOTES
    return EvidenceRelation.PARAPHRASES
