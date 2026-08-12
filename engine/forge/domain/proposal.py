"""Proposals — recorded intentions to change something, pending human decision.

This is the mechanism that makes ADR-001's segregated write-back real. Forge
may *propose* any change it likes; it may not enact one. A proposal carries
what would change, why, what evidence supports it, and how much the change can
be trusted.

Two properties are load-bearing:

* **A proposal is never self-approving.** Status transitions are explicit and
  validated, and approving one records a revision.
* **Safety classification is derived from provenance, not asserted.** A
  model-generated proposal cannot be labelled ``DETERMINISTIC_VERIFIED``, so
  "the model was very confident" can never be mistaken for "software checked
  this".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..ids import deterministic_id
from .enums import Derivation, EntityType, ProposalStatus, ProposalType, SafetyClass
from .provenance import Provenance, ProvenanceViolation, utc_now


class ProposalTransitionError(Exception):
    """Raised on an invalid status transition. Deliberately not a ValueError,
    for the same reason as ProvenanceViolation: it must not be folded into
    pydantic's field-error reporting."""


#: Terminal statuses cannot transition further.
_ALLOWED_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    ProposalStatus.PENDING: frozenset(
        {ProposalStatus.APPROVED, ProposalStatus.REJECTED, ProposalStatus.SUPERSEDED}
    ),
    ProposalStatus.APPROVED: frozenset({ProposalStatus.SUPERSEDED}),
    ProposalStatus.REJECTED: frozenset({ProposalStatus.SUPERSEDED}),
    ProposalStatus.SUPERSEDED: frozenset(),
}


class ProposedOperation(BaseModel):
    """The concrete change, in a form a human can review before it happens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Verb, e.g. "replace_frontmatter_line", "create_concept", "link_concept".
    action: str
    #: What the operation would apply to — a vault path, concept id, etc.
    target: str
    before: str | None = None
    after: str | None = None
    #: Structured detail for operations that a before/after pair cannot express.
    details: dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    """A change Forge would like to make, awaiting a human decision."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: ProposalType
    status: ProposalStatus = ProposalStatus.PENDING
    safety: SafetyClass

    #: Entity the proposal is about, when it targets something that exists.
    target_entity_type: EntityType | None = None
    target_entity_id: str | None = None

    operation: ProposedOperation
    reason: str

    #: Span ids backing this proposal. Required for anything model-generated:
    #: an unevidenced model proposal is exactly the thing Forge must not make.
    evidence_span_ids: tuple[str, ...] = ()
    #: Source this proposal arose from, for grouping and filtering.
    source_id: str | None = None

    provenance: Provenance
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_note: str | None = None
    superseded_by: str | None = None

    @staticmethod
    def make_id(proposal_type: ProposalType, target: str, fingerprint: str) -> str:
        """Deterministic identity.

        Re-running ingestion or diagnostics must not create a second copy of a
        proposal the user already rejected — so identity is derived from what
        the proposal *is*, not when it was generated.
        """
        return deterministic_id("proposal", proposal_type.value, target, fingerprint)

    @model_validator(mode="after")
    def _check(self) -> Proposal:
        # Safety classification must be earned, not asserted.
        if (
            self.safety is SafetyClass.DETERMINISTIC_VERIFIED
            and self.provenance.derivation is Derivation.MODEL
        ):
            raise ProvenanceViolation(
                f"proposal {self.id} is model-derived and cannot be classified "
                f"DETERMINISTIC_VERIFIED; model confidence is not verification"
            )

        # A model-generated proposal must point at the evidence it came from.
        if self.provenance.derivation is Derivation.MODEL and not self.evidence_span_ids:
            raise ProvenanceViolation(
                f"proposal {self.id} is model-derived but cites no evidence spans; "
                f"unevidenced model proposals must not be created"
            )

        if self.status is ProposalStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError("a superseded proposal must record superseded_by")

        if self.status in (ProposalStatus.APPROVED, ProposalStatus.REJECTED):
            if self.decided_at is None:
                raise ValueError(f"{self.status.value} proposal must record decided_at")
        return self

    # -- transitions -------------------------------------------------------

    def _transition(
        self, to: ProposalStatus, *, by: str | None, note: str | None, superseded_by: str | None = None
    ) -> Proposal:
        if to not in _ALLOWED_TRANSITIONS[self.status]:
            raise ProposalTransitionError(
                f"cannot move proposal {self.id} from {self.status.value} to {to.value}"
            )
        return self.model_copy(
            update={
                "status": to,
                "decided_at": utc_now(),
                "decided_by": by,
                "decision_note": note,
                "superseded_by": superseded_by,
            }
        )

    def approve(self, *, by: str = "cli", note: str | None = None) -> Proposal:
        """Approving records the decision. It does **not** apply the change.

        Application is a separate, explicitly-flagged step (ADR-001 D2), so an
        approval can be recorded now and enacted later, or never.
        """
        return self._transition(ProposalStatus.APPROVED, by=by, note=note)

    def reject(self, *, by: str = "cli", note: str | None = None) -> Proposal:
        return self._transition(ProposalStatus.REJECTED, by=by, note=note)

    def supersede(self, by_proposal_id: str, *, by: str = "system") -> Proposal:
        return self._transition(
            ProposalStatus.SUPERSEDED,
            by=by,
            note=f"superseded by {by_proposal_id}",
            superseded_by=by_proposal_id,
        )

    @property
    def is_decided(self) -> bool:
        return self.status in (ProposalStatus.APPROVED, ProposalStatus.REJECTED)

    @property
    def auto_applicable(self) -> bool:
        """Whether this proposal is *eligible* for automated application.

        Eligibility is not permission: application still requires an approved
        status and an explicit ``--apply`` flag. Only deterministic, verified
        changes ever qualify.
        """
        return (
            self.safety is SafetyClass.DETERMINISTIC_VERIFIED
            and self.status is ProposalStatus.APPROVED
        )

    def summary(self) -> str:
        return f"[{self.status.value}] {self.type.value} {self.operation.target} — {self.reason}"
