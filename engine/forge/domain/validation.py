"""Cross-entity invariants.

Single-entity rules live in the models themselves (pydantic validators).
Rules that span entities — a claim needing evidence, a link needing endpoints —
live here, and are called by the storage layer before any write.

The point is that these are enforced *at the domain boundary*. It must not be
possible to persist a violating object by going around a service.
"""

from __future__ import annotations

from collections.abc import Sequence

from .entities import Claim, ClaimLink, EvidenceLink
from .enums import TIERS_WITHOUT_EVIDENCE, ClaimStatus
from .provenance import ProvenanceViolation


class ValidationError(ValueError):
    """Raised when a cross-entity invariant is violated."""


def validate_claim(claim: Claim, evidence: Sequence[EvidenceLink]) -> None:
    """A claim must be evidenced unless its tier is exempt.

    Only ``USER_ASSERTION`` is exempt: the user saying so *is* the warrant.
    Everything else — extraction, inference, synthesis — must point at a span.
    This is the mechanical enforcement of "every important generated claim
    should be traceable to evidence".
    """
    if claim.provenance.tier in TIERS_WITHOUT_EVIDENCE:
        return

    linked = [e for e in evidence if e.claim_id == claim.id]
    if not linked:
        raise ProvenanceViolation(
            f"claim {claim.id} has tier {claim.provenance.tier.value} which requires "
            f"evidence, but no EvidenceLink was supplied. Unevidenced non-user "
            f"claims must not be stored."
        )


def validate_supersession(old: Claim, new: Claim) -> None:
    """Supersession must preserve the old claim rather than delete it."""
    if old.id == new.id:
        raise ValidationError("a claim cannot supersede itself")
    if old.status is not ClaimStatus.SUPERSEDED:
        raise ValidationError(
            f"claim {old.id} must be marked SUPERSEDED before {new.id} replaces it"
        )
    if old.superseded_by != new.id:
        raise ValidationError(
            f"claim {old.id}.superseded_by is {old.superseded_by!r}, expected {new.id!r}"
        )
    if old.valid_to is None:
        raise ValidationError(f"superseded claim {old.id} must have valid_to set")


def validate_claim_link(link: ClaimLink, known_ids: set[str]) -> None:
    """Both endpoints must exist. A dangling edge is a corrupt graph."""
    missing = [i for i in (link.from_id, link.to_id) if i not in known_ids]
    if missing:
        raise ValidationError(
            f"ClaimLink {link.type.value} references unknown endpoint(s): {missing}"
        )
