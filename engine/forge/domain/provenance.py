"""Provenance — recorded structurally, enforced at construction.

The **provenance floor rule**:

    A derived object's provenance strength can never exceed the weakest
    provenance tier among its inputs.

This is what structurally prevents generated content from laundering itself
into evidence. Inference over extracted claims is ``MODEL_INFERENCE``, never
``SOURCE_FACT``; synthesis over inferences is ``SYNTHESIS``.

The rule is enforced *here*, in a pydantic validator on :class:`Provenance`,
not in a service layer and not in documentation. It is impossible to construct
a violating ``Provenance`` object, so it is impossible to persist one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import TIER_STRENGTH, TIERS_WITHOUT_EVIDENCE, Derivation, EntityType, ProvenanceTier


class ProvenanceViolation(Exception):
    """Raised when an operation would breach the provenance rules.

    Always a programming error: it means code tried to assert something more
    strongly than its inputs permit.

    Deliberately **not** a ``ValueError``. Pydantic collects ``ValueError``
    raised inside validators into its own ``ValidationError``, which would bury
    a provenance breach among ordinary field errors and make it impossible to
    catch specifically. Inheriting from ``Exception`` makes it propagate
    unwrapped, so a breach is always loud and always distinguishable.
    """


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProvenanceInput(BaseModel):
    """One input an object was derived from, with the tier it carried."""

    model_config = ConfigDict(frozen=True)

    entity_type: EntityType
    entity_id: str
    tier: ProvenanceTier


def floor_tier(tiers: Iterable[ProvenanceTier]) -> ProvenanceTier | None:
    """Strongest tier a derived object may claim, given its inputs' tiers.

    Returns ``None`` when there are no inputs (nothing constrains the result).
    """
    ranked = sorted(tiers, key=lambda t: TIER_STRENGTH[t])
    return ranked[0] if ranked else None


def violates_floor(tier: ProvenanceTier, inputs: Sequence[ProvenanceInput]) -> bool:
    floor = floor_tier(i.tier for i in inputs)
    if floor is None:
        return False
    return TIER_STRENGTH[tier] > TIER_STRENGTH[floor]


class Provenance(BaseModel):
    """Immutable record of how an assertable object came to exist."""

    model_config = ConfigDict(frozen=True)

    tier: ProvenanceTier
    derivation: Derivation

    #: Component that produced this, e.g. "corpus_indexer" or "ClaimExtractionNode".
    agent: str
    agent_version: str = "0.1.0"

    #: Required when derivation is MODEL; forbidden otherwise.
    model_id: str | None = None
    prompt_version: str | None = None

    inputs: tuple[ProvenanceInput, ...] = ()
    workflow_run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _enforce(self) -> Provenance:
        # 1. Floor rule.
        if violates_floor(self.tier, self.inputs):
            floor = floor_tier(i.tier for i in self.inputs)
            raise ProvenanceViolation(
                f"Provenance floor violated: cannot assert tier {self.tier.value} "
                f"from inputs whose weakest tier is {floor.value if floor else '?'}. "
                f"Inputs: {[(i.entity_type.value, i.tier.value) for i in self.inputs]}"
            )

        # 2. Model derivation must identify its model; deterministic must not.
        if self.derivation is Derivation.MODEL and not self.model_id:
            raise ProvenanceViolation(
                "derivation=MODEL requires model_id — an unattributable model "
                "output cannot be traced and must not be stored"
            )
        if self.derivation is Derivation.DETERMINISTIC and self.model_id:
            raise ProvenanceViolation(
                f"derivation=DETERMINISTIC must not carry model_id ({self.model_id!r}); "
                "deterministic work is not model work"
            )

        # 3. Model-derived output cannot be a SOURCE_FACT. A source fact is
        #    verbatim; the moment a model touches it, it is at best an
        #    extraction.
        if self.derivation is Derivation.MODEL and self.tier is ProvenanceTier.SOURCE_FACT:
            raise ProvenanceViolation(
                "derivation=MODEL cannot produce SOURCE_FACT; verbatim source "
                "content is extracted deterministically"
            )
        return self

    @property
    def requires_evidence(self) -> bool:
        return self.tier not in TIERS_WITHOUT_EVIDENCE

    def derive(
        self,
        *,
        tier: ProvenanceTier,
        agent: str,
        derivation: Derivation,
        entity_type: EntityType,
        entity_id: str,
        model_id: str | None = None,
        prompt_version: str | None = None,
        workflow_run_id: str | None = None,
    ) -> Provenance:
        """Create child provenance with this object recorded as an input.

        Raises :class:`ProvenanceViolation` if ``tier`` exceeds what this
        object's tier permits.
        """
        return Provenance(
            tier=tier,
            derivation=derivation,
            agent=agent,
            model_id=model_id,
            prompt_version=prompt_version,
            inputs=(ProvenanceInput(entity_type=entity_type, entity_id=entity_id, tier=self.tier),),
            workflow_run_id=workflow_run_id or self.workflow_run_id,
        )


def deterministic_provenance(
    agent: str,
    tier: ProvenanceTier = ProvenanceTier.SOURCE_FACT,
    *,
    inputs: Sequence[ProvenanceInput] = (),
    workflow_run_id: str | None = None,
) -> Provenance:
    """Convenience constructor for provenance produced by ordinary software."""
    return Provenance(
        tier=tier,
        derivation=Derivation.DETERMINISTIC,
        agent=agent,
        inputs=tuple(inputs),
        workflow_run_id=workflow_run_id,
    )
