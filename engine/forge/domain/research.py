"""Phase 9 entities: `Question`, `Synthesis`, and the derived `KnowledgeGap`.

Reserved in the canonical model since Phase 1 and populated here. Two of the
three are stored entities; the third deliberately is not.

**`KnowledgeGap` is computed, never stored.** The model doc calls it "a derived
observation ... computed by deterministic graph queries, not generated", and
storing it would create a second thing to keep in sync with the graph it
describes: a stored gap is wrong the moment someone adds the missing claim, and
nothing would notice. Recomputing is cheap and cannot go stale. The cost is
that a user cannot yet dismiss a gap they disagree with, which is a real
product need and a deliberate follow-up rather than an oversight.

**`Synthesis` is the highest-risk object in the model**, because it is the one
most likely to be mistaken for evidence. Its invariant is enforced here and in
the store: it goes stale the moment any constituent claim changes status or
confidence, computed deterministically, never by a model.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field, model_validator

from ..ids import deterministic_id
from .entities import Entity
from .enums import GapKind, ProvenanceTier, QuestionStatus, SynthesisScope
from .provenance import Provenance, ProvenanceViolation, utc_now


class Question(Entity):
    """A research question. Asked by a human, answered by claims.

    Provenance is required and must be `USER_ASSERTION`: a question is
    something the user wants to know, and a model inventing questions and
    filing them alongside the user's own would quietly change what the gap
    report is measuring. A model may *suggest* a question; a human asking it
    is what makes it one.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    status: QuestionStatus = QuestionStatus.OPEN
    provenance: Provenance
    tags: tuple[str, ...] = ()
    note: str | None = None
    #: Concepts the question is about, when the user says so. Used to scope
    #: retrieval; never inferred here.
    concept_ids: tuple[str, ...] = ()
    resolved_at: datetime | None = None

    @staticmethod
    def make_id(text: str) -> str:
        return deterministic_id("question", text.strip().casefold())

    @model_validator(mode="after")
    def _asked_by_a_human(self) -> Question:
        if self.provenance.tier is not ProvenanceTier.USER_ASSERTION:
            raise ProvenanceViolation(
                "a Question is something the user asked, so its tier must be "
                f"USER_ASSERTION, not {self.provenance.tier.value}"
            )
        if self.status is QuestionStatus.ANSWERED and self.resolved_at is None:
            raise ValueError("an answered question must record when it was resolved")
        return self


class Synthesis(Entity):
    """A generated aggregate over claims.

    `stale` is not a hint. It means the claims this was written from have
    changed since it was written, so the text may assert something the model
    no longer holds. A stale synthesis is never silently hidden and never
    silently trusted: it is shown, marked, with what changed underneath it.
    """

    model_config = ConfigDict(extra="forbid")

    scope: SynthesisScope
    #: The concept or question this is about. `TOPIC` scope has no single id.
    scope_id: str | None = None
    body: str = Field(min_length=1)
    source_claim_ids: tuple[str, ...] = ()
    provenance: Provenance
    prompt_version: str | None = None
    generated_at: datetime = Field(default_factory=utc_now)
    stale: bool = False
    #: Why it went stale, in the words of the deterministic check. Empty while
    #: fresh. A bare `stale = true` tells a reader nothing about what moved.
    stale_reason: str | None = None
    superseded_by: str | None = None

    @staticmethod
    def make_id(scope: SynthesisScope, scope_id: str, generated_at: datetime) -> str:
        return deterministic_id("synthesis", scope.value, scope_id, generated_at.isoformat())

    @model_validator(mode="after")
    def _shape(self) -> Synthesis:
        if self.scope is not SynthesisScope.TOPIC and not self.scope_id:
            raise ValueError(f"a {self.scope.value}-scoped synthesis needs a scope_id")
        if not self.source_claim_ids:
            raise ValueError(
                "a synthesis with no constituent claims cannot be checked for "
                "staleness, which is the one invariant it has"
            )
        if self.provenance.tier is not ProvenanceTier.SYNTHESIS:
            raise ProvenanceViolation(
                "a Synthesis is tier SYNTHESIS by definition, not "
                f"{self.provenance.tier.value}"
            )
        if self.stale and not self.stale_reason:
            raise ValueError("a stale synthesis must say what made it stale")
        return self


class KnowledgeGap(Entity):
    """A structural observation about what the model does not hold.

    Derived, not stored. Its `id` is deterministic from the kind and subject so
    two runs over an unchanged graph produce identical gaps, which is what
    makes a gap report diffable between runs.

    **A gap is never a judgement.** "This concept has no claims" is a fact
    about the graph; whether it matters is the user's call. Nothing in Forge
    acts on a gap.
    """

    model_config = ConfigDict(extra="forbid")

    kind: GapKind
    #: The entity the gap is about: a concept id, question id, or claim id.
    subject_id: str
    subject_type: str
    subject_label: str
    detail: str
    #: Ordering hint only, not a probability. Higher means "more likely worth
    #: a human's attention", by rules stated in the detector.
    weight: float = 0.0
    evidence: tuple[str, ...] = ()

    @staticmethod
    def make_id(kind: GapKind, subject_id: str) -> str:
        return deterministic_id("gap", kind.value, subject_id)
