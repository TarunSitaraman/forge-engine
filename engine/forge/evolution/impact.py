"""Assessment -> impact. Deterministic, by design.

The model says how new evidence relates to *one claim*. Deciding what that
means for the knowledge base as a whole is a different question, and it is a
policy question rather than a semantic one — so ordinary software answers it.

Asking a model to also classify overall impact would add a second call, a
second failure mode, and a second thing to audit, in exchange for a decision
that is nine lines of precedence rules. It would also let the model overrule
its own conservatism: a run containing one POTENTIAL_CONFLICT must route to a
human regardless of how many SUPPORTS accompany it, and that guarantee should
not depend on a generated token.

**No confidence scores.** The brief forbids inventing one and it is right to:
a number a model emits about its own certainty is not a measurement, and
attaching one would make the output look calibrated when it is not. Outcomes
are categorical, and provenance carries the rest.
"""

from __future__ import annotations

from typing import Sequence

from ..domain import AssessmentClass, AssessmentRecord, ImpactClass

#: Assessment classes that produce a reviewable proposal. The other two —
#: IRRELEVANT and INSUFFICIENT_EVIDENCE — are honest outcomes that correctly
#: produce nothing.
ACTIONABLE: frozenset[AssessmentClass] = frozenset(
    {
        AssessmentClass.SUPPORTS,
        AssessmentClass.REFINES,
        AssessmentClass.POTENTIAL_CONFLICT,
    }
)

#: Precedence, strongest first. A single potential conflict dominates any
#: number of supports: the workflow must stop for a human even when most of
#: the evidence was agreeable.
_PRECEDENCE: tuple[tuple[AssessmentClass, ImpactClass], ...] = (
    (AssessmentClass.POTENTIAL_CONFLICT, ImpactClass.POTENTIAL_CONFLICT),
    (AssessmentClass.REFINES, ImpactClass.REFINES),
    (AssessmentClass.SUPPORTS, ImpactClass.SUPPORTS),
)


def impact_of(assessment: AssessmentClass) -> ImpactClass:
    """What one assessment means on its own."""
    for cls, impact in _PRECEDENCE:
        if assessment is cls:
            return impact
    return ImpactClass.NO_MATERIAL_CHANGE


def classify_impact(
    assessments: Sequence[AssessmentRecord],
    *,
    claims_examined: int,
    has_new_evidence: bool = True,
) -> ImpactClass:
    """Overall impact of one evolution run.

    ``claims_examined`` distinguishes the two ways a run can produce no
    change, which look identical in the assessment list but mean opposite
    things:

    * Claims existed and none was affected -> ``NO_MATERIAL_CHANGE``.
    * No existing claim was even related -> ``NEW_KNOWLEDGE``: the evidence is
      about something Forge does not yet know, which is a finding, not a
      non-event.
    """
    present = {a.classification for a in assessments}
    for cls, impact in _PRECEDENCE:
        if cls in present:
            return impact

    if has_new_evidence and claims_examined == 0:
        return ImpactClass.NEW_KNOWLEDGE
    return ImpactClass.NO_MATERIAL_CHANGE


def requires_human_review(impact: ImpactClass) -> bool:
    """Whether policy forces a stop.

    Only ``POTENTIAL_CONFLICT`` forces it. Everything else still *produces*
    proposals, and proposals always require approval before activation — so a
    human decides in every case. The difference is that a conflict halts the
    workflow itself rather than leaving a proposal in the queue.
    """
    return impact is ImpactClass.POTENTIAL_CONFLICT


def actionable(assessments: Sequence[AssessmentRecord]) -> list[AssessmentRecord]:
    return [a for a in assessments if a.classification in ACTIONABLE]
