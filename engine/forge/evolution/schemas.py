"""Strict schemas for evidence assessment.

Same discipline as the Phase 2 extraction schemas, for the same reason: a
schema whose fields all have defaults accepts ``{}`` as valid, so a model that
produced nothing scores as a success. Here that would mean silently concluding
"no impact" whenever the model failed — the most dangerous possible default,
because it is indistinguishable from a real answer.

Every field that carries meaning is required and non-empty.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..domain import AssessmentClass

#: Part of the derivation key. Bump on any shape change so cached assessments
#: from an older schema are recomputed rather than mixed in.
SCHEMA_VERSION = "assess/0.1.0"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClaimAssessment(StrictSchema):
    """The model's judgement about one existing claim.

    ``evidence_span_ids`` is required and is checked against the store: a model
    that cites a span which does not exist has its assessment rejected, not
    repaired. Repairing a hallucinated citation would mean inventing the
    evidence for a knowledge change, which is the exact failure this system
    exists to prevent.
    """

    claim_id: str = Field(min_length=1, max_length=64)
    classification: AssessmentClass
    #: Why. Shown to the human reviewer, so it must be substantive — a
    #: one-word rationale is not reviewable.
    rationale: str = Field(min_length=12, max_length=800)
    #: Spans from the *new* evidence that justify this classification.
    evidence_span_ids: list[str] = Field(min_length=1, max_length=10)
    #: Optional sharper statement, required by the caller when the
    #: classification is REFINES (enforced in code, not requested politely).
    refined_statement: str = Field(default="", max_length=600)

    @field_validator("evidence_span_ids")
    @classmethod
    def _no_blank_ids(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("evidence_span_ids must contain at least one non-empty id")
        return cleaned


class AssessmentResponse(StrictSchema):
    """One model call may assess several claims at once."""

    assessments: list[ClaimAssessment] = Field(min_length=1, max_length=12)


class RelevanceJudgement(StrictSchema):
    """Optional LLM refinement of the deterministic candidate set.

    The model may only *narrow* what deterministic selection already found. It
    is never given the corpus and asked what is relevant — that would be both
    expensive and ungroundable.
    """

    concept_ids: list[str] = Field(min_length=0, max_length=20)
    rationale: str = Field(min_length=1, max_length=400)
