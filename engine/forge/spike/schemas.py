"""Response schemas for the local-model capability spike.

These are deliberately small. The spike measures whether a local model can
produce *reliable structured output at all*; it is not a prompt-optimization
exercise. Large schemas would conflate "the model is weak" with "the schema
was demanding".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpikeSchema(BaseModel):
    """Base for spike response schemas.

    ``extra="forbid"`` is deliberate. Without it, a model replying ``{}`` or
    ``{"bogus": 1}`` validates against any schema whose fields all have
    defaults, and the spike reports a high success rate for a model that
    produced nothing. The primary collections below are also required and
    non-empty for the same reason: "extracted zero concepts from a text full of
    concepts" is a failure, and a spike that scores it as success is worse than
    no spike at all.
    """

    model_config = ConfigDict(extra="forbid")


class ExtractedConcept(BaseModel):
    name: str = Field(description="Short canonical name of the concept")
    kind: str = Field(description="One of: pattern, algorithm, data_structure, technology, concept")


class ConceptExtraction(SpikeSchema):
    """Task 1 — structured concept extraction."""

    concepts: list[ExtractedConcept] = Field(min_length=1, max_length=12)


class ExtractedClaim(BaseModel):
    statement: str = Field(description="A single assertion made by the text")
    #: Verbatim supporting text. Requested so the spike can check whether the
    #: model can ground a claim at all — the precondition for provenance.
    evidence_quote: str = Field(default="", description="Verbatim quote supporting the statement")


class ClaimExtraction(SpikeSchema):
    """Task 2 — simple claim extraction."""

    claims: list[ExtractedClaim] = Field(min_length=1, max_length=8)


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str = Field(description="One of: PART_OF, DEPENDS_ON, REQUIRES, IMPLEMENTS, EXPLAINS, RELATED_TO")


class RelationshipExtraction(SpikeSchema):
    """Task 3 — relationship extraction."""

    relationships: list[ExtractedRelationship] = Field(min_length=1, max_length=10)


class SmallSynthesis(SpikeSchema):
    """Task 4 — small synthesis task."""

    summary: str = Field(min_length=1, description="2-3 sentence synthesis across the provided notes")
    key_points: list[str] = Field(default_factory=list, max_length=5)
