"""Strict typed schemas for LLM extraction.

Every schema here is ``extra="forbid"`` with required, non-empty primary
fields. The Phase 1 capability spike found the reason: schemas whose fields all
have defaults accept ``{}`` as valid, so a model that produced nothing scores as
a success. The same mistake in the extraction path would silently write empty
knowledge.

A response that does not validate produces ``EXTRACTION_FAILED`` or
``PARTIAL_EXTRACTION``. It is never repaired into plausible-looking data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bumped whenever a schema changes shape. Part of the derivation key, so a
#: schema change invalidates cached extractions rather than mixing formats.
SCHEMA_VERSION = "extract/0.2.0"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExtractedConcept(StrictSchema):
    """A concept the model claims the text is about."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="concept", max_length=40)
    #: Where in the provided text the model found it. Used to check the model
    #: is not inventing content, not to locate the concept precisely.
    mention: str = Field(default="", max_length=400)

    @field_validator("name")
    @classmethod
    def _sane_name(cls, v: str) -> str:
        if not any(ch.isalnum() for ch in v):
            raise ValueError(f"concept name has no alphanumeric content: {v!r}")
        return v


class ExtractedClaim(StrictSchema):
    """A single assertion the text makes."""

    statement: str = Field(min_length=8, max_length=600)
    #: Verbatim supporting text. Required — a claim Forge cannot ground is a
    #: claim Forge must not store.
    evidence_quote: str = Field(min_length=1, max_length=800)
    concept: str = Field(default="", max_length=120)


class ConceptExtractionResponse(StrictSchema):
    concepts: list[ExtractedConcept] = Field(min_length=1, max_length=15)


class ClaimExtractionResponse(StrictSchema):
    claims: list[ExtractedClaim] = Field(min_length=1, max_length=10)


class TerminologyResponse(StrictSchema):
    """Optional terminology/alias extraction."""

    terms: list[str] = Field(min_length=1, max_length=20)
