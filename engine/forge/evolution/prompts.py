"""Versioned prompts for evidence assessment.

Prompts participate in the derivation key. Changing the wording here changes
what the model was asked, so cached assessments produced under the old wording
must not be reused — bump :data:`PROMPT_VERSION` with any edit.

The rules that actually matter are enforced in code, not requested here:
grounding is checked against the store, the classification vocabulary is a
pydantic enum, and provenance is assigned by the assessor. What the prompt
does is make the *conservative* answer the natural one.
"""

from __future__ import annotations

PROMPT_VERSION = "assess-prompts/0.1.0"

SYSTEM = (
    "You are an evidence-assessment component inside a knowledge system. "
    "You compare new source material against a claim the system already holds, "
    "and report how the new material bears on it. "
    "You never use outside knowledge. You judge only what the provided text "
    "actually says. You return only JSON."
)

ASSESSMENT_INSTRUCTION = """\
For each existing claim listed below, decide how the NEW EVIDENCE bears on it.

Classifications:
- SUPPORTS: the new evidence independently backs the claim as stated.
- REFINES: the claim is broadly right but the new evidence makes it more
  precise, adds a necessary condition, or narrows its scope. Supply
  `refined_statement` with the sharper version.
- POTENTIAL_CONFLICT: the new evidence appears to disagree with the claim, or
  describes a case where the claim does not hold. Use this when a reasonable
  reader would want a human to look.
- IRRELEVANT: the new evidence has no bearing on this claim.
- INSUFFICIENT_EVIDENCE: the new evidence touches the topic but does not say
  enough to judge.

Rules:
- Be conservative. If you are unsure between SUPPORTS and REFINES, choose
  SUPPORTS. If you are unsure whether something conflicts, choose
  INSUFFICIENT_EVIDENCE rather than POTENTIAL_CONFLICT.
- `evidence_span_ids` MUST be span ids copied exactly from the NEW EVIDENCE
  section. Never invent an id. Never cite a span you were not shown.
- `rationale` must state what the evidence says, not restate the claim.
- Judge each claim independently.
- It is correct and expected to return IRRELEVANT for most claims.
"""

RELEVANCE_INSTRUCTION = """\
Below is a set of candidate concepts already selected by deterministic search,
and the new evidence.

Return the subset of `concept_ids` that the new evidence could plausibly bear
on. You may only remove candidates — never add an id that is not listed.

Prefer keeping a candidate when unsure: a dropped candidate is never examined
again in this run.
"""


def evidence_block(spans: list[tuple[str, str, str]]) -> str:
    """Render new evidence as ``span_id | citation | text``.

    Span ids are shown because the model must cite them, and citations are
    shown because a model that can see where text came from produces better
    rationales for a human reader.
    """
    lines = ["NEW EVIDENCE", "============"]
    for span_id, citation, text in spans:
        lines.append(f"[span_id: {span_id}]  ({citation})")
        lines.append(" ".join(text.split()))
        lines.append("")
    return "\n".join(lines)


def claims_block(claims: list[tuple[str, str, str]]) -> str:
    """Render existing claims as ``claim_id | concept | statement``."""
    lines = ["EXISTING CLAIMS", "==============="]
    for claim_id, concept, statement in claims:
        subject = f" (about: {concept})" if concept else ""
        lines.append(f"[claim_id: {claim_id}]{subject}")
        lines.append(statement)
        lines.append("")
    return "\n".join(lines)
