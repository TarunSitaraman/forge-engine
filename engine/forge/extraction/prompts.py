"""Versioned extraction prompts.

Prompts are versioned because they participate in the derivation key: changing
a prompt must invalidate cached extractions, or the store ends up holding a mix
of results from different instructions with no way to tell them apart.

Kept as data, not logic. The rules that matter — what may be asserted, what
must be evidenced — are enforced in code, not requested politely here.
"""

from __future__ import annotations

PROMPT_VERSION = "extract-prompts/0.2.0"

SYSTEM = (
    "You are a precise information-extraction component inside a knowledge "
    "system. You extract only what the provided text actually states. "
    "You never add outside knowledge. You return only JSON."
)

CONCEPT_INSTRUCTION = """\
List the distinct technical concepts this text is about.

Rules:
- Only concepts the text actually discusses. Do not add related concepts you
  happen to know about.
- Use the shortest canonical name (e.g. "Retrieval Augmented Generation", not
  "the RAG technique described here").
- `kind` should be one of: concept, technology, pattern, algorithm,
  data_structure, project, person.
- `mention` must be a short phrase copied verbatim from the text.
"""

CLAIM_INSTRUCTION = """\
List the individual factual assertions this text makes.

Rules:
- One assertion per claim. Split compound statements.
- `evidence_quote` MUST be copied verbatim from the text. If you cannot quote
  it, do not make the claim.
- Do not infer beyond what is written.
- `concept` is the main concept the claim is about, if clear.
"""

TERMINOLOGY_INSTRUCTION = """\
List technical terms, acronyms, and their expansions used in this text.
Return the terms exactly as they appear.
"""


def build_messages(instruction: str, text: str) -> list[tuple[str, str]]:
    """(role, content) pairs. Kept transport-agnostic."""
    return [
        ("system", SYSTEM),
        ("user", f"{instruction}\n\n--- TEXT START ---\n{text}\n--- TEXT END ---"),
    ]
