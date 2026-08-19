"""Versioned extraction prompts.

Prompts are versioned because they participate in the derivation key: changing
a prompt must invalidate cached extractions, or the store ends up holding a mix
of results from different instructions with no way to tell them apart.

Kept as data, not logic. The rules that matter — what may be asserted, what
must be evidenced — are enforced in code, not requested politely here.
"""

from __future__ import annotations

PROMPT_VERSION = "extract-prompts/0.3.0"

SYSTEM = (
    "You are a precise information-extraction component inside a knowledge "
    "system. You extract only what the provided text actually states. "
    "You never add outside knowledge. You return only JSON."
)

CONCEPT_INSTRUCTION = """\
List the distinct technical concepts this text is about.

A concept here means something that would deserve its own reference page: a
named technology, technique, pattern, algorithm, data structure, architectural
idea, or role. Prefer few and significant over many and shallow. If the text
is about three things, return three.

Do NOT list:
- generic words that merely appear in the text (e.g. "RAM", "HTML", "Answer",
  "Vector", "Fluency", "schemas", "Client query")
- commands, flags, parameters, configuration keys, environment variables, type
  names, or file names (e.g. "git commit", "maxmemory", "VARCHAR(n)",
  ".dockerignore", "terminationGracePeriodSeconds")
- section headings, document structure, or formatting elements
- two concepts joined by "and" — split them or omit them

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
List the individual factual assertions this text makes about the world.

Each claim must stand on its own, read by someone who will never see this
document. A reader must be able to understand and check it without the
surrounding text.

Do NOT make claims about the document itself. Anything of the form "The text
provides...", "The text says...", "This section lists...", "The architecture
diagram shows..." is a statement about a document, not knowledge. State the
underlying fact instead, or omit it.

Do not restate the same fact several ways. If two assertions differ only in
phrasing, keep the single clearest one.

Rules:
- One assertion per claim. Split compound statements.
- `evidence_quote` MUST be copied verbatim from the text, character for
  character. Do not paraphrase, re-order, expand an abbreviation, or turn a
  table row into a sentence. If the supporting content is a table row, quote
  the row exactly as written, pipes included — or make no claim.
- If you cannot quote it exactly, do not make the claim. A dropped claim costs
  nothing; an unquotable one is discarded anyway.
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
