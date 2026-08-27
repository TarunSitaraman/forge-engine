"""Versioned prompts for vault-grounded answering.

Versioned for the same reason extraction prompts are: an answer produced under
different instructions is a different artifact, and a changed prompt must be
visible rather than silent.
"""

from __future__ import annotations

ANSWER_PROMPT_VERSION = "answer-prompts/0.1.0"

SYSTEM = (
    "You answer questions using ONLY the numbered passages provided from the "
    "user's own knowledge vault. You never use outside knowledge. You cite "
    "every statement."
)

INSTRUCTION = """\
Answer the question using only the numbered passages below.

Rules:
- Cite the passage supporting each statement as [1], [2], and so on. Every
  sentence that states a fact needs a citation.
- Cite only passage numbers that appear below. Never invent a number.
- If the passages do not answer the question, say exactly:
  NOT IN VAULT
  followed by one sentence on what is missing. Do not answer from your own
  knowledge, and do not guess — an unanswerable question is a useful result.
- Do not repeat the question. Do not describe the passages ("the text says").
  State the answer.
- Prefer the vault's own terminology over synonyms.
"""


def build_answer_messages(question: str, passages: list[str]) -> list[tuple[str, str]]:
    """(role, content) pairs. Passages arrive pre-numbered from the caller."""
    body = "\n\n".join(passages)
    return [
        ("system", SYSTEM),
        (
            "user",
            f"{INSTRUCTION}\n\n--- PASSAGES ---\n{body}\n--- END PASSAGES ---\n\n"
            f"QUESTION: {question}",
        ),
    ]
