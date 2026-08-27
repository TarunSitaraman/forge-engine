"""Vault-grounded question answering. One model call per question."""

from .prompts import ANSWER_PROMPT_VERSION, build_answer_messages
from .service import ANSWERER_VERSION, DEFAULT_PASSAGES, NOT_IN_VAULT, Answer, Answerer

__all__ = [
    "Answer",
    "Answerer",
    "ANSWERER_VERSION",
    "ANSWER_PROMPT_VERSION",
    "DEFAULT_PASSAGES",
    "NOT_IN_VAULT",
    "build_answer_messages",
]
