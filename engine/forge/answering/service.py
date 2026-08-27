"""Answer questions from the vault, with every statement cited.

This is the economic inversion the direction plan is built on. Answering a
question over retrieved spans costs one model call; pre-extracting the corpus so
a question *might* be answerable was measured at 3,372 calls and 153 hours.

Two rules make the answer trustworthy rather than merely fluent:

* **Retrieval failure is not a model problem.** With no hits, no call is made
  and the caller is told the vault has nothing — never "let the model try".
* **Citations are verified deterministically.** The model cites `[n]`; every
  `n` is checked against the passages actually supplied. A citation to `[7]`
  when six passages were given is a fabrication, and it is reported rather
  than rendered. This is the same check as extraction's quote grounding, and
  it costs nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import CompletionRequest, LLMProvider, Message
from ..logging import get_logger
from ..retrieval.search import SearchHit, SearchQuery, SearchService
from .prompts import ANSWER_PROMPT_VERSION, build_answer_messages

log = get_logger(__name__)

ANSWERER_VERSION = "answerer/0.1.0"

#: How many spans to put in front of the model. Enough for context, few enough
#: that a weak retrieval result does not bury the relevant passage.
DEFAULT_PASSAGES = 8

#: Excluded from answering by default. `docs/` is the engine's own manual, not
#: vault knowledge; without this it took five of the top eight spans for
#: "what is retrieval augmented generation?", answering from the documentation
#: of the tool rather than from the user's notes.
ENGINE_DOCS = ("docs/",)

#: Multiplier applied when a span's heading path or filename matches the query.
#:
#: Swept against the 24-query labelled set: 1.0 gives R@10 0.489 / MRR 0.482;
#: 1.25 gives R@10 0.510 / MRR 0.526, the only value that improves both. 1.5
#: keeps the recall gain but loses most of the MRR gain, and 2.0+ is worse than
#: no boost at all. **24 queries is a small set** — the +0.021 recall move is
#: about half a query, so treat 1.25 as the best available choice rather than a
#: tuned optimum.
TITLE_BOOST = 1.25

_CITATION_RE = re.compile(r"\[(\d+)\]")

#: The model's exact signal that the vault cannot answer the question.
NOT_IN_VAULT = "NOT IN VAULT"


@dataclass
class Answer:
    """A vault-grounded answer, with its evidence and its defects."""

    question: str
    text: str
    passages: list[SearchHit] = field(default_factory=list)
    #: Passage numbers the answer actually cited, in order of first use.
    cited: list[int] = field(default_factory=list)
    #: Citations naming a passage that was never supplied. Always a defect.
    invalid_citations: list[int] = field(default_factory=list)
    llm_calls: int = 0
    answered: bool = True

    @property
    def grounded(self) -> bool:
        """Every citation resolves, and at least one was made."""
        return not self.invalid_citations and bool(self.cited)

    def sources(self) -> list[str]:
        """Citation strings for the passages the answer used."""
        return [
            self.passages[n - 1].citation
            for n in self.cited
            if 1 <= n <= len(self.passages)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "answered": self.answered,
            "grounded": self.grounded,
            "cited": self.cited,
            "invalid_citations": self.invalid_citations,
            "sources": self.sources(),
            "passages_considered": len(self.passages),
            "llm_calls": self.llm_calls,
        }


class Answerer:
    """Retrieve, then answer strictly from what was retrieved."""

    version = ANSWERER_VERSION
    prompt_version = ANSWER_PROMPT_VERSION

    def __init__(
        self,
        search: SearchService,
        provider: LLMProvider | None,
        *,
        passages: int = DEFAULT_PASSAGES,
        exclude_sources: tuple[str, ...] = ENGINE_DOCS,
    ) -> None:
        self.search = search
        self.provider = provider
        self.passages = passages
        self.exclude_sources = exclude_sources

    def ask(self, question: str, *, semantic: bool = False) -> Answer:
        hits = self.search.search(
            SearchQuery(
                text=question,
                limit=self.passages,
                semantic=semantic,
                exclude_sources=self.exclude_sources,
                title_boost=TITLE_BOOST,
            )
        )
        if not hits:
            return Answer(
                question=question,
                text=(
                    f"{NOT_IN_VAULT}\nNothing in the vault matched this question, so no "
                    f"answer was attempted."
                ),
                answered=False,
            )

        if self.provider is None:
            return Answer(
                question=question,
                text="No model configured; retrieval succeeded but no answer was generated.",
                passages=hits,
                answered=False,
            )

        numbered = [f"[{i}] {h.citation}\n{h.span.text}" for i, h in enumerate(hits, 1)]
        messages = [
            Message(role=role, content=content)
            for role, content in build_answer_messages(question, numbered)
        ]
        response = self.provider.complete(
            CompletionRequest(messages=messages, temperature=0.0)
        )
        text = response.text.strip()

        used: list[int] = []
        invalid: list[int] = []
        for match in _CITATION_RE.finditer(text):
            n = int(match.group(1))
            if 1 <= n <= len(hits):
                if n not in used:
                    used.append(n)
            elif n not in invalid:
                invalid.append(n)

        answered = not text.upper().startswith(NOT_IN_VAULT)
        if invalid:
            log.warning(
                "answer_cited_missing_passages", invalid=invalid, supplied=len(hits)
            )

        return Answer(
            question=question,
            text=text,
            passages=hits,
            cited=used,
            invalid_citations=invalid,
            llm_calls=1,
            answered=answered,
        )
