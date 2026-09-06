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
#: **1.0, meaning off.** It shipped at 1.25 on a sweep over 1,724 spans with
#: `nomic-embed-text` (2026-08-28) that showed R@10 0.489 -> 0.510. That sweep
#: is superseded: re-run on the post-split corpus, 670 sources and 8,133 spans,
#: **every boost is a regression** and 1.25 is merely the mildest, costing
#: 0.062 of R@10 and turning a hit into a miss.
#:
#: The damage is concentrated exactly where answering needs help. Per-category
#: R@10 under b=1.25: `exact_concept`, `technology`, `project` and
#: `related_concept` are unchanged, `dsa` loses 0.125 and `fuzzy_concept` loses
#: 0.200 of its 0.300. A boost can only fire when the query's words appear in a
#: filename or heading, which is precisely where BM25 already ranks the page
#: first; it adds nothing there and pushes down the differently-worded pages a
#: question actually needs.
#:
#: The retrieval doc previously argued this might not transfer, because the
#: eval scores document recall while answering does span selection. That was
#: wrong: `ask()` and `RetrievalEvaluator._lexical` make the identical
#: `SearchService.search(SearchQuery(...))` call, so the measurement is of this
#: exact operation.
#:
#: See docs/research/retrieval-baseline.md, "Title boosting measured".
TITLE_BOOST = 1.0

#: Retrieval mode for answering. Measured on the 24-query labelled set with
#: `nomic-embed-text` over 1,724 spans (2026-08-28):
#:
#:     method          R@5     R@10    P@5     MRR
#:     lexical         0.406   0.588   0.158   0.427
#:     semantic        0.628   0.733   0.242   0.653
#:     hybrid(w=0.75)  0.607   0.774   0.225   0.626
#:
#: Lexical is the worst option on every metric, so answering no longer defaults
#: to it. w=0.75 is chosen over semantic-alone because answering hands the model
#: eight passages at once: getting the right document *into* that set (R@10,
#: 0.774 vs 0.733) matters more than its rank within it, and the residual
#: lexical weight still catches exact-term queries an embedding blurs.
#:
#: The two are within noise of each other on 24 queries — this is the better
#: available choice, not a tuned optimum. With no embeddings stored, retrieval
#: degrades to lexical on its own.
ANSWER_SEMANTIC = True
ANSWER_SEMANTIC_WEIGHT = 0.75

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

    def ask(self, question: str, *, semantic: bool = ANSWER_SEMANTIC) -> Answer:
        hits = self.search.search(
            SearchQuery(
                text=question,
                limit=self.passages,
                semantic=semantic,
                exclude_sources=self.exclude_sources,
                title_boost=TITLE_BOOST,
                semantic_weight=ANSWER_SEMANTIC_WEIGHT,
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
