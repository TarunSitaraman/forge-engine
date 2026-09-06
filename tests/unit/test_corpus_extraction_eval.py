"""Scoring extraction against the vault's own page names.

The metric choices are the substance here, so they are what gets pinned. Two
of them are easy to regress into something that reassures:

* **Junk beats recall.** An extractor emitting every name in the vocabulary
  scores 1.000 self-recovery. If that run does not also show its junk, the
  headline is worthless — this is the failure the labelled set was built for
  and it has to hold on this set too.
* **Off-vocabulary is not junk.** A concept the vault has no page for is
  extraction working. Folding it into junk would score a correct extractor as
  a broken one, and the two rates are therefore asserted to disagree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from forge.evaluation.corpus_extraction import (
    CorpusExtractionReport,
    VaultPage,
    Vocabulary,
    run,
    score_page,
)
from forge.extraction.extractor import _grounded

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "concept_extraction_eval.py"

FORBIDDEN = ["RAM", "Answer", "VARCHAR(n)"]


class _Concept:
    def __init__(self, name: str) -> None:
        self.canonical_name = name


def vocabulary() -> Vocabulary:
    return Vocabulary.from_concepts(
        _Concept(n) for n in ("B-tree Index", "Vector Databases", "RAG", "Heap")
    )


def page(**kw) -> VaultPage:
    base = {"path": "Technologies/Docs/rag.md", "canonical_name": "rag", "text": ""}
    return VaultPage(**{**base, **kw})


# -- targets --------------------------------------------------------------


def test_the_title_counts_as_the_pages_own_name():
    """`rag.md` titled "RAG (Retrieval-Augmented Generation)" is recovered by `RAG`.

    Scoring the filename stem alone would record a miss for an extractor that
    named the concept exactly as the page's own heading does, which is not a
    fact about extraction.
    """
    p = page(title="RAG (Retrieval-Augmented Generation)")
    assert set(p.targets) == {"rag", "RAG (Retrieval-Augmented Generation)"}
    assert score_page(p, ["RAG"], vocabulary(), FORBIDDEN).recovered


def test_a_title_that_only_restates_the_stem_is_not_a_second_target():
    assert page(title="RAG").targets == ("rag",)


def test_recovery_is_case_and_punctuation_insensitive():
    p = page(path="x.md", canonical_name="B-tree Index")
    assert score_page(p, ["b tree index"], vocabulary(), FORBIDDEN).recovered_as == "B-tree Index"


def test_a_page_whose_concept_never_came_back_is_a_miss():
    score = score_page(page(), ["Chunking"], vocabulary(), FORBIDDEN)
    assert not score.recovered
    assert score.recovered_as is None


# -- the three buckets ----------------------------------------------------


def test_junk_vocabulary_and_off_vocabulary_are_exclusive():
    score = score_page(
        page(), ["rag", "Vector Databases", "RAM", "Chunk Overlap"], vocabulary(), FORBIDDEN
    )
    assert score.junk == ["RAM"]
    assert score.in_vocabulary == ["Vector Databases", "rag"]
    assert score.off_vocabulary == ["Chunk Overlap"]
    buckets = score.junk + score.in_vocabulary + score.off_vocabulary
    assert sorted(buckets) == sorted(score.emitted), "every emitted name lands in exactly one"


def test_forbidden_wins_over_the_vocabulary():
    """A name on the junk list stays junk even if the vault has such a page.

    Order matters because the two lists can overlap: the vault could grow a
    page called `Answer` tomorrow, and that must not launder the string that
    was put on the forbidden list for being emitted as a concept.
    """
    vocab = Vocabulary.from_concepts([_Concept("RAM")])
    score = score_page(page(), ["RAM"], vocab, FORBIDDEN)
    assert score.junk == ["RAM"]
    assert score.in_vocabulary == []


def test_off_vocabulary_is_not_counted_as_junk():
    report = CorpusExtractionReport(model_id="m", prompt_version="p")
    report.scores.append(score_page(page(), ["rag", "Chunk Overlap"], vocabulary(), FORBIDDEN))
    assert report.off_vocabulary_rate == 0.5
    assert report.junk_rate == 0.0, "a concept the vault lacks a page for is not junk"


def test_a_greedy_extractor_scores_perfect_recovery_and_shows_its_junk():
    """The whole reason junk is the headline and recovery is not.

    An extractor emitting the entire vocabulary plus every forbidden string
    recovers every page's concept. Recovery alone would rank it best.
    """
    report = CorpusExtractionReport(model_id="greedy", prompt_version="p")
    vocab = vocabulary()
    for name in vocab.names.values():
        report.scores.append(
            score_page(
                page(path=f"{name}.md", canonical_name=name),
                list(vocab.names.values()) + FORBIDDEN,
                vocab,
                FORBIDDEN,
            )
        )
    assert report.self_recovery == 1.0
    assert report.junk_rate == pytest.approx(3 / 7)


# -- what is and is not scored --------------------------------------------


def test_a_page_whose_calls_did_not_all_return_is_not_scored():
    """The rule `extraction.py` learned from a run that reported junk=0.00.

    A timeout returns a truncated result rather than raising, and a page that
    emitted nothing emits no junk either.
    """
    report = CorpusExtractionReport(model_id="m", prompt_version="p")
    report.scores.append(score_page(page(), ["rag"], vocabulary(), FORBIDDEN))
    report.scores.append(
        score_page(page(path="b.md"), [], vocabulary(), FORBIDDEN, status="partial")
    )
    assert len(report.complete) == 1
    assert not report.trustworthy
    assert report.self_recovery == 1.0, "scored over the page that completed, not both"


def test_an_empty_report_reports_zero_rather_than_dividing_by_zero():
    report = CorpusExtractionReport(model_id="m", prompt_version="p")
    assert report.self_recovery == 0.0
    assert report.junk_rate == 0.0
    assert not report.trustworthy, "nothing measured is not a clean run"


def test_the_summary_line_says_when_pages_were_dropped():
    report = CorpusExtractionReport(model_id="m", prompt_version="p")
    report.scores.append(score_page(page(), [], vocabulary(), FORBIDDEN, status="failed"))
    assert "0/1 pages" in report.summary_line()


# -- grounding ------------------------------------------------------------


def test_dropped_claims_are_in_the_grounding_denominator():
    """Otherwise the rate is over survivors of the same check and is always 1.000."""
    p = page(text="A B-tree keeps its leaves at one depth.")
    score = score_page(
        p,
        ["B-tree Index"],
        vocabulary(),
        FORBIDDEN,
        claims=[("leaves are level", "A B-tree keeps its leaves at one depth.")],
        dropped_claims=1,
    )
    report = CorpusExtractionReport(model_id="m", prompt_version="p")
    report.scores.append(score)
    assert report.grounding_rate == 0.5


def test_a_quote_absent_from_the_page_is_not_grounded():
    p = page(text="A B-tree keeps its leaves at one depth.")
    score = score_page(
        p, [], vocabulary(), FORBIDDEN, claims=[("invented", "B-trees are always red-black.")]
    )
    assert score.grounded_claims == 0


# -- the driver -----------------------------------------------------------


class _Result:
    def __init__(self, concepts, status="succeeded"):
        self.concepts = [type("C", (), {"name": n})() for n in concepts]
        self.claims = []
        self.llm_calls = len(concepts)
        self.failures: list[dict] = []
        self.status = type("S", (), {"value": status})()


class _Extractor:
    prompt_version = "test/1"

    def __init__(self, results):
        self.results = list(results)

    def model_id(self):
        return "test-model"

    def extract(self, spans):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_a_provider_failure_becomes_a_score_not_a_crash():
    pages = [page(path="a.md", canonical_name="a"), page(path="b.md", canonical_name="b")]
    extractor = _Extractor([RuntimeError("connection reset"), _Result(["b"])])
    report = run(pages, extractor, vocabulary(), lambda p: [], forbidden=FORBIDDEN)

    assert len(report.scores) == 2, "the run continues past a failed page"
    assert not report.trustworthy
    assert "RuntimeError: connection reset" in (report.failed[0].error or "")


def test_progress_is_reported_for_every_page_including_failures():
    """The callback is what a `--json` run shows on a rate-limited host.

    A page that fails must still announce itself, or a run stuck in provider
    backoff looks identical to a run that finished.
    """
    seen: list[tuple[str, int]] = []
    pages = [page(path="a.md", canonical_name="a"), page(path="b.md", canonical_name="b")]
    extractor = _Extractor([RuntimeError("down"), _Result(["b"])])
    run(
        pages,
        extractor,
        vocabulary(),
        lambda p: [],
        forbidden=FORBIDDEN,
        on_page=lambda score, position: seen.append((score.path, position)),
    )
    assert seen == [("a.md", 1), ("b.md", 2)]


# -- the script itself ----------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("concept_extraction_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_junk_list_comes_from_the_labelled_set():
    """Shared on purpose: the two evals' junk rates then mean the same thing.

    Retyping the strings here would let the lists drift apart while both
    columns still read "junk", which is the worst version of this.
    """
    forbidden = _script().forbidden_strings()
    assert "VARCHAR(n)" in forbidden
    assert "maxmemory" in forbidden
    assert len(forbidden) == len(set(forbidden)), "deduplicated across cases"


def test_the_scripted_provider_pairs_a_real_quote_with_an_ungrounded_one():
    """Scripted grounding must be able to fail, or it measures nothing.

    A check that always passes in the only mode runnable offline is worse than
    no check. The scrambled quote is built from the span's own vocabulary, so
    only the order-preserving half of `_grounded` can reject it.
    """
    import json

    module = _script()
    span_text = "A B-tree keeps every one of its leaves at exactly one depth below the root."
    current = {"page": VaultPage(path="x.md", canonical_name="B-tree Index")}
    provider = module.scripted_provider(current)

    request = type(
        "R",
        (),
        {
            "messages": [
                type("M", (), {"content": "system"})(),
                type(
                    "M",
                    (),
                    {
                        "content": "List the individual factual assertions this text makes.\n\n"
                        f"--- TEXT START ---\n{span_text}\n--- TEXT END ---"
                    },
                )(),
            ]
        },
    )()
    claims = json.loads(provider.responder(request))["claims"]

    assert len(claims) == 2
    assert _grounded(claims[0]["evidence_quote"], span_text)
    assert not _grounded(claims[1]["evidence_quote"], span_text)
