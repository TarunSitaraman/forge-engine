"""Scoring extraction against the vault's own page names.

The metric choices are the substance here, so they are what gets pinned. Two
of them are easy to regress into something that reassures:

* **Junk beats recall.** An extractor emitting every name in the vocabulary
  scores 1.000 self-recovery. If that run does not also show its junk, the
  headline is worthless; this is the failure the labelled set was built for
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
from forge.extraction.extractor import _grounded, _tokens

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


def test_a_degenerate_line_yields_no_probe_rather_than_a_fake_one():
    """The harness bug that read as a defect in `_grounded`.

    Reversing the *words* of `result = groupAnagrams(["eat","tea","tan"])`
    barely moves the *tokens*, because one word holds most of them; reversing
    a near-palindrome like `self.parent[x] = self.parent[self.parent[x]]`
    moves them not at all. Neither is a reordering, so a grounding check is
    right to accept them, and counting their acceptance against it reported
    the check failing on 11 of 545 pages when nothing was wrong.

    Lines like these must be skipped, not turned into probes.
    """
    probe = _script().adversarial_probe(
        "self.parent[x] = self.parent[self.parent[x]]\n"
        "self.parent[x] = self.parent[self.parent[x]]"
    )
    assert probe is None


def test_the_probe_reverses_tokens_not_words():
    """Word reversal is not enough when one word holds six tokens.

    `result = groupAnagrams(["eat","tea","tan","ate","nat","bat"])` is four
    whitespace-delimited words. Reversed as words, seven of its eight tokens
    stay in their original order and the "probe" is very nearly the original.
    """
    line = 'result = groupAnagrams(["eat","tea","tan","ate","nat","bat"])'
    probe = _script().adversarial_probe(line)
    assert probe is not None
    quote, scrambled = probe
    assert quote == line
    assert _tokens(scrambled) == list(reversed(_tokens(line)))

    by_words = " ".join(reversed(line.split()))
    assert _tokens(by_words) != list(reversed(_tokens(line))), (
        "if these ever agree, the word-level shortcut is safe again and this "
        "test is the place to say so"
    )


def test_probe_selection_never_consults_the_grounding_check():
    """Otherwise the probe is rejected by construction and proves nothing.

    Selecting scrambles until one `_grounded` rejects would make every run
    pass whatever `_grounded` does. Both selection criteria are properties of
    the input, so the module needs no reference to the function under test.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adversarial_probe"
    )
    # Names the body actually references, so the docstring may discuss the
    # function under test without tripping this.
    referenced = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "_grounded" not in referenced


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


class TestItRefusesTheWrongCorpus:
    """The eval must not score the engine's own documentation and call it a vault.

    Vault resolution looks upward from the working directory, so running the
    script from inside a forge-engine checkout resolves the repository itself.
    `forge bootstrap` then derives a couple of dozen concepts from `docs/`, and
    the run reports a self-recovery rate over architecture notes in exactly the
    format it uses for the 545-page vault. Nothing in the output said which
    corpus it measured, and the number is the kind somebody quotes weeks later.

    The same defect existed in `forge index` until 2026-09-01, when vault
    resolution stopped preferring the installed module's location.
    """

    def _run(self, cwd, args):
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parents[2] / "scripts" / "concept_extraction_eval.py"
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )

    def test_an_engine_checkout_is_refused_by_name(self, tmp_path):
        checkout = tmp_path / "forge-engine"
        (checkout / "engine" / "forge").mkdir(parents=True)
        (checkout / "engine" / "forge" / "__init__.py").write_text("", encoding="utf-8")
        (checkout / ".git").mkdir()
        (checkout / "docs").mkdir()
        (checkout / "docs" / "architecture.md").write_text("# Architecture\n\nProse.\n", "utf-8")

        result = self._run(tmp_path, ["--vault", str(checkout), "--limit", "2"])

        assert result.returncode == 2
        assert "engine's own checkout" in result.stderr
        assert "self-recovery" not in result.stdout, "it must not report a score"

    def test_a_real_vault_is_not_refused(self, tmp_path):
        vault = tmp_path / "notes"
        (vault / ".git").mkdir(parents=True)
        for name in ("alpha", "beta"):
            (vault / f"{name}.md").write_text(
                f"# {name.title()}\n\n"
                f"{name.title()} is a way of doing things that people use often "
                f"and it needs enough words to clear the extractor's own floor "
                f"for a span to be worth sending anywhere at all.\n",
                encoding="utf-8",
            )

        result = self._run(tmp_path, ["--vault", str(vault), "--limit", "2"])

        assert result.returncode == 0, result.stderr[-500:]
        assert "engine's own checkout" not in result.stderr

    def test_the_corpus_is_named_before_the_run_not_after(self, tmp_path):
        """An 80-minute paced run must not hide which vault it is scoring."""
        vault = tmp_path / "notes"
        (vault / ".git").mkdir(parents=True)
        (vault / "alpha.md").write_text(
            "# Alpha\n\nAlpha is a way of doing things that people use often and "
            "it needs enough words to clear the extractor's own floor.\n",
            encoding="utf-8",
        )

        result = self._run(tmp_path, ["--vault", str(vault), "--limit", "1"])

        assert "vault      :" in result.stderr
        assert "vocabulary :" in result.stderr
        assert "budget     :" in result.stderr


class TestItRefusesPacingItsBudgetCannotAfford:
    """A run paced faster than the provider's token budget fails after page one.

    Groq counts *reserved* output against a tokens-per-minute allowance, so a
    4096 ceiling against 8,000 TPM buys about 1.5 calls a minute once the prompt
    is charged. A 40-page run at `--sleep 20` on 2026-09-09 completed its first
    page on whatever was left in the bucket and then 429'd through every retry
    on every page after it, for as long as it was left running.

    The budget was documented in `CLOUD_PRESETS` since August. What was missing
    was anything that did the division, so the floor is computed here now.
    """

    def _floor(self, max_tokens, tpm=8000):
        import os
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import concept_extraction_eval as eval_script

        settings = type("S", (), {"llm": type("L", (), {"cloud": type("C", (), {
            "max_tokens": max_tokens})()})()})()
        old = os.environ.get("FORGE_CLOUD_PRESET")
        os.environ["FORGE_CLOUD_PRESET"] = "groq"
        try:
            return eval_script._pacing_floor(settings)
        finally:
            if old is None:
                os.environ.pop("FORGE_CLOUD_PRESET", None)
            else:
                os.environ["FORGE_CLOUD_PRESET"] = old

    def test_the_floor_is_above_the_pacing_that_failed(self):
        assert self._floor(4096) > 20, "the run that 429'd must not be allowed"
        assert 35 < self._floor(4096) < 45

    def test_a_smaller_reservation_buys_calls_back(self):
        """Halving the ceiling roughly halves the wait, which is the lever."""
        assert self._floor(1024) < self._floor(4096) / 2 + 1

    def test_no_floor_is_claimed_for_a_provider_nobody_measured(self):
        import os
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import concept_extraction_eval as eval_script

        settings = type("S", (), {"llm": type("L", (), {"cloud": type("C", (), {
            "max_tokens": 4096})()})()})()
        old = os.environ.get("FORGE_CLOUD_PRESET")
        os.environ["FORGE_CLOUD_PRESET"] = "openrouter"
        try:
            assert eval_script._pacing_floor(settings) is None
        finally:
            if old is None:
                os.environ.pop("FORGE_CLOUD_PRESET", None)
            else:
                os.environ["FORGE_CLOUD_PRESET"] = old

    def test_groq_records_the_budget_the_floor_is_computed_from(self):
        from forge.config import CLOUD_PRESETS

        assert CLOUD_PRESETS["groq"]["tokens_per_minute"] == 8000
        assert CLOUD_PRESETS["groq"]["max_tokens"] == 4096


class TestARunCanBeResumed:
    """Nine hours of paced calls left nothing on disk, so completed work persists.

    A hosted free tier limits a longer window than a minute, on evidence from
    2026-09-09: a correctly paced run still 429'd after about ten calls and
    stayed refused for hours. A sample big enough to be worth quoting may
    therefore not fit in one sitting, and pacing cannot fix that. Resuming can.
    """

    def _run(self, cwd, args):
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parents[2] / "scripts" / "concept_extraction_eval.py"
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )

    def _vault(self, tmp_path, pages=3):
        vault = tmp_path / "notes"
        (vault / ".git").mkdir(parents=True)
        for n in range(pages):
            (vault / f"topic{n}.md").write_text(
                f"# Topic{n}\n\nTopic{n} is a way of doing things that people "
                f"use often and it needs enough words to clear the extractor's "
                f"own floor for a span to be worth sending anywhere at all.\n",
                encoding="utf-8",
            )
        return vault

    def test_completed_pages_are_not_run_again(self, tmp_path):
        import json

        vault = self._vault(tmp_path)
        cache = tmp_path / "cache.json"

        first = self._run(tmp_path, ["--vault", str(vault), "--limit", "2", "--cache", str(cache)])
        assert first.returncode == 0, first.stderr[-400:]
        assert len(json.loads(cache.read_text())) == 2

        second = self._run(tmp_path, ["--vault", str(vault), "--limit", "2", "--cache", str(cache)])
        assert "2 page(s) already scored" in second.stderr
        assert "nothing left to run" in second.stderr

    def test_the_report_still_covers_every_page_across_sittings(self, tmp_path):
        vault = self._vault(tmp_path, pages=4)
        cache = tmp_path / "cache.json"

        self._run(tmp_path, ["--vault", str(vault), "--limit", "2", "--cache", str(cache)])
        widened = self._run(
            tmp_path, ["--vault", str(vault), "--limit", "4", "--cache", str(cache), "--json"]
        )

        import json as jsonlib

        payload = jsonlib.loads(widened.stdout)
        assert payload["sampled"] == 4, "cached pages must count toward the report"
        assert "2 to run" in widened.stderr, "only the new pages cost calls"

    def test_the_budget_describes_what_is_left_not_the_whole_sample(self, tmp_path):
        """Otherwise a resumed run announces a cost it is not going to pay."""
        vault = self._vault(tmp_path, pages=4)
        cache = tmp_path / "cache.json"

        self._run(tmp_path, ["--vault", str(vault), "--limit", "2", "--cache", str(cache)])
        widened = self._run(tmp_path, ["--vault", str(vault), "--limit", "4", "--cache", str(cache)])

        budget = [ln for ln in widened.stderr.splitlines() if ln.startswith("budget")][0]
        assert "12 model call" in budget, budget
