"""The report diff, and the formatting defect that made it lie.

`scripts/assessment_compare.py` is a dev tool, not shipped code, but it is the
thing a model-swap decision gets read off. Its first version printed a validity
rate that rose 0.94 -> 1.00 as "+0", because it chose the format by whether the
new value looked integral. A comparison tool that renders an improvement as no
change is worse than no tool, so the delta formatting is pinned here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "assessment_compare.py"


def _module():
    spec = importlib.util.spec_from_file_location("assessment_compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = _module()


def _case(case_id: str, expected: str, actual: str) -> dict:
    return {
        "case": case_id,
        "expected": expected,
        "actual": actual,
        "structured_output_valid": True,
        "grounded": True,
        "classification_correct": expected == actual,
        "proposal_correct": expected == actual,
        "cached_on_repeat": True,
        "latency_ms": 1000.0,
        "detail": "",
    }


def _report(tmp_path: Path, name: str, cases: list[dict], **overrides) -> Path:
    correct = sum(c["classification_correct"] for c in cases)
    payload = {
        "provider_id": "cloud",
        "model_id": name,
        "scripted": False,
        "cases": len(cases),
        "structured_output_validity": 1.0,
        "grounding_rate": 1.0,
        "classification_accuracy": round(correct / len(cases), 4),
        "proposal_correctness": round(correct / len(cases), 4),
        "cache_effectiveness": 1.0,
        "mean_latency_ms": 1000.0,
        "results": cases,
    }
    payload.update(overrides)
    path = tmp_path / f"{name.replace('/', '-')}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestDelta:
    def test_an_improvement_to_a_round_number_is_not_printed_as_zero(self):
        """The defect: 1.00 is integral, so it formatted with 0 decimals."""
        assert compare.delta(1.0, 0.94) == "+0.06"

    def test_no_change_prints_nothing(self):
        """A column of +0.00 buries the row that actually moved."""
        assert compare.delta(0.72, 0.72) == ""

    def test_a_regression_keeps_its_sign(self):
        assert compare.delta(0.60, 0.72) == "-0.12"


class TestGuards:
    def test_a_scripted_run_is_refused(self, tmp_path):
        path = _report(tmp_path, "mock", [_case("a", "SUPPORTS", "SUPPORTS")], scripted=True)
        with pytest.raises(SystemExit, match="by construction"):
            compare.load(path)

    def test_a_report_without_per_case_results_is_refused(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"provider_id": "cloud", "model_id": "m"}), encoding="utf-8")
        with pytest.raises(SystemExit, match="--json"):
            compare.load(path)

    def test_two_different_datasets_are_refused_rather_than_overlapped(self, tmp_path, monkeypatch, capsys):
        """Diffing the fitted set against the held-out one would produce a
        confident, meaningless table."""
        base = _report(tmp_path, "a", [_case("x", "SUPPORTS", "SUPPORTS"), _case("y", "REFINES", "REFINES")])
        cand = _report(tmp_path, "b", [_case("x", "SUPPORTS", "SUPPORTS")])
        monkeypatch.setattr("sys.argv", ["assessment_compare.py", str(base), str(cand)])
        with pytest.raises(SystemExit, match="not the same dataset"):
            compare.main()


class TestMovedCases:
    def test_a_net_zero_swap_still_names_the_cases_that_moved(self, tmp_path, monkeypatch, capsys):
        """The measured trap: 13/18 both ways, one case fixed and one broken.
        A headline delta shows nothing; this must show both."""
        base = _report(
            tmp_path,
            "baseline",
            [_case("fixed-one", "INSUFFICIENT_EVIDENCE", "SUPPORTS"), _case("broken-one", "REFINES", "REFINES")],
        )
        cand = _report(
            tmp_path,
            "candidate",
            [
                _case("fixed-one", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
                _case("broken-one", "REFINES", "INSUFFICIENT_EVIDENCE"),
            ],
        )
        monkeypatch.setattr("sys.argv", ["assessment_compare.py", str(base), str(cand)])
        assert compare.main() == 0

        out = capsys.readouterr().out
        assert "+0 cases" in out
        assert "fixed (1)" in out and "fixed-one" in out
        assert "broken (1)" in out and "broken-one" in out

    def test_a_swap_that_changes_nothing_says_so(self, tmp_path, monkeypatch, capsys):
        cases = [_case("a", "SUPPORTS", "SUPPORTS"), _case("b", "REFINES", "SUPPORTS")]
        base = _report(tmp_path, "baseline", cases)
        cand = _report(tmp_path, "candidate", cases)
        monkeypatch.setattr("sys.argv", ["assessment_compare.py", str(base), str(cand)])
        compare.main()
        assert "No case changed classification" in capsys.readouterr().out


class TestStability:
    """`assessment_eval.stability` — the answer to "is this delta real?".

    Added after the fitted set scored 18/21 and 15/21 on consecutive days
    against the same model, prompt and command, with the three differing cases
    being exactly the three a prompt revision had been credited with fixing.
    """

    @staticmethod
    def _eval_module():
        spec = importlib.util.spec_from_file_location(
            "assessment_eval", SCRIPT.parent / "assessment_eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _report(self, module, answers: dict[str, tuple[str, str]]):
        report = module.AssessmentReport(provider_id="cloud", model_id="m", scripted=False)
        for case_id, (expected, actual) in answers.items():
            result = module.CaseResult(case_id=case_id, expected=expected)
            result.actual = actual
            result.classification_correct = expected == actual
            report.results.append(result)
        return report

    def test_a_case_answered_differently_across_runs_is_counted_unstable(self):
        module = self._eval_module()
        runs = [
            self._report(module, {"flaky": ("INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE")}),
            self._report(module, {"flaky": ("INSUFFICIENT_EVIDENCE", "SUPPORTS")}),
        ]
        marks = module.stability(runs)
        assert marks["unstable"] == 1
        assert marks["cases"]["flaky"]["correct"] == 1
        assert marks["cases"]["flaky"]["answers"] == {"INSUFFICIENT_EVIDENCE": 1, "SUPPORTS": 1}

    def test_a_case_wrong_the_same_way_every_time_is_stable_not_unstable(self):
        """A consistent failure is a finding about the model. A flipping one is
        not, and conflating them is how a lucky run becomes a fixed case."""
        module = self._eval_module()
        runs = [self._report(module, {"hard": ("INSUFFICIENT_EVIDENCE", "SUPPORTS")})] * 3
        marks = module.stability(runs)
        assert marks["unstable"] == 0
        assert marks["cases"]["hard"]["correct"] == 0

    def test_the_score_spread_is_reported(self):
        module = self._eval_module()
        runs = [
            self._report(module, {"a": ("SUPPORTS", "SUPPORTS"), "b": ("REFINES", "REFINES")}),
            self._report(module, {"a": ("SUPPORTS", "SUPPORTS"), "b": ("REFINES", "SUPPORTS")}),
        ]
        marks = module.stability(runs)
        assert marks["scores"] == [2, 1]
        assert (marks["min_correct"], marks["max_correct"]) == (1, 2)


class TestUnavailableIsNotAMiss:
    """An outage is not a wrong answer.

    2026-09-05: a three-run variance measurement lost DNS at case 13 of run 3.
    The remaining nine cases were recorded as ordinary misses, which would have
    entered the stability table as nine cases that "changed their answer" and
    dropped the run's score by nine. The failure mode this whole document warns
    about, arriving through the tool built to detect it.
    """

    @staticmethod
    def _eval_module():
        spec = importlib.util.spec_from_file_location(
            "assessment_eval", SCRIPT.parent / "assessment_eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _run(self, module, answered: int, lost: int):
        report = module.AssessmentReport(provider_id="cloud", model_id="m", scripted=False)
        for i in range(answered):
            result = module.CaseResult(case_id=f"case-{i}", expected="SUPPORTS")
            result.actual = "SUPPORTS"
            result.classification_correct = True
            result.structured_output_valid = True
            result.grounded = True
            report.results.append(result)
        for i in range(answered, answered + lost):
            result = module.CaseResult(case_id=f"case-{i}", expected="SUPPORTS")
            result.unavailable = True
            result.detail = "semantic_analysis_unavailable: network unreachable"
            report.results.append(result)
        return report

    def test_accuracy_is_over_the_cases_that_reached_the_model(self):
        """12 of 12 answered correctly is 1.00, not 12/21 = 0.57."""
        module = self._eval_module()
        report = self._run(module, answered=12, lost=9)
        assert report.classification_accuracy == 1.0
        assert report.unavailable == 9
        assert len(report.measured) == 12

    def test_the_headline_says_the_run_is_incomplete(self):
        module = self._eval_module()
        report = self._run(module, answered=12, lost=9)
        assert "9/21 UNMEASURED" in report.headline()

    def test_an_unmeasured_case_is_not_counted_as_a_changed_answer(self):
        module = self._eval_module()
        complete = self._run(module, answered=21, lost=0)
        truncated = self._run(module, answered=12, lost=9)
        marks = module.stability([complete, truncated])
        assert marks["unstable"] == 0
        assert marks["cases"]["case-20"]["measured"] == 1
        assert marks["incomplete_runs"] == [2]

    def test_the_comparison_refuses_an_incomplete_report(self, tmp_path):
        module = self._eval_module()
        payload = self._run(module, answered=12, lost=9).to_dict()
        path = tmp_path / "truncated.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="never reached the model"):
            compare.load(path)


class TestRetryExhaustionIsAlsoANonAnswer:
    """The fifth non-answer scored as an answer, inside the fix for the fourth.

    2026-09-06: a `--repeat 3` run against a rate-limited host exhausted its
    retries on six cases. Those came back as AssessmentOutcome.RETRYABLE_FAILURE,
    which the previous fix did not list, so they were recorded as wrong answers
    with actual=None. The summary then reported "6 case(s) answered
    inconsistently" for a model that had given the same answer on every case it
    was actually asked.
    """

    @staticmethod
    def _eval_module():
        spec = importlib.util.spec_from_file_location(
            "assessment_eval", SCRIPT.parent / "assessment_eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_both_infrastructure_outcomes_count_as_no_answer(self):
        module = self._eval_module()
        outcome = module.AssessmentOutcome
        assert outcome.SEMANTIC_ANALYSIS_UNAVAILABLE in module.NO_ANSWER_OUTCOMES
        assert outcome.RETRYABLE_FAILURE in module.NO_ANSWER_OUTCOMES

    def test_a_rejected_assessment_is_still_the_models_fault(self):
        """The model answered and the answer was invalid. That is a result."""
        module = self._eval_module()
        assert module.AssessmentOutcome.ASSESSMENT_REJECTED not in module.NO_ANSWER_OUTCOMES

    def test_a_case_answered_once_and_dropped_once_is_not_inconsistent(self):
        """The exact shape of the bad report: one real answer, one dropped
        call, counted as two different answers."""
        module = self._eval_module()

        def run(actual, unavailable):
            report = module.AssessmentReport(provider_id="cloud", model_id="m", scripted=False)
            result = module.CaseResult(case_id="c", expected="REFINES")
            if unavailable:
                result.unavailable = True
            else:
                result.actual = actual
                result.classification_correct = actual == "REFINES"
            report.results.append(result)
            return report

        marks = module.stability([run("REFINES", False), run(None, True)])

        assert marks["unstable"] == 0, "a dropped call is not a second opinion"
        assert marks["cases"]["c"] == {"correct": 1, "measured": 1, "answers": {"REFINES": 1}}
