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
