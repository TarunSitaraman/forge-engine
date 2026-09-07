"""The `--sleep` wiring in `scripts/assessment_eval.py`.

`ThrottledProvider` is unit-tested next door. What this covers is the part
only the script knows: that the throttle is applied at all, that it is applied
*after* provider identity is read, and that its wait is subtracted from the
per-case latency. Get the last one wrong and a `--sleep 5` run reports every
case as five seconds slower than it was, quietly turning the pacing knob into
a latency regression in the record.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from itertools import pairwise
from pathlib import Path

import pytest
from forge.llm import MockProvider

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "assessment_eval.py"


def _module():
    spec = importlib.util.spec_from_file_location("assessment_eval_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env():
    """`main` writes FORGE_LLM_PROVIDER into the process environment."""
    keys = ("FORGE_LLM_PROVIDER", "FORGE_CLOUD_MODEL", "FORGE_MODEL_DEFAULT")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run(module, argv, monkeypatch, capsys) -> tuple[dict, list[float]]:
    """Drive `main` against a mock provider; return the report and call times.

    The provider timestamps each call, because the guarantee worth asserting is
    about the spacing of call *starts*. Asserting slept-seconds instead would
    fail for the right reason and read as a bug: at a small interval the work
    between two calls already covers part of the gap, so the throttle correctly
    sleeps less than the interval.
    """
    started_at: list[float] = []

    class NamedMock(MockProvider):
        def resolve_model(self, role: str) -> str:
            return "mock-model"

        def complete(self, request):
            started_at.append(time.monotonic())
            return super().complete(request)

    monkeypatch.setattr(module, "get_provider", lambda settings: NamedMock(default_response="{}"))
    monkeypatch.setattr(sys, "argv", ["assessment_eval.py", *argv])
    module.main()
    return json.loads(capsys.readouterr().out), started_at


INTERVAL = 0.05


def test_call_starts_are_spaced_by_at_least_the_interval(monkeypatch, capsys, clean_env):
    """The guarantee the flag exists for."""
    module = _module()
    payload, started_at = run(
        module,
        ["--provider", "cloud", "--sleep", str(INTERVAL), "--json"],
        monkeypatch,
        capsys,
    )

    assert payload["min_call_interval_seconds"] == INTERVAL
    assert len(started_at) == payload["cases"], "one call per case"
    gaps = [b - a for a, b in pairwise(started_at)]
    assert min(gaps) >= INTERVAL * 0.95, f"a gap of {min(gaps):.4f}s is under the interval"


def test_an_unpaced_run_has_gaps_shorter_than_the_interval(monkeypatch, capsys, clean_env):
    """Pins that the previous test is measuring the flag, not the machine.

    Without it, a run slow enough to space its own calls would satisfy the
    spacing assertion with the throttle doing nothing at all.
    """
    module = _module()
    _, started_at = run(module, ["--provider", "cloud", "--json"], monkeypatch, capsys)
    gaps = [b - a for a, b in pairwise(started_at)]
    assert min(gaps) < INTERVAL, (
        "unpaced calls are already further apart than the interval under test, "
        "so raise INTERVAL or this suite proves nothing about pacing"
    )


def test_the_throttle_never_sleeps_more_than_the_interval(monkeypatch, capsys, clean_env):
    module = _module()
    payload, started_at = run(
        module,
        ["--provider", "cloud", "--sleep", str(INTERVAL), "--json"],
        monkeypatch,
        capsys,
    )
    waited = payload["throttle_waited_seconds"]
    assert 0 < waited <= (len(started_at) - 1) * INTERVAL + 0.01


def test_the_wait_is_subtracted_from_the_reported_latency(monkeypatch, capsys, clean_env):
    """Otherwise the ms/case column measures the flag, not the model."""
    module = _module()
    payload, _ = run(
        module,
        ["--provider", "cloud", "--sleep", str(INTERVAL), "--json"],
        monkeypatch,
        capsys,
    )

    latencies = [r["latency_ms"] for r in payload["results"] if r["latency_ms"]]
    assert latencies, "the run recorded no latencies at all"
    assert max(latencies) < INTERVAL * 1000, (
        "a case reported at least a full throttle interval of latency, so the "
        "wait is being counted as model time"
    )


def test_identity_is_read_before_the_provider_is_wrapped(monkeypatch, capsys, clean_env):
    """A wrapper that hid `resolve_model` would record every model as unknown."""
    module = _module()
    payload, _ = run(
        module,
        ["--provider", "cloud", "--sleep", str(INTERVAL), "--json"],
        monkeypatch,
        capsys,
    )
    assert payload["model_id"] == "mock-model"


def test_an_unthrottled_run_records_a_zero_interval(monkeypatch, capsys, clean_env):
    module = _module()
    payload, _ = run(module, ["--provider", "cloud", "--json"], monkeypatch, capsys)
    assert payload["min_call_interval_seconds"] == 0.0
    assert payload["throttle_waited_seconds"] == 0.0


def test_sleep_is_refused_with_the_scripted_provider():
    """It makes no network calls, so pacing it only makes the run slower."""
    module = _module()
    with pytest.raises(SystemExit):
        module.main()


def test_a_negative_interval_is_refused(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        sys, "argv", ["assessment_eval.py", "--provider", "cloud", "--sleep", "-1"]
    )
    with pytest.raises(SystemExit):
        module.main()
