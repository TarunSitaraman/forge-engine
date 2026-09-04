"""The held-out assessment set must stay held out.

`assessment-v1.yaml` stopped being a clean instrument the moment
`assess-prompts/0.2.0` was written against its failures. This set exists to
answer what that one no longer can, and it only keeps that property while
nobody teaches the prompt its cases.

These are structural guards, not quality checks. They cannot stop someone
writing a cue for a far-transfer case, but they make it fail loudly.
"""

from __future__ import annotations

from collections import Counter

import pytest
import yaml

from forge.evaluation.assessment import DEFAULT_ASSESSMENT_SET

HOLDOUT = DEFAULT_ASSESSMENT_SET.parent / "assessment-holdout-v1.yaml"


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return yaml.safe_load(HOLDOUT.read_text(encoding="utf-8"))["cases"]


class TestItIsASeparateInstrument:
    def test_the_fitted_set_is_a_different_file(self):
        """Appending these to assessment-v1 would destroy both: the fitted
        score becomes uninterpretable and the held-out score becomes fitted."""
        assert HOLDOUT.exists()
        assert HOLDOUT != DEFAULT_ASSESSMENT_SET

    def test_no_case_id_is_shared_with_the_fitted_set(self, cases):
        fitted = yaml.safe_load(DEFAULT_ASSESSMENT_SET.read_text(encoding="utf-8"))["cases"]
        assert not {c["id"] for c in cases} & {c["id"] for c in fitted}

    def test_no_claim_is_reused_from_the_fitted_set(self, cases):
        """Surface reuse would let a memorised answer transfer."""
        fitted = yaml.safe_load(DEFAULT_ASSESSMENT_SET.read_text(encoding="utf-8"))["cases"]
        assert not {c["claim"] for c in cases} & {c["claim"] for c in fitted}


class TestTheStrataAreIntact:
    def test_every_case_declares_one(self, cases):
        assert all(c.get("stratum") for c in cases)

    def test_the_design_balance_holds(self, cases):
        got = Counter(c["stratum"] for c in cases)
        assert got == {
            "near-transfer": 5,
            "far-transfer": 5,
            "regression-probe": 4,
            "conflict": 2,
            "irrelevant": 2,
        }

    def test_near_and_far_transfer_are_all_insufficient_evidence(self, cases):
        """Both strata exist to probe one class. Mixing others in makes a
        per-stratum score uninterpretable."""
        for stratum in ("near-transfer", "far-transfer"):
            classes = {
                c["expected_classification"] for c in cases if c["stratum"] == stratum
            }
            assert classes == {"INSUFFICIENT_EVIDENCE"}, stratum

    def test_the_regression_probes_are_the_classes_that_regressed(self, cases):
        """0.2.0 escalated a REFINES case to POTENTIAL_CONFLICT. These probe
        for the cure being worse than the disease, so they must be cases a
        conflict-happy prompt would over-escalate — never conflicts."""
        probes = [c for c in cases if c["stratum"] == "regression-probe"]
        assert {c["expected_classification"] for c in probes} == {"REFINES", "SUPPORTS"}


class TestTheFalsePositiveRateStaysMeasurable:
    def test_non_conflict_cases_dominate(self, cases):
        """The gate metric is the false-positive conflict rate, so most cases
        must have a correct answer other than POTENTIAL_CONFLICT."""
        non_conflict = [
            c for c in cases if c["expected_classification"] != "POTENTIAL_CONFLICT"
        ]
        assert len(non_conflict) >= 0.75 * len(cases)

    def test_real_conflicts_are_present(self, cases):
        """Without them a prompt could score well by never flagging anything."""
        conflicts = [
            c for c in cases if c["expected_classification"] == "POTENTIAL_CONFLICT"
        ]
        assert len(conflicts) >= 2

    def test_cases_expecting_no_proposal_exist(self, cases):
        """Catches the opposite failure from the usual one: a pipeline that
        manufactures a proposal from every assessment."""
        assert any(c.get("expected_proposal") is None for c in cases)


class TestFarTransferIsNotTaughtByThePrompt:
    """The load-bearing guard. `far-transfer` measures whether the class is
    understood rather than whether five listed patterns are recognised. It
    only measures that while the prompt names none of them."""

    CUE_TERMS = (
        "mechanism",
        "population",
        "partly",
        "single observation",
        "anecdote",
        "intended, planned",
    )

    def test_no_far_transfer_case_is_described_by_a_current_cue(self, cases):
        from forge.evolution.prompts import ASSESSMENT_INSTRUCTION

        flat = " ".join(ASSESSMENT_INSTRUCTION.split()).lower()
        far = [c for c in cases if c["stratum"] == "far-transfer"]

        for case in far:
            note = case["note"].lower()
            leaked = [t for t in self.CUE_TERMS if t in flat and t in note]
            assert not leaked, (
                f"{case['id']} is now described by prompt cue(s) {leaked} — it has "
                "become a fitted case and must move to near-transfer, with a "
                "replacement written for far-transfer"
            )

    def test_the_far_categories_are_the_documented_five(self, cases):
        """Named so a reviewer can check the set still probes what it claims."""
        ids = {c["id"] for c in cases if c["stratum"] == "far-transfer"}
        assert ids == {
            "ht-far-correlation-for-causal-claim",
            "ht-far-aggregate-hides-subgroup",
            "ht-far-secondhand-attribution",
            "ht-far-term-defined-differently",
            "ht-far-direction-without-magnitude",
        }
