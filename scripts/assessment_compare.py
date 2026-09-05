#!/usr/bin/env python3
"""Diff two `assessment_eval.py --json` reports, case by case.

    python3 scripts/assessment_eval.py --provider cloud --model A --json > a.json
    python3 scripts/assessment_eval.py --provider cloud --model B --json > b.json
    python3 scripts/assessment_compare.py a.json b.json

Written for one question: when a model is swapped, *which cases moved* — not
whether the headline went up. Two runs can score identically and disagree on a
third of the set (measured: the corroboration check, 13/18 both ways, fixing
one case and breaking another). A headline delta cannot show that and a pair of
eyeballed dumps is where the reading goes wrong.

The per-class table is the part to read. Overall accuracy on these sets is
dominated by classes that were never in question; the open one is
INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in payload:
        raise SystemExit(f"{path}: no per-case results — re-run with --json")
    if payload.get("scripted"):
        raise SystemExit(
            f"{path}: scripted run. Classification is 1.0 by construction there, "
            "so comparing it to anything is meaningless."
        )
    return payload


def by_case(payload: dict) -> dict[str, dict]:
    return {r["case"]: r for r in payload["results"]}


def label(payload: dict) -> str:
    return f"{payload['provider_id']}/{payload['model_id']}"


def score(results: list[dict]) -> tuple[int, int]:
    return sum(1 for r in results if r["classification_correct"]), len(results)


def delta(new: float, old: float) -> str:
    """Signed delta, or blank when nothing moved — a column of +0.00 hides the row that did.

    Always two decimals. Formatting by whether the value looks integral printed
    a rate that rose 0.94 -> 1.00 as "+0", which reads as no change.
    """
    if abs(new - old) < 1e-9:
        return ""
    return f"{new - old:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    base_cases, cand_cases = by_case(base), by_case(cand)

    only_base = sorted(base_cases.keys() - cand_cases.keys())
    only_cand = sorted(cand_cases.keys() - base_cases.keys())
    if only_base or only_cand:
        # Comparing a fitted run against a held-out one would produce a
        # confident, meaningless table. Refuse rather than diff the overlap.
        raise SystemExit(
            "these runs are not the same dataset:\n"
            f"  only in {args.baseline.name}: {only_base or 'none'}\n"
            f"  only in {args.candidate.name}: {only_cand or 'none'}"
        )

    base_correct, total = score(base["results"])
    cand_correct, _ = score(cand["results"])

    print(f"baseline : {label(base):<40} class={base['classification_accuracy']:.2f} ({base_correct}/{total})")
    print(f"candidate: {label(cand):<40} class={cand['classification_accuracy']:.2f} ({cand_correct}/{total})")
    print(f"           {'':<40} {cand_correct - base_correct:+d} cases")
    print()

    for name, key in (
        ("valid   ", "structured_output_validity"),
        ("grounded", "grounding_rate"),
        ("proposal", "proposal_correctness"),
        ("cache   ", "cache_effectiveness"),
    ):
        d = delta(cand[key], base[key])
        print(f"{name}  {base[key]:.2f} -> {cand[key]:.2f}  {d}".rstrip())
    print(
        f"latency   {base['mean_latency_ms']:.0f} -> {cand['mean_latency_ms']:.0f} ms/case"
    )
    print()

    classes = sorted({r["expected"] for r in base["results"]})
    print(f"{'expected class':<24} {'baseline':>9} {'candidate':>10}")
    for cls in classes:
        b = score([r for r in base["results"] if r["expected"] == cls])
        c = score([r for r in cand["results"] if r["expected"] == cls])
        moved = "" if b[0] == c[0] else f"  {c[0] - b[0]:+d}"
        print(f"{cls:<24} {f'{b[0]}/{b[1]}':>9} {f'{c[0]}/{c[1]}':>10}{moved}")
    print()

    fixed, broken, still = [], [], []
    for case_id in sorted(base_cases):
        b, c = base_cases[case_id], cand_cases[case_id]
        if not b["classification_correct"] and c["classification_correct"]:
            fixed.append((case_id, b))
        elif b["classification_correct"] and not c["classification_correct"]:
            broken.append((case_id, c))
        elif not b["classification_correct"]:
            still.append((case_id, b, c))

    print(f"fixed ({len(fixed)})")
    for case_id, b in fixed:
        print(f"  {case_id:<44} was {b['actual']}")
    print(f"broken ({len(broken)})")
    for case_id, c in broken:
        print(f"  {case_id:<44} now {c['actual']}, expected {c['expected']}")
    print(f"still wrong ({len(still)})")
    for case_id, b, c in still:
        same = "" if b["actual"] == c["actual"] else f" (was {b['actual']})"
        print(f"  {case_id:<44} {c['actual']}, expected {c['expected']}{same}")

    if not fixed and not broken:
        print("\nNo case changed classification. A model swap that moves nothing is a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
