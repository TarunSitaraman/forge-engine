#!/usr/bin/env python3
"""Re-check stored proposals' evidence quotes against their spans.

    python3 scripts/audit_grounding.py                 # audit the whole store
    python3 scripts/audit_grounding.py --json          # machine-readable
    python3 scripts/audit_grounding.py --show-passing  # print every row

**Zero LLM calls.** Grounding is a deterministic string check, so re-running it
over already-extracted proposals costs nothing and needs no model. That is the
whole point of this script.

WHY IT EXISTS

`_grounded` compared bag-of-words overlap until 2026-08-19, which accepted any
quote reassembled from the span's own vocabulary — including one that inverted
the span's meaning. Proposals extracted before the fix were admitted under that
looser rule and are still sitting in the store.

Re-extracting them would cost a full run, because the extractor version is part
of the derivation key and bumping it invalidates every cached result. Auditing
them costs seconds. So the fix ships without a version bump, and this script is
how the existing corpus gets held to the new rule: anything it reports was
stored on evidence that would not be accepted today, and should be rejected
rather than approved.

Exit codes: 0 clean, 1 ungrounded quotes found, 2 could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.config import ConfigError, Settings  # noqa: E402
from forge.extraction.extractor import _grounded, _ordered_overlap, _tokens  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402


def audit(store: SqliteStore) -> list[dict]:
    """Every proposal carrying an evidence quote, re-checked against its span."""
    rows: list[dict] = []
    for proposal in store.list_proposals(limit=100_000):
        quote = (proposal.operation.details or {}).get("evidence_quote")
        if not quote:
            continue
        for span_id in proposal.evidence_span_ids:
            span = store.get_span(span_id)
            if span is None:
                rows.append(
                    {
                        "proposal_id": proposal.id,
                        "status": proposal.status.value,
                        "span_id": span_id,
                        "quote": quote,
                        "grounded": False,
                        "overlap": 0.0,
                        "note": "span missing from store",
                    }
                )
                continue
            rows.append(
                {
                    "proposal_id": proposal.id,
                    "status": proposal.status.value,
                    "span_id": span_id,
                    "quote": quote,
                    "grounded": _grounded(quote, span.text),
                    "overlap": round(_ordered_overlap(_tokens(quote), _tokens(span.text)), 3),
                    "note": "",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--show-passing", action="store_true")
    args = parser.parse_args()

    try:
        settings = Settings.load(vault_path=args.vault)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    store = SqliteStore(settings.db_path)
    store.initialize()
    try:
        rows = audit(store)
    finally:
        store.close()

    failed = [r for r in rows if not r["grounded"]]

    if args.as_json:
        print(json.dumps({"checked": len(rows), "ungrounded": failed}, indent=2, sort_keys=True))
        return 1 if failed else 0

    if not rows:
        print("no proposals carry an evidence quote — nothing to audit")
        return 0

    for row in rows:
        if row["grounded"] and not args.show_passing:
            continue
        mark = "ok  " if row["grounded"] else "FAIL"
        print(f"{mark} {row['proposal_id'][:12]}  {row['status']:<9} overlap={row['overlap']:.3f}")
        print(f"     quote: {row['quote'][:100]!r}")
        if row["note"]:
            print(f"     note : {row['note']}")

    print(f"\n{len(rows)} quote(s) checked, {len(failed)} ungrounded.")
    if failed:
        print(
            "These were admitted under the pre-2026-08-19 bag-of-words rule and would\n"
            "be dropped by extraction today. Reject them rather than approving:\n"
            "  forge proposals reject <id>"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
