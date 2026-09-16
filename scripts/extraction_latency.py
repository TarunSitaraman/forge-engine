#!/usr/bin/env python3
"""Measure seconds per extraction call, so `forge extract-plan` can be believed.

    python3 scripts/extraction_latency.py --provider cloud --spans 10
    python3 scripts/extraction_latency.py --provider ollama --spans 10
    python3 scripts/extraction_latency.py --spans 10          # scripted, offline

`forge extract-plan` computes the call count for a run deterministically and
then refuses to turn it into hours without a `--seconds-per-call` you have
measured, on the grounds that an invented rate is a fabricated measurement.
That is the right refusal, but nothing measured the rate, so in practice the
number got borrowed from whatever document was nearest. This script is the
missing instrument, and it exists because of what the borrowing cost:

- `extraction-cost.md` §1: a whole-vault estimate of ~236 h, built by applying
  a latency to `forge index` spans when extraction runs over `forge ingest`
  spans, a 4x overcount. The conclusion drawn from it was "infeasible".
- `extraction-cost.md` §2: the 63 s/call from the Phase 4 *assessment* eval,
  applied to extraction, which sends a full span and asks for up to 15
  concepts. Real extraction measured ~228 s/call, 3.6x higher.
- The same section published that gap as 7x for three weeks, by dividing a
  per-span figure by a per-call one.

Three errors, one shape: a rate measured on one task, on one machine, quoted
for another. So this reports **per call and per span side by side**, names the
provider and model in the output, and prints the `extract-plan` command with
the measured rate already substituted in.

**It is a rate, not a quality measurement.** It says nothing about whether the
concepts are any good; `extraction_eval.py` and `concept_extraction_eval.py`
do that. Running it with the scripted provider measures the harness and will
report a rate near zero, which is correct and useless for planning: only a
real provider produces a number worth putting in a plan.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.config import Settings  # noqa: E402
from forge.extraction import CandidateExtractor  # noqa: E402
from forge.extraction.prompts import CONCEPT_INSTRUCTION  # noqa: E402
from forge.llm import get_provider  # noqa: E402
from forge.llm.mock import MockProvider  # noqa: E402
from forge.logging import configure_logging  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402

#: The span the timing loop is currently on, so the scripted responder can
#: quote it and pass the grounding check. Unused for real providers.
current_span: dict[str, str] = {}


def _provider(name: str, settings: Settings):
    if name == "scripted":
        # Each extraction call validates against its own schema, and both
        # reject the other's key, so one fixed payload cannot satisfy both.
        # The two are told apart by the instruction the extractor actually
        # sends, imported rather than guessed at from a keyword: a first
        # draft keyed on the word "claim" appearing anywhere in the prompt
        # and misrouted on spans whose own text discusses claims, which on
        # this vault is a lot of them.
        def respond(request):
            text = " ".join(m.content for m in request.messages)
            if CONCEPT_INSTRUCTION.split("\n")[0] in text:
                return json.dumps(
                    {
                        "concepts": [
                            {"name": "Placeholder", "kind": "concept", "mention": "placeholder"}
                        ]
                    }
                )
            # evidence_quote is grounding-checked against the span, so it is
            # quoted from the span the loop is on rather than invented. An
            # ungrounded claim is still a clean call, but it would print
            # "FAILED" in a table whose only job is to be read as a rate.
            quote = " ".join((current_span.get("text") or "").split()[:12])[:200]
            return json.dumps(
                {
                    "claims": [
                        {
                            "statement": "A placeholder claim, for timing only.",
                            "evidence_quote": quote or "placeholder",
                            "concept": "Placeholder",
                        }
                    ]
                }
            )

        return MockProvider(responder=respond)
    settings.llm.provider = "cloud" if name == "cloud" else "ollama"
    return get_provider(settings)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="scripted", choices=["scripted", "ollama", "cloud"])
    ap.add_argument("--spans", type=int, default=10, help="Spans to time. Each is 2 calls.")
    ap.add_argument("--vault", type=Path, default=None)
    ap.add_argument("--log-level", default="WARNING", help="Engine log level.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed, for a repeatable set.")
    args = ap.parse_args()

    # The output is a table; interleaved structlog lines make it unreadable.
    configure_logging(level=args.log_level)

    settings = Settings.load(vault_path=args.vault)
    store = SqliteStore(settings.db_path)
    store.initialize()

    # Ingestion spans, not index spans. Extraction runs inside the ingestion
    # pipeline, and confusing the two is the 4x overcount that produced the
    # retracted ~236 h estimate. Walking sources -> documents -> spans is what
    # `forge ingest` wrote, so this counts the same population a real run
    # would send.
    candidates = []
    for source in store.list_sources():
        for document in store.documents_for_source(source.id):
            candidates.extend(store.spans_for_document(document.id))

    # MIN_SPAN_CHARS is not the whole filter: `CandidateExtractor._select`
    # also drops navigation spans. Timing a span the extractor would skip
    # records a near-zero duration and zero calls, which dilutes the mean,
    # breaks the calls-per-span ratio, and drags min(per_span) under the 0.1 s
    # guard so the spread sentence is silently suppressed. Ask the extractor
    # itself which spans it wants rather than approximating its rules here.
    probe = CandidateExtractor(None, max_spans=len(candidates) or 1)
    eligible = list(probe._select(candidates))

    # Sample deterministically rather than taking the first N in store order.
    # A quotable rate must not depend on which document happens to sort first,
    # or drift when the vault is re-ingested in a different order.
    rng = random.Random(args.seed)
    spans = rng.sample(eligible, min(args.spans, len(eligible))) if eligible else []
    if not spans:
        print("no spans in the store: run `forge index` (and `forge ingest`) first", file=sys.stderr)
        return 2

    provider = _provider(args.provider, settings)
    reachable, detail = provider.health()
    if not reachable:
        print(f"provider not reachable: {detail}", file=sys.stderr)
        return 2

    # One span at a time, so a per-span spread is visible. A mean alone hides
    # the thing worth knowing: the local run that produced 228 s/call had one
    # span at 92 s and another at 235 s, and a gap over an hour with nothing
    # completing. A single number would have reported none of that.
    per_span: list[float] = []
    calls = 0
    extractor = CandidateExtractor(provider, max_spans=1)

    for i, span in enumerate(spans, 1):
        current_span["text"] = span.text
        started = time.perf_counter()
        # Throttle wait is not model latency. `get_provider` returns a
        # ThrottledProvider whenever FORGE_LLM_MIN_INTERVAL is set, and its
        # sleeps land inside this timed region. Without the subtraction, a run
        # paced at 40 s would publish a real 10 s/call as 40 s/call and then
        # print it into the extract-plan command, which is precisely the
        # fabricated rate this script exists to stop. assessment_eval.py and
        # concept_extraction_eval.py subtract it for the same reason.
        slept_before = getattr(provider, "slept_seconds", 0.0)
        result = extractor.extract([span])
        waited = getattr(provider, "slept_seconds", 0.0) - slept_before
        elapsed = time.perf_counter() - started - waited
        per_span.append(elapsed)
        calls += result.llm_calls
        print(
            f"  span {i:>2}  {elapsed:7.1f}s  {result.llm_calls} call(s)  "
            f"{len(result.concepts)} concept(s)  {len(result.claims)} claim(s)"
            + ("  FAILED" if result.failures else "")
        )

    total = sum(per_span)
    if not calls:
        print("\nno model calls were made; nothing to report as a rate", file=sys.stderr)
        return 1

    per_call = total / calls
    print(f"\nprovider      : {provider.capabilities.name}")
    print(f"model         : {extractor.model_id()}")
    print(f"spans         : {len(per_span)}")
    print(f"calls         : {calls}  ({calls / len(per_span):.1f} per span)")
    print(f"total         : {total:.1f}s (model time; throttle waits excluded)")
    print(f"per call      : {per_call:.1f}s")
    print(f"per span      : {statistics.mean(per_span):.1f}s mean, "
          f"{min(per_span):.1f}s min, {max(per_span):.1f}s max")
    if len(per_span) > 1:
        print(f"median/span   : {statistics.median(per_span):.1f}s")

    # The spread is the caveat that keeps getting dropped when a number is
    # quoted onward, so it is printed as a sentence rather than left to the
    # reader to compute from min and max. Below a tenth of a second the ratio
    # is measuring the timer, not the model: the scripted provider returns in
    # microseconds and would otherwise announce a dramatic-looking "13x".
    if len(per_span) > 1 and min(per_span) >= 0.1:
        print(f"\nslowest span was {max(per_span) / min(per_span):.1f}x the fastest. "
              "Quote the spread with the rate.")

    if provider.capabilities.name == "mock":
        print("\nThis was the scripted provider. It measured this harness, not a model,"
              "\nand the rate above is not usable for planning. Re-run with"
              "\n--provider cloud or --provider ollama for a number worth quoting.")
        store.close()
        return 0

    print(f"\nNow plan a real run with the rate you just measured:\n"
          f"  forge extract-plan . --seconds-per-call {per_call:.1f}")
    print("Quote it with the provider, the model and the spread. A rate measured"
          "\non one task, on one machine, is not a rate for another.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
