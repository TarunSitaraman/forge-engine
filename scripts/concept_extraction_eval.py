#!/usr/bin/env python3
"""Score concept extraction against the vault's own 545 page names.

    python3 scripts/concept_extraction_eval.py                        # scripted (offline)
    python3 scripts/concept_extraction_eval.py --provider cloud --limit 40
    python3 scripts/concept_extraction_eval.py --provider ollama --model qwen3:8b

**The reference set nobody wrote for this purpose.** `forge bootstrap` derives
545 concepts from the vault's directory listing with zero model calls: a human
decided `Binary Search` deserved one canonical page and created it. That is the
judgement extraction is trying to reproduce, and it is already on disk. The
labelled set in `forge extraction-eval` has six hand-written cases whose
expected concepts were chosen by whoever wrote the prompt; this one has 545
that were not.

**What is reported, and what is deliberately not.** Self-recovery (did the page
about X yield X?), junk rate against the strings this corpus has actually
emitted as concepts, and off-vocabulary rate as a shape measurement that is not
a quality claim in either direction. Not global recall: a page about B-trees is
not supposed to mention 544 other concepts, so "found 3 of 545" would be an
arithmetic fact and not a finding. See `forge.evaluation.corpus_extraction`.

**Read the caveat under `--provider scripted`.** That mode answers with the
page's own name, so self-recovery is 1.0 by construction. What it genuinely
measures is the harness: span construction from real vault files, schema
validation, the grounding check against real span text, normalization, and the
scoring itself. Only a real provider measures extraction quality.

**Cost.** One page is up to `--max-spans` concept calls plus the same number of
claim calls. Default `--limit 40` and `--max-spans 3` is roughly 240 calls,
which is what a rate-limited hosted key will tolerate in one sitting. Raise
`--limit` when you have the budget; the sample is seeded, so a larger run is a
superset of nothing and must be compared as its own measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.bootstrap import build_plan
from forge.config import Settings
from forge.corpus.indexer import CorpusIndexer
from forge.evaluation.corpus_extraction import (
    CorpusExtractionReport,
    VaultPage,
    Vocabulary,
    run,
)
from forge.evaluation.extraction import DEFAULT_EXTRACTION_SET, ExtractionDataset
from forge.extraction import CandidateExtractor

# Imported rather than re-derived: these are the extractor's own rules for
# which spans are worth a call. Approximating them here would let this script
# count a page as thin that the extractor would have extracted from, or the
# reverse, and the difference would land silently in self-recovery.
from forge.extraction.extractor import MIN_SPAN_CHARS, _is_navigation, _tokens
from forge.llm import MockProvider, get_provider
from forge.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]


def forbidden_strings(path: Path | None = None) -> list[str]:
    """The junk list, taken from the labelled set rather than rewritten here.

    Sharing it is the point: `extraction-eval`'s junk rate and this one's then
    mean the same thing and can be quoted side by side. These 25 strings are
    not hypothetical bad output — every one was observed coming out of this
    corpus in the August 2026 samples.
    """
    dataset = ExtractionDataset.load(path or ROOT / DEFAULT_EXTRACTION_SET)
    seen: dict[str, None] = {}
    for case in dataset:
        for term in case.forbidden:
            seen.setdefault(term, None)
    return list(seen)


#: A candidate line must be this proportion distinct tokens before it is used
#: to build an adversarial probe. Reversing `self.parent[x] = self.parent[
#: self.parent[x]]` produces a near-identical token sequence, and reversing a
#: grid of ones and zeros produces the same multiset in nearly the same order,
#: so neither is a reordering in any sense a grounding check should catch.
#: Measured 2026-09-06: without this filter 11 of 545 pages emitted a "probe"
#: that was really the original, and the eval reported them as the check
#: failing. That was the harness, not the check.
MIN_DISTINCT_TOKEN_RATIO = 0.6

#: Shortest line worth quoting, and the fewest tokens whose order carries
#: information. Below six, reversal is too easily a coincidence.
MIN_PROBE_CHARS = 40
MIN_PROBE_TOKENS = 6


def adversarial_probe(body: str) -> tuple[str, str] | None:
    """Pick a line of the span and return it with its token order reversed.

    Returns ``None`` when the span has no line whose order can be meaningfully
    reversed, rather than emitting a degenerate probe and counting its survival
    against the check.

    **Selection never consults `_grounded`.** It would be trivial to try
    scrambles until one the check rejects, and the resulting test would pass by
    construction and detect nothing. Both criteria here are properties of the
    input alone: enough tokens, and enough of them distinct.

    Reversal is at the **token** level, not the word level. `result =
    groupAnagrams(["eat","tea","tan","ate","nat","bat"])` is four words, one of
    which holds six tokens, so reversing words leaves the token order almost
    untouched. That is what 11 pages of this vault were quietly exercising.
    """
    for raw in body.splitlines():
        line = raw.strip()
        if len(line) < MIN_PROBE_CHARS:
            continue
        tokens = _tokens(line)
        if len(tokens) < MIN_PROBE_TOKENS:
            continue
        if len(set(tokens)) / len(tokens) < MIN_DISTINCT_TOKEN_RATIO:
            continue
        return line[:400], " ".join(reversed(tokens))
    return None


def scripted_provider(current: dict) -> MockProvider:
    """Answers each call from the page under test, exercising the real path.

    Concepts: the page's own name, so the recovery path is driven end to end.

    Claims: **a real quote, plus an adversarial probe wherever one can be
    built.** The real quote is lifted verbatim out of the span the prompt
    actually contains, parsed back out of the request rather than invented, the
    same technique the assessment eval uses for span ids. The probe is that
    line's tokens in reverse, so it is assembled entirely from the span's own
    vocabulary and only the order-preserving half of `_grounded` can reject it.

    A check that cannot fail in the only mode runnable offline is worse than no
    check, because it reassures. So every probe emitted must come back dropped,
    and the run reports the count rather than leaving the reader to infer it
    from a rate.
    """

    def respond(request):
        text = request.messages[-1].content
        body = text.split("--- TEXT START ---", 1)[-1].split("--- TEXT END ---", 1)[0].strip()
        name = current["page"].canonical_name
        if text.lstrip().startswith("List the individual factual assertions"):
            probe = adversarial_probe(body)
            fallback = next(
                (line.strip() for line in body.splitlines() if len(line.strip()) >= MIN_PROBE_CHARS),
                body[:200],
            )[:400]
            quote = probe[0] if probe else fallback
            claims = [
                {
                    "statement": f"{name} is described in this section.",
                    "evidence_quote": quote,
                    "concept": name,
                }
            ]
            if probe is not None:
                current["probes"] = current.get("probes", 0) + 1
                by_page = current.setdefault("probes_by_page", {})
                path = current["page"].path
                by_page[path] = by_page.get(path, 0) + 1
                claims.append(
                    {
                        "statement": f"{name} has an unsupported property.",
                        "evidence_quote": probe[1],
                        "concept": name,
                    }
                )
            else:
                current["unprobed_spans"] = current.get("unprobed_spans", 0) + 1
            return json.dumps({"claims": claims})
        return json.dumps({"concepts": [{"name": name, "kind": "concept"}]})

    return MockProvider(responder=respond)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="scripted", choices=("scripted", "ollama", "cloud"))
    parser.add_argument("--vault", type=Path, default=None, help="Defaults to FORGE_VAULT_PATH.")
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        metavar="N",
        help="Pages to sample. 0 means all of them, which is 545 pages of calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260906,
        help="Sampling seed. Fixed so two runs of the same size score the same pages.",
    )
    parser.add_argument(
        "--max-spans",
        type=int,
        default=3,
        help=(
            "Spans per page sent to the model. The production default is 12; "
            "3 is the eval default because a hosted key will not survive 545 "
            "pages at 12. Runs at different values are not comparable."
        ),
    )
    parser.add_argument("--model", default=None, help="Override the model for this run.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--detail", action="store_true", help="Per-page emitted concepts.")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 (all) or positive")
    if args.max_spans < 1:
        parser.error("--max-spans must be at least 1")
    if args.model and args.provider == "scripted":
        parser.error(
            "--model needs a real provider: the scripted one answers from the "
            "vault and never reaches a model"
        )

    # Without this, structlog falls back to its default PrintLogger, which
    # writes to stdout — straight into the file `--json` is redirected to,
    # making the report unparseable.
    configure_logging()

    if args.provider != "scripted":
        os.environ["FORGE_LLM_PROVIDER"] = args.provider
        if args.model:
            key = "FORGE_CLOUD_MODEL" if args.provider == "cloud" else "FORGE_MODEL_DEFAULT"
            os.environ[key] = args.model

    try:
        settings = Settings.load(vault_path=args.vault) if args.vault else Settings.load()
    except Exception as exc:  # noqa: BLE001 - any config failure is "no vault", reported not raised
        print(f"no vault: {exc}", file=sys.stderr)
        print(
            "Set FORGE_VAULT_PATH or pass --vault. Vault resolution looks only "
            "upward from the working directory, so this must point at the notes.",
            file=sys.stderr,
        )
        return 2

    indexer = CorpusIndexer(settings)
    index = indexer.build_index()
    by_path = {f.path: f for f in index.files}
    plan = build_plan(index, decided=indexer._decided_targets())
    vocabulary = Vocabulary.from_concepts(plan.concepts)

    population = [
        VaultPage(
            path=c.vault_path,
            canonical_name=c.canonical_name,
            title=(by_path[c.vault_path].title or "") if c.vault_path in by_path else "",
        )
        for c in plan.concepts
        if c.vault_path
    ]
    population.sort(key=lambda p: p.path)  # deterministic before sampling

    spans_cache: dict[str, list] = {}

    def spans_for(page: VaultPage) -> list:
        if page.path not in spans_cache:
            indexed = by_path[page.path]
            source = next(s for s in indexer.to_sources(index) if s.locator == page.path)
            _, spans = indexer.to_document_and_spans(indexed, source)
            spans_cache[page.path] = spans
        return spans_cache[page.path]

    def extractable(page: VaultPage) -> bool:
        """Would the extractor spend a call on this page at all?

        A page with nothing above the extractor's own floor emits nothing, and
        scoring that as a failure to recover its concept would blame the model
        for a page that was never sent. Those pages are excluded from the
        sample and counted, rather than scored as misses.
        """
        return any(
            len(s.text.strip()) >= MIN_SPAN_CHARS and not _is_navigation(s.text)
            for s in spans_for(page)
        )

    eligible = [p for p in population if extractable(p)]
    thin = len(population) - len(eligible)

    rng = random.Random(args.seed)
    sample = eligible if args.limit == 0 else rng.sample(eligible, min(args.limit, len(eligible)))
    sample.sort(key=lambda p: p.path)

    # Text is attached only to the sampled pages, and only now: the grounding
    # re-check needs it, and reading 545 files to score 40 is waste. It is the
    # whole page rather than the span the model saw, deliberately — a quote
    # that grounds in a span grounds in the page, and quotes the extractor
    # already dropped are counted through `dropped_claims`, which is where the
    # rate can actually move. Leaving `text` empty scored grounding 0.000 on
    # the first smoke run, for claims that were in fact quoted verbatim.
    sample = [
        replace(page, text=(settings.vault_path / page.path).read_text(
            encoding="utf-8", errors="replace"
        ))
        for page in sample
    ]

    current: dict = {"page": sample[0] if sample else None}
    if args.provider == "scripted":
        provider = scripted_provider(current)
    else:
        provider = get_provider(settings)
        reachable, detail = provider.health()
        if not reachable:
            print(f"provider {args.provider!r} is unavailable: {detail}")
            print("\nNo results. This is reported rather than substituted with a weaker model.")
            return 2

    extractor = CandidateExtractor(provider, max_spans=args.max_spans)
    forbidden = forbidden_strings()
    total = len(sample)
    started = time.perf_counter()

    def announce(score, position: int) -> None:
        """Per-page progress on **stderr**.

        stdout is the report, and under `--json` it is a redirect that is only
        written once every page has run — so a working run and a hung one look
        identical for the twenty minutes a rate-limited provider spends in
        backoff. stderr stays on the terminal through the redirect.
        """
        if not score.complete:
            mark = "----"
        elif score.recovered:
            mark = "ok  "
        else:
            mark = "MISS"
        print(
            f"[{position:>3}/{total}] {mark} {score.path[:58]:<58} "
            f"{len(score.emitted):>2} concept(s), {len(score.junk)} junk",
            file=sys.stderr,
            flush=True,
        )

    # The scripted responder answers from whichever page is under test, so the
    # holder has to be advanced in step with the driver. `spans_for` is the one
    # hook called once per page before any model call.
    def spans_and_track(page: VaultPage) -> list:
        current["page"] = page
        return spans_for(page)

    report: CorpusExtractionReport = run(
        sample,
        extractor,
        vocabulary,
        spans_and_track,
        forbidden=forbidden,
        on_page=announce,
    )
    report.population = len(eligible)
    report.seed = args.seed
    report.duration_seconds = time.perf_counter() - started

    # The scripted mode's own verdict. Every probe emitted must come back
    # dropped; a survivor is a regression in `_grounded`, not a bad page.
    #
    # Compared per page against the probes *that page* was given. The first
    # version compared each page's kept claims to its dropped ones, which
    # flagged all 76 pages holding a span with no viable probe and reported
    # "1536 emitted, 1536 dropped, 76 survived" — three numbers that cannot all
    # be true. A survivor is a page the extractor dropped fewer claims for than
    # it was probed with.
    probes = int(current.get("probes", 0))
    probes_by_page: dict = current.get("probes_by_page", {})
    dropped = sum(s.dropped_claims for s in report.complete)
    survivors = [
        s for s in report.complete if s.dropped_claims < probes_by_page.get(s.path, 0)
    ]

    payload = {
        **report.to_dict(),
        "adversarial_probes": probes,
        "probes_dropped": dropped,
        "probe_survivors": [s.path for s in survivors],
        "spans_without_a_probe": int(current.get("unprobed_spans", 0)),
        "provider": args.provider,
        "max_spans": args.max_spans,
        "vault": str(settings.vault_path),
        "pages_excluded_as_thin": thin,
        "concept_pages": len(plan.concepts),
        "scripted": args.provider == "scripted",
    }

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"vault      : {settings.vault_path}")
        print(
            f"vocabulary : {len(vocabulary)} distinct names over "
            f"{len(plan.concepts)} concept pages from `forge bootstrap`"
        )
        print(
            f"excluded   : {thin} page(s) with nothing above the extractor's own floor"
        )
        print(f"sample     : {len(sample)} of {len(eligible)} eligible pages, seed {args.seed}")
        print(f"provider   : {args.provider}, max-spans {args.max_spans}\n")
        print(f"  {report.summary_line()}\n")
        print(
            f"  self-recovery   {report.self_recovery:.3f}   "
            "pages whose own concept came back out of their own text"
        )
        print(
            f"  junk rate       {report.junk_rate:.3f}   "
            f"emitted names on the observed-junk list ({len(forbidden)} strings)"
        )
        print(
            f"  off-vocabulary  {report.off_vocabulary_rate:.3f}   "
            "emitted names matching no vault page (shape, NOT a quality claim)"
        )
        print(
            f"  grounding       {report.grounding_rate:.3f}   "
            "quotes really in the span"
        )
        print(f"  concepts/page   {report.concepts_per_page:.2f}   over {report.emitted_total} emitted")

        if not report.trustworthy:
            print(
                f"\n  UNTRUSTWORTHY — {len(report.failed)} of {len(report.scores)} page(s) "
                "did not complete.\n"
                f"  Every rate above is over the {len(report.complete)} that did, and none of\n"
                "  them is a property of this model until the run is clean. A page that\n"
                "  emitted nothing scores zero junk, which is absence of output, not quality.",
                file=sys.stderr,
            )
            for score in report.failed:
                print(f"    [{score.path}] {score.status}", file=sys.stderr)
                for line in score.failures or ([score.error] if score.error else []):
                    print(f"        {line}", file=sys.stderr)
            shapes = {line for s in report.failed for line in s.failures}
            if len(report.failed) == len(report.scores) and len(shapes) <= 1:
                print(
                    "\n  Every page failed identically — that is the provider or the\n"
                    "  configuration, not the model's extraction quality. Check\n"
                    "  `forge model-test` and the model name your endpoint accepts.",
                    file=sys.stderr,
                )

        if args.detail:
            print()
            for score in report.scores:
                mark = "" if score.complete else f"  [{score.status}, NOT SCORED]"
                got = score.recovered_as or "-"
                print(f"  [{score.path}]{mark}")
                print(f"      wanted : {score.canonical_name}   got: {got}")
                if score.emitted:
                    print(f"      emitted: {', '.join(score.emitted)}")
                if score.junk:
                    print(f"      JUNK   : {', '.join(score.junk)}")

        if args.provider == "scripted":
            print(
                f"\nadversarial probes: {probes} emitted, {dropped} dropped, "
                f"{probes - dropped} survived on {len(survivors)} page(s)"
            )
            if survivors:
                print(
                    "  *** REGRESSION: a quote built by reversing a span's own token\n"
                    "  order was accepted as evidence. That is the defect the window in\n"
                    "  `_grounded` exists to prevent. Pages:"
                )
                for score in survivors[:10]:
                    print(f"    {score.path}")
            else:
                print("  every probe was rejected, which is the check working.")
            if current.get("unprobed_spans"):
                print(
                    f"  {current['unprobed_spans']} span(s) had no line whose token order\n"
                    "  could be meaningfully reversed and were left unprobed rather than\n"
                    "  counted against the check."
                )
            print(
                "\nCAVEAT: the scripted provider answers with the page's own name, so "
                "self-recovery\nis 1.000 by construction and says nothing about any model. "
                "What this mode does\nmeasure is the harness: span construction from real "
                "vault files, schema validation,\nthe grounding check, normalization and "
                "scoring. Use --provider cloud or ollama for\na number about extraction."
                "\n\nRead the probe line above rather than the grounding rate. The rate "
                "sits a little\nover 0.500 because spans with no line whose token order can "
                "be meaningfully\nreversed contribute a kept claim and no probe. The exact "
                "statement is the one\nthe probe line makes: every probe emitted came back "
                "dropped."
            )

    if args.provider == "scripted" and survivors:
        return 1
    return 0 if report.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
