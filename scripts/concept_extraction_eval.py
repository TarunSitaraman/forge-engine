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
from forge.extraction.extractor import MIN_SPAN_CHARS, _is_navigation
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


def scripted_provider(current: dict) -> MockProvider:
    """Answers each call from the page under test, exercising the real path.

    Concepts: the page's own name, so the recovery path is driven end to end.

    Claims: **two per span, one grounded and one not.** The grounded quote is
    lifted verbatim out of the span the prompt actually contains, parsed back
    out of the request rather than invented — the same technique the assessment
    eval uses for span ids. The ungrounded one is that same sentence with its
    words reversed, so it is built entirely from the span's own vocabulary and
    can only be caught by the ordered-overlap check rather than by a word the
    span does not have.

    A check that cannot fail in the only mode runnable offline is worse than no
    check, because it reassures. So scripted grounding is 0.500 by construction
    and a run reporting 1.000 means the drop path stopped working.
    """

    def respond(request):
        text = request.messages[-1].content
        body = text.split("--- TEXT START ---", 1)[-1].split("--- TEXT END ---", 1)[0].strip()
        name = current["page"].canonical_name
        if text.lstrip().startswith("List the individual factual assertions"):
            quote = next(
                (line.strip() for line in body.splitlines() if len(line.strip()) >= 40),
                body[:200],
            )[:400]
            scrambled = " ".join(reversed(quote.split()))
            return json.dumps(
                {
                    "claims": [
                        {
                            "statement": f"{name} is described in this section.",
                            "evidence_quote": quote,
                            "concept": name,
                        },
                        {
                            "statement": f"{name} has an unsupported property.",
                            "evidence_quote": scrambled,
                            "concept": name,
                        },
                    ]
                }
            )
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

    payload = {
        **report.to_dict(),
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
                "\nCAVEAT: the scripted provider answers with the page's own name, so "
                "self-recovery\nis 1.000 by construction and says nothing about any model. "
                "What this mode does\nmeasure is the harness: span construction from real "
                "vault files, schema validation,\nthe grounding check, normalization and "
                "scoring. Use --provider cloud or ollama for\na number about extraction."
                "\n\nGrounding should read 0.500 here: the scripted answer pairs a verbatim "
                "quote with\none scrambled from the span's own words, so a run reporting "
                "1.000 means the drop\npath has stopped working."
            )

    return 0 if report.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
