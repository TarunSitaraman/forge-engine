#!/usr/bin/env python3
"""Run the evidence-assessment evaluation set through the real pipeline.

    python3 scripts/assessment_eval.py                    # scripted (offline)
    python3 scripts/assessment_eval.py --provider ollama  # a real local model
    python3 scripts/assessment_eval.py --provider cloud   # a real hosted model

Each case builds a real store, a real span, and a real claim, then drives the
production :class:`EvidenceAssessor` and :class:`EvolutionProposer`. Only the
provider changes between modes, which is the point: the pipeline being measured
is the one that ships.

**Read the caveat in the output.** With the scripted provider, classification
accuracy is 1.0 by construction — the script answers with the expected label.
What that mode genuinely measures is the pipeline: schema validation, the
grounding check against real span ids, the classification-to-proposal mapping,
and cache effectiveness. Only a real provider measures classification quality.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.config import Settings  # noqa: E402
from forge.logging import configure_logging  # noqa: E402
from forge.domain import (  # noqa: E402
    AssessmentClass,
    Claim,
    Concept,
    ConceptKind,
    Derivation,
    Document,
    EvidenceLink,
    EvidenceRelation,
    Provenance,
    ProvenanceTier,
    Source,
    SourceKind,
    Span,
)
from forge.evaluation.assessment import (  # noqa: E402
    DEFAULT_ASSESSMENT_SET,
    AssessmentDataset,
    AssessmentReport,
    CaseResult,
)
from forge.evolution import EvidenceAssessor, EvolutionProposer  # noqa: E402
from forge.llm import MockProvider, get_provider, provider_identity  # noqa: E402
from forge.llm.base import ProviderUnavailable  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scripted_provider(dataset: AssessmentDataset) -> MockProvider:
    """Answers each case with its expected label, citing the real span shown.

    The span id is parsed out of the prompt Forge actually sent, so the
    downstream grounding check is exercised for real.
    """
    expected = {case.evidence[:60]: case for case in dataset}

    def respond(request):
        text = request.messages[-1].content
        claim_ids = re.findall(r"\[claim_id: ([^\]]+)\]", text)
        span_ids = re.findall(r"\[span_id: ([^\]]+)\]", text)
        case = next((c for key, c in expected.items() if key in text), None)
        if case is None or not claim_ids or not span_ids:
            return json.dumps({"assessments": []})
        return json.dumps(
            {
                "assessments": [
                    {
                        "claim_id": claim_ids[0],
                        "classification": case.expected_classification.value,
                        "rationale": case.note or "assessed against the provided evidence",
                        "evidence_span_ids": [span_ids[0]],
                        "refined_statement": case.refined_statement,
                    }
                ]
            }
        )

    return MockProvider(responder=respond)


def build_case(store: SqliteStore, case, index: int):
    """One real concept, claim, and evidence span per case."""
    provenance = Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent="eval-harness",
        model_id="fixture",
    )
    concept = Concept(
        id=Concept.make_id(f"Concept {index}"),
        canonical_name=f"Concept {index}",
        kind=ConceptKind.TECHNOLOGY,
        provenance=provenance,
        origin_proposal_id="eval",
    )
    store.put_concept(concept)

    source = Source.for_path(
        f"eval/case-{index}.pdf", kind=SourceKind.PDF, content_hash=f"case-{index}"
    )
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, f"case-{index}"),
        source_id=source.id,
        parser="eval",
        parser_version="1",
        content_hash=f"case-{index}",
    )
    store.put_document(document)

    claim_span = Span(
        id=Span.make_id(document.id, 0, "p.1"),
        document_id=document.id,
        ordinal=0,
        locator="p.1 L1-L2",
        start_line=1,
        end_line=2,
        text=case.claim,
        content_hash=f"claim-{index}",
        page=1,
    )
    evidence_span = Span(
        id=Span.make_id(document.id, 1, "p.2"),
        document_id=document.id,
        ordinal=1,
        locator="p.2 L1-L4",
        heading_path=("Findings",),
        start_line=1,
        end_line=4,
        text=case.evidence,
        content_hash=f"evidence-{index}",
        page=2,
    )
    store.put_spans([claim_span, evidence_span])

    claim = Claim(
        id=Claim.make_id(case.claim, claim_span.id),
        statement=case.claim,
        subject_concept_id=concept.id,
        provenance=provenance,
    )
    store.put_claim(
        claim,
        [
            EvidenceLink(
                id=EvidenceLink.make_id(claim.id, claim_span.id, EvidenceRelation.INFERS_FROM),
                claim_id=claim.id,
                span_id=claim_span.id,
                relation=EvidenceRelation.INFERS_FROM,
                provenance=provenance,
            )
        ],
    )
    return claim, evidence_span, source


def stability(reports: list[AssessmentReport]) -> dict:
    """How often each case got the right answer, across repeated runs.

    The reason this exists: on 2026-09-05 the fitted set scored 15/21 on the
    same model, prompt and command that had scored 18/21 the day before, and
    the three cases that differed were precisely the three that a prompt
    revision had been credited with fixing. A single run cannot tell a fixed
    case from a lucky one, so anything argued from a one- or two-case delta
    needs this table underneath it.
    """
    per_case: dict[str, dict] = {}
    for report in reports:
        for result in report.results:
            entry = per_case.setdefault(result.case_id, {"correct": 0, "answers": {}})
            entry["correct"] += int(result.classification_correct)
            answer = result.actual or "no result"
            entry["answers"][answer] = entry["answers"].get(answer, 0) + 1
    scores = [sum(r.classification_correct for r in report.results) for report in reports]
    return {
        "runs": len(reports),
        "min_correct": min(scores) if scores else 0,
        "max_correct": max(scores) if scores else 0,
        "scores": scores,
        "unstable": sum(1 for e in per_case.values() if len(e["answers"]) > 1),
        "cases": per_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="scripted", choices=("scripted", "ollama", "cloud"))
    parser.add_argument("--dataset", type=Path, default=ROOT / DEFAULT_ASSESSMENT_SET)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run the whole set N times and report the spread. The same model, "
            "prompt and set scored 18/21 and 15/21 on consecutive days, so a "
            "single run cannot support a one- or two-case delta. Each "
            "repetition gets its own store, so none of them are served the "
            "previous run's cached assessments."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the model for this run without editing forge.env. The "
            "point is the comparison: three prompt-side fixes have now failed "
            "to move INSUFFICIENT_EVIDENCE on openai/gpt-oss-120b, so the open "
            "question is whether a stronger model handles the class at all. "
            "Sets FORGE_CLOUD_MODEL for --provider cloud and FORGE_MODEL_DEFAULT "
            "for --provider ollama."
        ),
    )
    parser.add_argument(
        "--corroborate",
        action="store_true",
        help=(
            "Enable the second corroboration pass over SUPPORTS/REFINES. Off "
            "by default: measured 13/18 with and 13/18 without on the held-out "
            "set, fixing one case and breaking another."
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.model and args.provider == "scripted":
        parser.error(
            "--model needs a real provider: the scripted one answers from the "
            "dataset and never reaches a model"
        )

    # Without this, structlog has never been configured and falls back to its
    # default PrintLogger, which writes to **stdout** — straight into the file
    # `--json` redirects, making the report unparseable. configure_logging
    # routes through stdlib logging, which this engine points at stderr.
    configure_logging()

    dataset = AssessmentDataset.load(args.dataset)
    workdir = Path(tempfile.mkdtemp(prefix="forge-assess-eval-"))
    settings = Settings.load(state_dir=workdir / "state")

    scripted = args.provider == "scripted"
    corroborate = args.corroborate
    if scripted:
        provider = scripted_provider(dataset)
        provider_id, model_id = "mock", "mock-1"
    else:
        import os

        os.environ["FORGE_LLM_PROVIDER"] = args.provider
        if args.model:
            # The cloud provider reads one model; Ollama binds four roles that
            # each fall back to FORGE_MODEL_DEFAULT. Setting the role default
            # rather than the four role variables leaves an explicitly bound
            # role alone, which is the behaviour someone overriding one model
            # on the command line would expect.
            if args.provider == "cloud":
                os.environ["FORGE_CLOUD_MODEL"] = args.model
            else:
                os.environ["FORGE_MODEL_DEFAULT"] = args.model
        settings = Settings.load(state_dir=workdir / "state")
        provider = get_provider(settings)
        reachable, detail = provider.health()
        if not reachable:
            print(f"provider {args.provider!r} is unavailable: {detail}")
            print("\nNo results. This is reported rather than substituted with a weaker model.")
            shutil.rmtree(workdir, ignore_errors=True)
            return 2
        provider_id, model_id = provider_identity(provider, "analysis")

    reports: list[AssessmentReport] = []
    total = len(dataset)

    for repetition in range(args.repeat):
        report = AssessmentReport(
            provider_id=provider_id, model_id=model_id, scripted=scripted
        )
        reports.append(report)
        # A fresh directory per repetition. Sharing one would hand every
        # repeat a warm assessment cache and return the first run's answers
        # N times, which is the opposite of what a variance measurement needs.
        run_dir = workdir / f"run-{repetition}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.repeat > 1:
            print(f"--- run {repetition + 1}/{args.repeat} ---", file=sys.stderr, flush=True)

        def note(result: CaseResult, position: int, report: AssessmentReport = report) -> None:
            """Append the result, and say so on **stderr**.

            stdout is the report. Under ``--json`` that is redirected to a file
            which is only written once every case has run, so a working run and
            a hung one look identical for several minutes — and on a
            rate-limited host, where the provider sits in backoff, that is
            exactly when you want to see it moving. stderr stays on the
            terminal through the redirect.
            """
            report.results.append(result)
            mark = "ok  " if result.classification_correct else "MISS"
            detail = result.actual or (result.detail[:40] if result.detail else "no result")
            print(
                f"[{position:>2}/{total}] {mark} {result.case_id:<44} "
                f"{detail:<26} {result.latency_ms / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        for index, case in enumerate(dataset):
            store = SqliteStore(run_dir / f"case-{index}.db")
            store.initialize()
            claim, evidence_span, source = build_case(store, case, index)
            assessor = EvidenceAssessor(
                store,
                provider,
                provider_id=provider_id,
                model_id=model_id,
                corroborate=corroborate,
            )

            result = CaseResult(case_id=case.id, expected=case.expected_classification.value)
            started = time.perf_counter()
            try:
                batch = assessor.assess([evidence_span], [claim])
            except ProviderUnavailable as exc:
                result.detail = f"provider unavailable: {exc}"
                note(result, index + 1)
                store.close()
                continue
            result.latency_ms = (time.perf_counter() - started) * 1000

            if not batch.ok:
                result.detail = f"{batch.outcome.value}: {batch.detail[:120]}"
                note(result, index + 1)
                store.close()
                continue

            if not batch.records:
                result.structured_output_valid = True  # it validated; it was then rejected
                result.detail = (
                    f"assessment rejected: {batch.rejected[0]['reason'][:100]}"
                    if batch.rejected
                    else "no assessment produced"
                )
                note(result, index + 1)
                store.close()
                continue

            record = batch.records[0]
            result.structured_output_valid = True
            result.actual = record.classification.value
            result.classification_correct = record.classification is case.expected_classification
            # Grounded means every cited span resolves in the store *and* was one
            # of the spans shown. The assessor rejects anything else, so a record
            # existing at all implies grounding — asserted rather than assumed.
            result.grounded = all(
                store.get_span(span_id) is not None for span_id in record.evidence_span_ids
            ) and set(record.evidence_span_ids) <= {evidence_span.id}

            proposer = EvolutionProposer(store, workflow_id="eval", source_id=source.id)
            refined = {record.claim_id: assessor.refined_statement_for(record)}
            proposals = proposer.propose([record], refined_statements=refined)
            produced = proposals.created[0].type if proposals.created else None
            result.proposal_correct = produced == case.expected_proposal
            if not result.proposal_correct:
                result.detail = (
                    f"expected proposal {case.expected_proposal} but produced {produced}"
                )

            repeat = assessor.assess([evidence_span], [claim])
            result.cached_on_repeat = repeat.cache.hits == 1 and repeat.llm_calls == 0

            note(result, index + 1)
            store.close()

    report = reports[0]
    payload: dict = report.to_dict()
    if args.repeat > 1:
        payload = {
            "repeat": args.repeat,
            "provider_id": provider_id,
            "model_id": model_id,
            "scripted": scripted,
            "runs": [r.to_dict() for r in reports],
            "stability": stability(reports),
        }

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"dataset : {dataset.path} (v{dataset.version}, {len(dataset)} cases)")
        print(f"balance : {dataset.by_classification()}")
        print(f"provider: {provider_id} / {model_id}")
        print(
            "corroborate: "
            + ("on (second pass over SUPPORTS/REFINES)" if corroborate else "OFF")
            + "\n"
        )
        for index, run in enumerate(reports):
            prefix = f"  run {index + 1}  " if args.repeat > 1 else "  "
            print(prefix + run.headline())
        print()

        if args.repeat > 1:
            marks = stability(reports)
            print(f"{'case':<44} {'correct':>9}  answers")
            for case_id, entry in marks["cases"].items():
                if entry["correct"] == args.repeat:
                    continue
                answers = ", ".join(f"{k}x{v}" for k, v in entry["answers"].items())
                print(f"{case_id:<44} {entry['correct']}/{args.repeat:<7} {answers}")
            print()
            print(
                f"score range {marks['min_correct']}-{marks['max_correct']} of "
                f"{len(dataset)} across {args.repeat} runs; "
                f"{marks['unstable']} case(s) answered inconsistently."
            )
            if marks["unstable"]:
                print(
                    "A delta smaller than this spread is not a result. "
                    "Measured 2026-09-05: the same model, prompt and set scored "
                    "18/21 and 15/21 on different days, and the three cases that "
                    "moved were exactly the three a prompt revision was credited "
                    "with fixing."
                )
        else:
            for result in report.results:
                mark = "ok " if result.classification_correct and result.proposal_correct else "FAIL"
                print(
                    f"  [{mark}] {result.case_id:<30} expected={result.expected:<22} "
                    f"actual={result.actual or '-'}"
                )
                if result.detail:
                    print(f"          {result.detail}")
        if report.scripted:
            print(f"\nCAVEAT: {report.to_dict()['caveat']}")

    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if all(r.proposal_correctness == 1.0 for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
