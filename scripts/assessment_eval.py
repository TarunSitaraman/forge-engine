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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="scripted", choices=("scripted", "ollama", "cloud"))
    parser.add_argument("--dataset", type=Path, default=ROOT / DEFAULT_ASSESSMENT_SET)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    dataset = AssessmentDataset.load(args.dataset)
    workdir = Path(tempfile.mkdtemp(prefix="forge-assess-eval-"))
    settings = Settings.load(state_dir=workdir / "state")

    scripted = args.provider == "scripted"
    if scripted:
        provider = scripted_provider(dataset)
        provider_id, model_id = "mock", "mock-1"
    else:
        import os

        os.environ["FORGE_LLM_PROVIDER"] = args.provider
        settings = Settings.load(state_dir=workdir / "state")
        provider = get_provider(settings)
        reachable, detail = provider.health()
        if not reachable:
            print(f"provider {args.provider!r} is unavailable: {detail}")
            print("\nNo results. This is reported rather than substituted with a weaker model.")
            shutil.rmtree(workdir, ignore_errors=True)
            return 2
        provider_id, model_id = provider_identity(provider, "analysis")

    report = AssessmentReport(
        provider_id=provider_id, model_id=model_id, scripted=scripted
    )

    for index, case in enumerate(dataset):
        store = SqliteStore(workdir / f"case-{index}.db")
        store.initialize()
        claim, evidence_span, source = build_case(store, case, index)
        assessor = EvidenceAssessor(
            store, provider, provider_id=provider_id, model_id=model_id
        )

        result = CaseResult(case_id=case.id, expected=case.expected_classification.value)
        started = time.perf_counter()
        try:
            batch = assessor.assess([evidence_span], [claim])
        except ProviderUnavailable as exc:
            result.detail = f"provider unavailable: {exc}"
            report.results.append(result)
            store.close()
            continue
        result.latency_ms = (time.perf_counter() - started) * 1000

        if not batch.ok:
            result.detail = f"{batch.outcome.value}: {batch.detail[:120]}"
            report.results.append(result)
            store.close()
            continue

        if not batch.records:
            result.structured_output_valid = True  # it validated; it was then rejected
            result.detail = (
                f"assessment rejected: {batch.rejected[0]['reason'][:100]}"
                if batch.rejected
                else "no assessment produced"
            )
            report.results.append(result)
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

        report.results.append(result)
        store.close()

    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"dataset : {dataset.path} (v{dataset.version}, {len(dataset)} cases)")
        print(f"balance : {dataset.by_classification()}")
        print(f"provider: {report.provider_id} / {report.model_id}\n")
        print("  " + report.headline())
        print()
        for result in report.results:
            mark = "ok " if result.classification_correct and result.proposal_correct else "FAIL"
            print(
                f"  [{mark}] {result.case_id:<30} expected={result.expected:<22} "
                f"actual={result.actual or '-'}"
            )
            if result.detail:
                print(f"          {result.detail}")
        if report.scripted:
            print(f"\nCAVEAT: {payload['caveat']}")

    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if report.proposal_correctness == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
