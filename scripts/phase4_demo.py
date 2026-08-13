#!/usr/bin/env python3
"""Phase 4 demonstration — Forge evaluates how new evidence changes what it knows.

    python3 scripts/phase4_demo.py

The story, in one line:

    Forge did not merely store the new paper. Forge evaluated how the new
    evidence affected existing knowledge, proposed a change, waited for human
    approval, and recorded the resulting revision.

Step 10 kills the process's workflow objects entirely and rebuilds them from
the checkpoint on disk, so "resumable" is demonstrated rather than asserted.

**No local model is reachable in this environment** (the sandbox network policy
blocks ollama.com and huggingface.co), so assessment runs against a *scripted*
provider through the real ``LLMProvider`` interface. Everything else — the
graph, the grounding check, the derivation cache, the proposal system, the
activation, the revision log — is the production path, unchanged. On a machine
with Ollama or a cloud key, the same arc runs with `forge evolve paper-b.pdf`.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.activation import ProposalActivator  # noqa: E402
from forge.config import Settings  # noqa: E402
from forge.domain import (  # noqa: E402
    EntityType,
    ProposalStatus,
    ProposalType,
    WorkflowStatus,
)
from forge.evolution.service import EvolutionService  # noqa: E402
from forge.extraction import CandidateExtractor  # noqa: E402
from forge.graph import KnowledgeGraph  # noqa: E402
from forge.ingestion import IngestionPipeline, IngestOptions  # noqa: E402
from forge.llm import MockProvider  # noqa: E402
from forge.proposals import ProposalService  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pdf"
PAPER_A = FIXTURES / "paper-a-rag-accuracy.pdf"
PAPER_B = FIXTURES / "paper-b-rag-failure-modes.pdf"
STEP = 0


def step(text: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 76}\n{STEP:>2}. {text}\n{'=' * 76}")


def extraction_provider() -> MockProvider:
    """Answers as a competent local extraction model would (Phase 2 path)."""

    def respond(request):
        content = request.messages[1].content
        if "concepts" in content and "assertions" not in content:
            return json.dumps(
                {
                    "concepts": [
                        {
                            "name": "Retrieval Augmented Generation",
                            "kind": "technology",
                            "mention": "Retrieval Augmented Generation",
                        }
                    ]
                }
            )
        if "assertions" in content:
            quote = "RAG can improve factual accuracy on knowledge-intensive tasks."
            if quote not in content:
                return json.dumps({"claims": []})
            return json.dumps(
                {
                    "claims": [
                        {
                            "statement": "RAG can improve factual accuracy",
                            "evidence_quote": quote,
                            "concept": "Retrieval Augmented Generation",
                        }
                    ]
                }
            )
        return "{}"

    return MockProvider(responder=respond)


def assessment_provider() -> MockProvider:
    """Answers as a competent local *analysis* model would.

    Reads the real prompt — the span ids and claim ids it cites are parsed out
    of the message Forge actually sent — so the grounding check downstream is
    exercised for real rather than fed pre-baked ids.
    """

    def respond(request):
        text = request.messages[-1].content
        claim_ids = re.findall(r"\[claim_id: ([^\]]+)\]", text)
        spans = re.findall(r"\[span_id: ([^\]]+)\]\s+\(([^)]*)\)\n([^\n]*)", text)
        conflict_span = next(
            (s for s in spans if "introduce errors" in s[2]), spans[0] if spans else None
        )
        if not claim_ids or conflict_span is None:
            return json.dumps({"assessments": []})
        return json.dumps(
            {
                "assessments": [
                    {
                        "claim_id": claim_ids[0],
                        "classification": "POTENTIAL_CONFLICT",
                        "rationale": (
                            "The new source reports that RAG can introduce errors when the "
                            "retrieved context is irrelevant, which the existing claim does "
                            "not qualify."
                        ),
                        "evidence_span_ids": [conflict_span[0]],
                        "refined_statement": "",
                    }
                ]
            }
        )

    return MockProvider(responder=respond)


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="forge-p4-"))
    settings = Settings.load(state_dir=workdir / "state")
    store = SqliteStore(settings.db_path)
    store.initialize()
    proposals = ProposalService(store)

    # ---------------------------------------------------------------- 1
    step("Start with existing Forge knowledge: ingest paper A")
    pipeline = IngestionPipeline(
        settings, store, extractor=CandidateExtractor(extraction_provider(), max_spans=6)
    )
    options = IngestOptions(extract=True, propose=True, max_spans=6)
    report_a = pipeline.ingest_path(PAPER_A, options)
    print(f"source   : {report_a.sources[0].locator}")
    print(f"spans    : {report_a.sources[0].spans}   proposals: {report_a.sources[0].proposals_created}")

    for proposal in proposals.list(status=ProposalStatus.PENDING, limit=20):
        proposals.approve(proposal.id, note="reviewed")
    ProposalActivator(store).activate_approved()
    claims = list(store.list_claims())
    concepts = list(store.list_concepts())
    print(f"concept  : {concepts[0].qualified_name}")
    print(f"claim    : {claims[0].statement!r}  [{claims[0].status.value}]")
    original_claim = claims[0]

    # ---------------------------------------------------------------- 2
    step("Ingest paper B — a second document that qualifies the first")
    report_b = pipeline.ingest_path(PAPER_B)  # deterministic only: no extraction
    source_b = report_b.sources[0]
    print(f"source   : {source_b.locator}")
    print(f"status   : {source_b.status.value}   spans: {source_b.spans}   llm calls: {source_b.llm_calls}")

    # ---------------------------------------------------------------- 3
    step("Evidence spans created, with page and section provenance")
    document = store.documents_for_source(source_b.source_id)[-1]
    for span in store.spans_for_document(document.id):
        print(f"  {span.citation()}")
        print(f"    {' '.join(span.text.split())[:88]}")

    # ---------------------------------------------------------------- 4
    step("Run the evolution workflow (LangGraph orchestrates; services do the work)")
    service = EvolutionService(
        store,
        settings,
        provider=assessment_provider(),
        provider_id="mock",
        model_id="mock-1",
    )
    outcome = service.start(source_b.source_id)
    run = outcome.run
    print(f"workflow : {run.id}")
    print(f"nodes    : {' -> '.join(n.node for n in run.nodes)}")

    # ---------------------------------------------------------------- 5
    step("Existing concept identified — deterministically, and it says why")
    for candidate in run.candidates:
        print(f"  {candidate.concept_name}   [{candidate.selector}]  {candidate.detail}")
    narrowing_calls = next(
        (n.llm_calls for n in run.nodes if n.node == "identify_affected_concepts"), 0
    )
    print(f"  model calls used to narrow: {narrowing_calls}")

    # ---------------------------------------------------------------- 6
    step("Existing claim retrieved")
    for claim_id in run.related_claim_ids:
        claim = store.get_claim(claim_id)
        print(f"  {claim.statement!r}  [{claim.status.value}]")

    # ---------------------------------------------------------------- 7
    step("Semantic assessment performed, and grounded in real spans")
    for assessment in run.assessments:
        print(f"  classification : {assessment.classification.value}")
        print(f"  rationale      : {assessment.rationale}")
        print(f"  provider/model : {assessment.provider_id} / {assessment.model_id}")
        print(f"  prompt/schema  : {assessment.prompt_version} / {assessment.schema_version}")
        for span_id in assessment.evidence_span_ids:
            span = store.get_span(span_id)
            print(f"  cites          : {span.citation()}  <- resolves to a real stored span")
            print(f"                   {' '.join(span.text.split())[:80]}")

    # ---------------------------------------------------------------- 8
    step("Impact classified, and a proposal generated — knowledge is NOT yet changed")
    print(f"impact   : {run.impact.value}")
    for proposal_id in run.proposal_ids:
        proposal = store.get_proposal(proposal_id)
        print(f"  {proposal.id[:12]}  {proposal.type.value}  [{proposal.status.value}]  safety={proposal.safety.value}")
        print(f"    target : {store.get_claim(proposal.operation.target).statement!r}")
        print(f"    reason : {proposal.reason[:96]}")
    print(f"claim still: {store.get_claim(original_claim.id).status.value}  (unchanged)")

    # ---------------------------------------------------------------- 9
    step("Workflow paused for human review, and checkpointed")
    print(f"status   : {run.status.value}")
    print(f"awaiting : {outcome.interrupt_payload['reason']}")
    print(f"decide by: {outcome.interrupt_payload['how_to_decide']}")
    checkpoint = settings.state_dir / "checkpoints.db"
    print(f"checkpoint on disk: {checkpoint.name}  ({checkpoint.stat().st_size} bytes)")

    # ---------------------------------------------------------------- 10
    step("Process exits — every workflow object is destroyed")
    service.close()
    del service, outcome, run
    store.close()
    del store
    print("closed the store, the checkpointer connection, and the service.")
    print("nothing about this workflow now exists in memory.")

    # ---------------------------------------------------------------- 11
    step("Human approves the proposal")
    store = SqliteStore(settings.db_path)
    store.initialize()
    proposals = ProposalService(store)
    reloaded = store.list_workflows(status=WorkflowStatus.WAITING_FOR_REVIEW)[0]
    print(f"found paused workflow from disk: {reloaded.id[:12]}")
    for proposal_id in reloaded.proposal_ids:
        decided = proposals.approve(proposal_id, note="reviewed: the qualification is fair")
        print(f"approved {decided.id[:12]} -> {decided.status.value} by {decided.decided_by}")

    # ---------------------------------------------------------------- 12
    step("Workflow resumes from the checkpoint — it does not restart")
    service = EvolutionService(
        store,
        settings,
        provider=assessment_provider(),
        provider_id="mock",
        model_id="mock-1",
    )
    resumed = service.resume(reloaded.id)
    print(f"status   : {resumed.status.value}")
    print(f"nodes    : {' -> '.join(n.node for n in resumed.run.nodes)}")
    resume_calls = sum(
        n.llm_calls for n in resumed.run.nodes if n.node in ("activate_changes", "record_revision")
    )
    print(f"llm calls during resume : {resume_calls}  <- the assessment was not repeated")
    print(f"llm calls for the whole run: {resumed.run.llm_calls}")

    # ---------------------------------------------------------------- 13
    step("Knowledge revision created")
    changed = store.get_claim(original_claim.id)
    print(f"claim    : {changed.statement!r}")
    print(f"status   : {original_claim.status.value} -> {changed.status.value}")
    print("the statement itself is unchanged: Forge flags doubt, it does not rewrite.")
    for revision in store.revisions_for(EntityType.CLAIM, changed.id):
        print(f"  [{revision.op.value}] {revision.note or ''}")

    # ---------------------------------------------------------------- 14
    step("New evidence linked to the existing claim")
    graph = KnowledgeGraph(store)
    for evidence in graph.get_claim_evidence(changed.id):
        print(f"  -[{evidence['relation']}]-> {evidence['citation']}")
        print(f"       {' '.join((evidence['text'] or '').split())[:80]}")

    # ---------------------------------------------------------------- 15
    step("Original evidence preserved — nothing was overwritten")
    print("both the paper-A quote and the paper-B qualification are attached:")
    sources = {e["source_id"] for e in graph.get_claim_evidence(changed.id)}
    print(f"  distinct sources evidencing this claim: {len(sources)}")
    print(f"  revisions in the log: {store.count_revisions()} (append-only)")

    # ---------------------------------------------------------------- 16
    step("Provenance preserved end to end")
    proposal = store.get_proposal(reloaded.proposal_ids[0])
    print(f"proposal provenance : tier={proposal.provenance.tier.value} "
          f"derivation={proposal.provenance.derivation.value}")
    print(f"                      model={proposal.provenance.model_id}")
    print(f"                      prompt={proposal.provenance.prompt_version} "
          f"schema={proposal.provenance.schema_version}")
    print(f"                      derivation_key={(proposal.provenance.derivation_key or '')[:16]}")
    link = [
        e for e in store.evidence_for_claim(changed.id) if e.relation.value == "infers_from"
    ][0]
    print(f"evidence link       : relation={link.relation.value} "
          f"tier={link.provenance.tier.value}")
    print("                      (INFERS_FROM, never QUOTES — a model may not assert a verbatim quote)")

    # ---------------------------------------------------------------- 17
    step("The workflow can be inspected afterwards")
    detail = service.explain(reloaded.id)
    print(f"workflow : {detail['id'][:12]}  [{detail['status']}]")
    print(f"impact   : {detail['impact']}")
    print(f"counts   : {json.dumps(detail['counts'])}")
    print(f"cost     : {detail['llm_calls']} llm call(s), {detail['cache_hits']} cache hit(s), "
          f"{detail['duration_ms']}ms")
    print("\nwhy Forge proposed this:")
    for candidate in detail["candidates_detail"]:
        print(f"  considered {candidate['concept']} because {candidate['detail']}")
    for assessment in detail["assessments_detail"]:
        print(f"  assessed {assessment['classification']} — {assessment['rationale'][:70]}")

    # ---------------------------------------------------------------- 18
    step("Re-running is safe — no duplicate knowledge, no wasted model calls")
    before = (
        store.count_revisions(),
        len(store.list_claims()),
        store.counts()["evidence_links"],
        store.counts()["proposals"],
    )
    again = service.start(reloaded.source_id)
    after = (
        store.count_revisions(),
        len(store.list_claims()),
        store.counts()["evidence_links"],
        store.counts()["proposals"],
    )
    print(f"status        : {again.status.value}")
    print(f"llm calls     : {again.run.llm_calls}   cache hits: {again.run.cache_hits}")
    print(f"revisions     : {before[0]} -> {after[0]}")
    print(f"claims        : {before[1]} -> {after[1]}")
    print(f"evidence links: {before[2]} -> {after[2]}")
    print(f"proposals     : {before[3]} -> {after[3]}")
    print(f"IDENTICAL     : {before == after}")

    print(f"\n{'=' * 76}")
    print("Forge did not merely store the new paper.")
    print("Forge evaluated how the new evidence affected existing knowledge,")
    print("proposed a change, waited for human approval, and recorded the revision.")
    print(f"{'=' * 76}")
    print(f"\ndemo state (safe to delete): {workdir}")

    service.close()
    store.close()
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
