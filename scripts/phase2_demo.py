#!/usr/bin/env python3
"""Phase 2 demonstration — all five required demos, end to end.

    python3 scripts/phase2_demo.py

Demos 1, 2 and 5 need no model at all and exercise exactly what the CLI does.

Demos 3 and 4 require semantic extraction. **No local model is reachable in
this environment** (the sandbox network policy blocks ollama.com), so they run
against a *scripted* provider that returns fixed, realistic responses through
the real `LLMProvider` interface. Every other component — pipeline, grounding
check, matcher, proposal service, storage — is the production path, unchanged.

On a machine with Ollama the identical path runs with:

    forge ingest paper.pdf --extract
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.config import Settings  # noqa: E402
from forge.corpus.indexer import CorpusIndexer  # noqa: E402
from forge.extraction import CandidateExtractor  # noqa: E402
from forge.ingestion import IngestionPipeline, IngestOptions  # noqa: E402
from forge.llm import MockProvider  # noqa: E402
from forge.proposals import ProposalService, build_repair_proposals  # noqa: E402
from forge.retrieval import SearchQuery, SearchService  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pdf"


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def scripted_provider() -> MockProvider:
    """A provider that answers as a competent local model would.

    Responses vary by the text they are shown, so the two PDFs produce
    overlapping-but-different concepts — which is what Demo 3 needs to be a
    real test rather than a tautology.
    """

    def respond(request):
        content = request.messages[1].content
        wants_concepts = "concepts" in content

        if "Chunking Strategy" in content:
            return json.dumps(
                {"concepts": [{"name": "Chunking Strategy", "kind": "concept", "mention": "Chunk size"}]}
                if wants_concepts
                else {
                    "claims": [
                        {
                            "statement": "Chunk size materially affects retrieval quality",
                            "evidence_quote": "Chunk size materially affects retrieval quality.",
                            "concept": "Chunking Strategy",
                        }
                    ]
                }
            )
        if "Hybrid Search" in content:
            return json.dumps(
                {"concepts": [{"name": "Hybrid Search", "kind": "concept", "mention": "Hybrid Search"}]}
                if wants_concepts
                else {
                    "claims": [
                        {
                            "statement": "Hybrid search outperforms dense-only retrieval",
                            "evidence_quote": "Combining BM25 with dense vectors outperforms either alone.",
                            "concept": "Hybrid Search",
                        }
                    ]
                }
            )
        if "RAG systems depend" in content:
            return json.dumps(
                {
                    "concepts": [
                        {"name": "Retrieval Augmented Generation", "kind": "technology", "mention": "RAG systems"},
                        {"name": "Hybrid Search", "kind": "concept", "mention": "Hybrid Search"},
                    ]
                }
                if wants_concepts
                else {
                    "claims": [
                        {
                            "statement": "RAG systems depend heavily on the retrieval step",
                            "evidence_quote": "RAG systems depend heavily on the retrieval step.",
                            "concept": "Retrieval Augmented Generation",
                        }
                    ]
                }
            )
        if "Heap" in content:
            return json.dumps(
                {"concepts": [{"name": "Heap", "kind": "data_structure", "mention": "A heap"}]}
                if wants_concepts
                else {
                    "claims": [
                        {
                            "statement": "A heap keeps its smallest element at the root",
                            "evidence_quote": "A heap maintains the smallest element at its root.",
                            "concept": "Heap",
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "concepts": [
                    {"name": "Retrieval Augmented Generation", "kind": "technology", "mention": "RAG"}
                ]
            }
            if wants_concepts
            else {
                "claims": [
                    {
                        "statement": "RAG grounds generation in retrieved passages",
                        "evidence_quote": "RAG grounds generation in retrieved passages.",
                        "concept": "Retrieval Augmented Generation",
                    }
                ]
            }
        )

    return MockProvider(responder=respond)


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="forge-demo-"))
    settings = Settings.load(state_dir=workdir / "state")
    store = SqliteStore(settings.db_path)
    store.initialize()

    extractor = CandidateExtractor(scripted_provider(), max_spans=6)
    pipeline = IngestionPipeline(settings, store, extractor=extractor)
    options = IngestOptions(extract=True, propose=True, max_spans=6)

    # ---------------------------------------------------------------- 1
    banner("DEMO 1 — ingest a PDF (no paid API)")
    report = pipeline.ingest_path(FIXTURES / "multipage.pdf", options)
    source = report.sources[0]
    print(f"Source registered   : {source.source_id}")
    print(f"Document created    : {source.document_id}")
    print(f"{source.spans} spans extracted  ({source.pages} pages, {source.chars} chars)")
    print(f"{source.concepts_proposed} concepts proposed")
    print(f"{source.claims_proposed} claims proposed")
    print(f"{source.proposals_created} proposals created")
    print(f"LLM calls           : {source.llm_calls}  (local/scripted — no paid API)")

    # ---------------------------------------------------------------- 2
    banner("DEMO 2 — ingest the same PDF again")
    before = store.counts()
    again = pipeline.ingest_path(FIXTURES / "multipage.pdf", options)
    after = store.counts()
    repeat = again.sources[0]
    print(f"Status              : {repeat.status.value}")
    print(f"New LLM extraction calls : {again.llm_calls}")
    print(f"Duplicate documents      : {after['documents'] - before['documents']}")
    print(f"Duplicate spans          : {after['spans'] - before['spans']}")

    # ---------------------------------------------------------------- 3
    banner("DEMO 3 — a second, overlapping PDF")
    overlap = pipeline.ingest_path(FIXTURES / "overlapping.pdf", options)
    second = overlap.sources[0]
    print(f"{second.spans} spans, {second.concepts_proposed} concepts, {second.claims_proposed} claims")

    service = ProposalService(store)
    from forge.domain import ProposalType

    print("\nConcept outcomes for THIS document (no silent merges):")
    for proposal in service.list(source_id=second.source_id, limit=50):
        details = proposal.operation.details
        if "match_kind" not in details:
            continue
        print(f"  {proposal.operation.target:<34} -> {details['match_kind']}")
        print(f"        {proposal.reason[:88]}")
        for candidate in details.get("candidates", [])[:3]:
            print(
                f"        candidate: {candidate['canonical_name']} "
                f"[{candidate['signal']} {candidate['score']}] {candidate.get('vault_path') or ''}"
            )

    print("\nEvidence links (claim -> span -> document -> source):")
    search = SearchService(store)
    for proposal in service.list(type=ProposalType.NEW_CLAIM, limit=3):
        span_id = proposal.evidence_span_ids[0]
        hit = search.span(span_id)
        print(f"  claim: {proposal.operation.after}")
        print(f"     -> {hit.citation}")
        print(f"     -> quote: {proposal.operation.details['evidence_quote'][:70]!r}")

    # ---------------------------------------------------------------- 4
    banner("DEMO 4 — an ambiguous concept ('Heap')")
    ambiguous = [
        p
        for p in service.list(limit=200)
        if p.operation.details.get("match_kind") == "ambiguous"
    ]
    if not ambiguous:
        print("  (no ambiguous concepts surfaced)")
    for proposal in ambiguous:
        print(f"Concept   : {proposal.operation.target}")
        print(f"Safety    : {proposal.safety.value}")
        print(f"Reason    : {proposal.reason}")
        print(f"Selection : {proposal.operation.after!r}  <- no automatic selection")
        print("Candidates:")
        for candidate in proposal.operation.details["candidates"]:
            print(f"   - {candidate.get('vault_path') or candidate['canonical_name']}")
        print(f"Proposal generated: {proposal.id[:12]} (status {proposal.status.value})")

    # ---------------------------------------------------------------- 5
    banner("DEMO 5 — metadata repair proposals: list and approve")
    index = CorpusIndexer(settings).build_index()
    created, skipped = service.create_many(build_repair_proposals(index.files))
    print(f"Repair proposals: {created} created, {skipped} already known")
    print(f"Counts by status: {service.counts()}")

    from forge.domain import ProposalStatus

    pending = service.list(status=ProposalStatus.PENDING, type=ProposalType.METADATA_REPAIR, limit=1)
    target = pending[0]
    print(f"\nApproving {target.id[:12]} — {target.operation.target}")
    print(f"  - {target.operation.before}")
    print(f"  + {target.operation.after}")

    decided = service.approve(target.id, note="deterministic and verified")
    print(f"\nStatus       : {decided.status.value}")
    print(f"Decided at   : {decided.decided_at}")

    from forge.domain import EntityType

    revisions = store.revisions_for(EntityType.PROPOSAL, decided.id)
    print(f"Revisions    : {[(r.op.value, r.note) for r in revisions]}")

    vault_file = settings.vault_path / decided.operation.target
    still_broken = decided.operation.before in vault_file.read_text(encoding="utf-8")
    print(f"\nVault file unmodified: {still_broken}")
    print(
        "Approval is RECORDED; source mutation remains DEFERRED.\n"
        "Applying requires an explicit flag:  forge proposals approve <id> --apply"
    )

    banner("SUMMARY")
    print(f"store counts: {json.dumps({k: v for k, v in store.counts().items() if v})}")
    print(f"demo state (safe to delete): {workdir}")
    store.close()
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
