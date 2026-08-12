#!/usr/bin/env python3
"""Phase 3 demonstration — Forge learns something.

    python3 scripts/phase3_demo.py

Runs the full arc: ingest a paper, extract candidates, approve them, activate
them into canonical knowledge, walk the evidence back to the page it came from,
build the graph, retrieve, evaluate, re-ingest, and hit an ambiguous concept
that Forge refuses to guess at.

**No local model is reachable in this environment** (the sandbox network policy
blocks ollama.com and huggingface.co), so extraction runs against a *scripted*
provider through the real `LLMProvider` interface. Everything else — pipeline,
grounding check, matcher, activation, graph, retrieval, evaluation — is the
production path, unchanged. On a machine with Ollama the same arc runs with
`forge ingest paper.pdf --extract`.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.activation import ProposalActivator, RelationshipActivator  # noqa: E402
from forge.config import Settings  # noqa: E402
from forge.domain import ProposalStatus, ProposalType  # noqa: E402
from forge.embeddings import HashingEmbeddingProvider  # noqa: E402
from forge.evaluation import DEFAULT_DATASET, EvalDataset, RetrievalEvaluator  # noqa: E402
from forge.extraction import CandidateExtractor  # noqa: E402
from forge.graph import KnowledgeGraph, check_integrity  # noqa: E402
from forge.identity import IdentityConfig, IdentityService  # noqa: E402
from forge.ingestion import IngestionPipeline, IngestOptions  # noqa: E402
from forge.llm import MockProvider  # noqa: E402
from forge.matching import build_ambiguity_index  # noqa: E402
from forge.proposals import ProposalService  # noqa: E402
from forge.retrieval import SearchQuery, SearchService  # noqa: E402
from forge.storage import SqliteStore  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pdf"
STEP = 0


def step(text: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 76}\n{STEP:>2}. {text}\n{'=' * 76}")


def scripted_provider() -> MockProvider:
    """Answers as a competent local extraction model would."""

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
        if "heap maintains" in content:
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
            {"concepts": [{"name": "Retrieval Augmented Generation", "kind": "technology", "mention": "RAG"}]}
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
    workdir = Path(tempfile.mkdtemp(prefix="forge-p3-"))
    settings = Settings.load(state_dir=workdir / "state")
    store = SqliteStore(settings.db_path)
    store.initialize()

    identity = IdentityService(IdentityConfig())
    pipeline = IngestionPipeline(
        settings, store, extractor=CandidateExtractor(scripted_provider(), max_spans=6)
    )
    options = IngestOptions(extract=True, propose=True, max_spans=6)
    proposals = ProposalService(store)
    activator = ProposalActivator(store, identity=identity)
    graph = KnowledgeGraph(store)

    # ---------------------------------------------------------------------- 1
    step("Start with the existing Forge corpus")
    started = time.time()
    corpus = pipeline.ingest_path(settings.vault_path)
    ingested = [s for s in corpus.sources if s.ok]
    print(f"ingested {len(ingested)} existing vault documents in {time.time() - started:.1f}s")
    print("(the whole corpus — retrieval and co-occurrence are only meaningful at real scale)")
    print(f"store: {json.dumps({k: v for k, v in store.counts().items() if v})}")

    # ---------------------------------------------------------------------- 2
    step("Ingest a new research PDF")
    report = pipeline.ingest_path(FIXTURES / "multipage.pdf", options)
    source = report.sources[0]
    print(f"source   : {source.locator}")
    print(f"status   : {source.status.value}  ({source.pages} pages, {source.spans} spans)")
    print(f"LLM calls: {source.llm_calls} (scripted local provider — no paid API)")

    # ---------------------------------------------------------------------- 3
    step("Extract candidate concepts and claims")
    print(f"{source.concepts_proposed} concept candidate(s), {source.claims_proposed} claim candidate(s)")
    print(f"{source.proposals_created} proposal(s) created — nothing is canonical yet")

    # ---------------------------------------------------------------------- 4
    step("Show the evidence spans behind them")
    search = SearchService(store)
    for proposal in proposals.list(type=ProposalType.NEW_CLAIM, source_id=source.source_id, limit=3):
        hit = search.span(proposal.evidence_span_ids[0])
        print(f"  claim : {proposal.operation.after}")
        print(f"  quote : {proposal.operation.details['evidence_quote'][:70]!r}")
        print(f"  span  : {hit.citation}\n")

    # ---------------------------------------------------------------------- 5
    step("Show the proposals awaiting a decision")
    pending = proposals.list(status=ProposalStatus.PENDING, source_id=source.source_id, limit=10)
    for proposal in pending:
        print(f"  {proposal.id[:12]}  {proposal.type.value:<14} {proposal.safety.value:<18} {proposal.operation.target[:44]}")

    # ---------------------------------------------------------------------- 6
    step("Approve a concept")
    concept_proposal = next(p for p in pending if p.type is ProposalType.NEW_CONCEPT)
    proposals.approve(concept_proposal.id, note="reviewed: correct concept")
    print(f"approved {concept_proposal.id[:12]} -> {concept_proposal.operation.target}")
    print("approval alone creates nothing — activation is a separate step")

    # ---------------------------------------------------------------------- 7
    step("Approve a claim")
    claim_proposal = next(p for p in pending if p.type is ProposalType.NEW_CLAIM)
    proposals.approve(claim_proposal.id, note="reviewed: quote checks out")
    print(f"approved {claim_proposal.id[:12]} -> {claim_proposal.operation.after}")
    print(f"the reviewer verified the quote: "
          f"{claim_proposal.operation.details['evidence_quote'][:60]!r}")

    # ---------------------------------------------------------------------- 8
    step("Show the canonical Concept and Claim (activation)")
    activation = activator.activate_approved()
    for result in activation.results:
        print(f"  [{result.outcome.value}] {result.reason}")
    print(f"\ncanonical concepts: {[c.qualified_name for c in store.list_concepts()]}")
    print(f"canonical claims  : {len(store.list_claims())}")

    # ---------------------------------------------------------------------- 9
    step("Navigate: Claim -> Evidence -> PDF -> Page -> Span")
    claim = store.list_claims()[0]
    print(f"claim      : {claim.statement}")
    print(f"provenance : {claim.provenance.tier.value} via {claim.provenance.derivation.value} "
          f"({claim.provenance.model_id})")
    print(f"origin     : proposal {claim.origin_proposal_id[:12]}")
    for evidence in graph.get_claim_evidence(claim.id):
        print(f"  -[{evidence['relation']}]-> {evidence['citation']}")
        print(f"       page    : {evidence['page']}")
        print(f"       section : {' > '.join(evidence['heading_path'])}")
        print(f"       text    : {evidence['text'][:90]}")

    # ---------------------------------------------------------------------- 10
    step("Build the relationship graph (SQLite, no graph database)")
    # Activate everything else so there are concepts to relate.
    for proposal in proposals.list(status=ProposalStatus.PENDING, limit=50):
        if proposal.type in (ProposalType.NEW_CONCEPT, ProposalType.NEW_CLAIM):
            proposals.approve(proposal.id, note="demo batch")
    activator.activate_approved()

    relationships = RelationshipActivator(store)
    candidates = relationships.discover_cooccurrence()
    result = relationships.activate(candidates)
    names = {c.id: c.qualified_name for c in store.list_concepts()}
    print(f"considered {result.candidates_considered}, created {result.created}, rejected {len(result.rejected)}")
    for rejection in result.rejected[:3]:
        print(f"  rejected: {rejection['reason'][:88]}")
    print("\nedges:")
    for link in store.all_links():
        print(f"  {names.get(link.from_id, link.from_id)} -[{link.type.value}]- "
              f"{names.get(link.to_id, link.to_id)}   ({link.rationale})")
    print(f"\nmetrics: {json.dumps(graph.metrics().to_dict())}")

    integrity = check_integrity(store)
    print(f"integrity: {'clean' if integrity.clean else integrity.by_code()}")

    # ---------------------------------------------------------------------- 11
    step("Search for a concept")
    query = "chunk size retrieval quality"
    hits = search.search(SearchQuery(text=query, limit=3))
    print(f"query  : {query!r}")
    print(f"matches: {len(hits)} span(s), lexical (FTS5/BM25), no embeddings involved")

    # ---------------------------------------------------------------------- 12
    step("Show retrieval results, each with its provenance")
    for hit in hits:
        print(f"  {hit.score:7.3f}  {hit.citation}")
        print(f"           {' '.join(hit.span.text.split())[:88]}")

    # ---------------------------------------------------------------------- 13
    step("Run retrieval evaluation (measured, not asserted)")
    dataset = EvalDataset.load(settings.vault_path / DEFAULT_DATASET)
    print(f"dataset: {dataset.path} v{dataset.version} — {len(dataset)} queries, {dataset.label_count()} labels")
    evaluation = RetrievalEvaluator(store, embeddings=HashingEmbeddingProvider()).run(
        dataset, methods=("lexical",)
    )
    for summary in evaluation.summaries:
        print("  " + summary.headline())
    print("\n(full lexical-vs-semantic-vs-hybrid comparison: docs/research/retrieval-baseline.md)")

    # ---------------------------------------------------------------------- 14
    step("Add the same PDF again")
    before = store.counts()
    repeat = pipeline.ingest_path(FIXTURES / "multipage.pdf", options)
    after = store.counts()
    print(f"source : {repeat.sources[0].locator}")
    print(f"status : {repeat.sources[0].status.value}  (content hash unchanged)")

    # ---------------------------------------------------------------------- 15
    step("Zero duplicate knowledge, zero unnecessary LLM work")
    print(f"new LLM calls       : {repeat.llm_calls}")
    print(f"duplicate documents : {after['documents'] - before['documents']}")
    print(f"duplicate spans     : {after['spans'] - before['spans']}")

    reactivation = activator.activate_approved()
    print(f"re-activation       : {reactivation.counts() or 'nothing pending'}")
    print(f"duplicate concepts  : {after['concepts'] - before['concepts']}")
    print(f"duplicate claims    : {after['claims'] - before['claims']}")

    # ---------------------------------------------------------------------- 16
    step("Demonstrate an ambiguous concept")
    identity.scaffold(build_ambiguity_index([s.locator for s in store.list_sources()] +
                                            _vault_paths(settings)))
    overlap = pipeline.ingest_path(FIXTURES / "overlapping.pdf", options)
    ambiguous = [
        p
        for p in proposals.list(source_id=overlap.sources[0].source_id, limit=50)
        if p.operation.details.get("match_kind") == "ambiguous"
    ]
    for proposal in ambiguous:
        print(f"concept    : {proposal.operation.target}")
        print(f"safety     : {proposal.safety.value}")
        print(f"selection  : {proposal.operation.after!r}   <- none made")
        print(f"reason     : {proposal.reason[:96]}")
        print("candidates :")
        for candidate in proposal.operation.details["candidates"]:
            print(f"   - {candidate.get('qualified_name') or candidate['canonical_name']}"
                  f"   {candidate.get('vault_path') or ''}")

    # ---------------------------------------------------------------------- 17
    step("Forge refuses to guess")
    for proposal in ambiguous:
        approved = proposals.approve(proposal.id, note="approved, but which Heap?")
        outcome = activator.activate(approved)
        print(f"activating it anyway -> {outcome.outcome.value}")
        print(f"  {outcome.reason[:150]}")

    print("\nresolving the collision explicitly:")
    identity.decide("Heap", "data-structure/Heap", by="demo")
    print("  forge identity decide Heap data-structure/Heap")
    resolved = ProposalActivator(store, identity=identity).activate(
        store.get_proposal(ambiguous[0].id)
    ) if ambiguous else None
    if resolved:
        print(f"  now activates -> {resolved.outcome.value}: {resolved.reason}")
        concept = store.get_concept(resolved.entity_id)
        if concept:
            print(f"  canonical concept: {concept.qualified_name}  (namespace preserved)")

    # ---------------------------------------------------------------------- summary
    step("Summary — Forge learned something")
    counts = {k: v for k, v in store.counts().items() if v}
    print(f"store            : {json.dumps(counts)}")
    print(f"graph            : {json.dumps(graph.metrics().to_dict())}")
    print(f"integrity        : {'clean' if check_integrity(store).clean else 'findings present'}")
    print(f"\ndemo state (safe to delete): {workdir}")

    store.close()
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


def _vault_paths(settings: Settings) -> list[str]:
    from forge.corpus.indexer import CorpusIndexer

    return CorpusIndexer(settings).discover()


if __name__ == "__main__":
    raise SystemExit(main())
