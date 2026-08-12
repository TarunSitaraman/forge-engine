# Forge — Target Architecture

*The engine that maintains the knowledge model: components, data flow, workflow orchestration, and the boundaries that keep the design honest.*

**Status:** proposed · **Depends on:** [current-state audit](./forge-current-state.md), [canonical model](../knowledge-model/canonical-model.md)

---

## 1. The load-bearing decision

> **Markdown is the source of truth. Every index is derived and
> rebuildable.**

The vector store, graph store, and relational metadata store are
**caches with structure**. Deleting all three and re-running ingestion
must reproduce them. Nothing may exist only in a database.

This single decision buys almost everything the constraints demand:

| It gives us | Because |
|---|---|
| Preservation of the existing repo (audit P1/P2) | The vault keeps working with no engine running |
| Local-first (Principle 9) | No hosted service holds anything unique |
| Escape hatch | The user's knowledge survives Forge's deletion |
| Cheap schema iteration | Change the model, rebuild the index — no migration |
| Obsidian compatibility (audit P3) | Nothing plugin-only or DB-only |

The cost is real and worth naming: **derived state must be
reconstructible from Markdown, so anything the engine learns that
Markdown cannot express is at risk on rebuild.** Contradictions,
confidence, and revision history are exactly that kind of state. This
is what makes the write-back question
([ADR-001](../decisions/001-forge-knowledge-os.md)) unavoidable rather
than a detail — and until it is answered, the engine must treat derived
stores as *durable but rebuildable-with-loss*, and back them up rather
than assume they are disposable.

---

## 2. Component architecture

```
 INTERFACES     CLI  |  Web UI  |  MCP server  |  Obsidian plugin
   (thin)       ---------------------------------------------------
                                     |
                              FastAPI  (HTTP/local)
                                     |
 ================================ FORGE CORE ======================
                                     |
   ORCHESTRATION       LangGraph  (stateful, resumable workflows)
                                     |
   +---------+-----------+-----------+-----------+---------------+
   |         |           |           |           |               |
 SOURCE   PARSING    KNOWLEDGE    RETRIEVAL   EVOLUTION      PROVENANCE
 REGISTRY  + CHUNK     ENGINE      ENGINE      ENGINE          LEDGER
   |         |           |           |           |               |
   |    deterministic  concept    hybrid      change          tiers,
   |    (no LLM)       resolve,   search,     analysis,       lineage,
   |                   relations  rerank      contradiction   revisions
   |                                          supersession
   +---------+-----------+-----------+-----------+---------------+
                                     |
                          LLM PROVIDER ABSTRACTION
                     Ollama (default) | OpenAI-compat | Anthropic
                       | Gemini | MockProvider (tests)
                                     |
 ================================ STORAGE =========================
                                     |
   MARKDOWN VAULT (truth)  ->  RELATIONAL | VECTOR | GRAPH  (derived)
        + Git history            metadata   spans    concepts,
                                 revisions  embeds   claims, edges
```

### Responsibilities

| Component | Owns | LLM? |
|---|---|---|
| **Source registry** | Identity, hashing, change detection, dedup by hash | No |
| **Parsing + chunking** | Format-specific extraction, structure-aware spans | No |
| **Knowledge engine** | Concept resolution, relationship discovery, claim extraction | Selectively |
| **Retrieval engine** | Hybrid search (lexical + vector + graph), reranking | No (rerank optional) |
| **Evolution engine** | Change analysis, contradiction detection, supersession, staleness | Selectively |
| **Provenance ledger** | Tier enforcement, lineage, revision log, floor rule | No |
| **Provider abstraction** | Model selection, retries, caching, structured output | — |

Note the shape: **four of seven core components need no LLM at all**,
and the provenance ledger — the component that guarantees Principle 10
— is entirely deterministic. That is the intended distribution.

---

## 3. Storage layout

| Store | Holds | Rebuildable | Default |
|---|---|---|---|
| **Markdown vault** | Human-authored knowledge | *is the truth* | filesystem + Git |
| **Relational** | Sources, documents, spans, revisions, run state, config | yes | SQLite → Postgres |
| **Vector** | Span + concept embeddings | yes | see [technology decisions](./technology-decisions.md) |
| **Graph** | Concepts, claims, typed edges | yes | see technology decisions |
| **Blob** | Original PDFs and fetched artifacts | **no — originals** | local dir, content-addressed |

**The blob store is the one exception to "everything is rebuildable."**
Original source bytes cannot be regenerated, so they are retained
verbatim, content-addressed by hash. This satisfies "preserve original
source data" from the brief and makes re-parsing with a better extractor
possible without re-downloading.

---

## 4. Ingestion data flow

Deterministic stages first; the LLM enters only at stage 5.

```
 SOURCE  ->  IDENTIFY  ->  [hash unchanged? -> STOP, 0 tokens]
               |  deterministic: hash, metadata, kind
               v
             PARSE      deterministic: format-specific extractor
               |
               v
             CHUNK      deterministic: structure-aware spans
               |
               v
             EMBED      local model, batched, cached by span hash
               |
               v
        CANDIDATE MATCH deterministic: alias + lexical + vector
               |        (narrows the whole graph to ~10 candidates)
               v
      == LLM BOUNDARY ==
               |
        EXTRACT CLAIMS  model, span-bound, schema-validated
               |
        RESOLVE CONCEPTS  model, only for candidates that don't match exactly
               |
        DISCOVER RELATIONSHIPS  model, over resolved concepts
               |
        ANALYZE CHANGE   model: known / supports / contradicts / refines / new
               |
      == LLM BOUNDARY ==
               |
        VALIDATE        deterministic: provenance floor, schema, invariants
               |
        COMMIT          transactional write + revision log
               |
        MARK STALE      deterministic traversal -> affected syntheses
```

### Cost control (Principle 8)

Five mechanisms, in order of impact:

1. **Hash short-circuit** — unchanged source costs zero tokens.
2. **Span-level caching** — only changed spans are re-embedded and
   re-extracted; editing one section of a 300-line doc reprocesses one
   section.
3. **Deterministic candidate narrowing** — the LLM compares against ~10
   candidates, never the full concept graph. This is what makes cost
   scale with *change size*, not corpus size.
4. **Exact-match bypass** — an alias hit resolves a concept with no
   model call.
5. **Response caching** keyed on `(prompt_version, model_id, input_hash)`.

Ingesting the existing 620-file vault a second time with no edits must
cost **zero LLM calls**. That is the benchmark.

---

## 5. LangGraph: where it is used, and where it is not

LangGraph earns its place where there is genuine **stateful, resumable,
conditionally-routed** work. Using it elsewhere adds indirection with no
benefit.

### 5.1 Uses LangGraph

**W1 — Ingestion & Evolution** *(the primary workflow, Phase 5)*

Typed state carried through every node:

```python
class IngestionState(TypedDict):
    source_id: str
    document: Document | None
    spans: list[Span]
    candidate_concepts: list[ConceptMatch]
    extracted_claims: list[Claim]
    resolved: list[ConceptResolution]
    relationships: list[ClaimLink]
    change_analysis: ChangeAnalysis | None
    contradictions: list[Contradiction]
    pending_approval: list[ApprovalRequest]
    errors: list[NodeError]
    run_id: str
```

Nodes — each with one responsibility, named for what it does:

`SourceIdentificationNode` · `ParseNode` · `ChunkNode` · `EmbedNode` ·
`CandidateMatchNode` · `ClaimExtractionNode` · `ConceptResolutionNode` ·
`RelationshipDiscoveryNode` · `ChangeAnalysisNode` ·
`ContradictionAnalysisNode` · `ApprovalGateNode` ·
`KnowledgeUpdateNode` · `SynthesisStalenessNode`

Conditional routing:

- unchanged hash → terminate immediately
- no claims extracted → skip to commit (record the source, assert
  nothing)
- contradiction with severity above threshold → `ApprovalGateNode`
- concept merge above similarity threshold → `ApprovalGateNode`
- node failure → retry with backoff → quarantine, never partial-commit

Why LangGraph specifically: **checkpointing** (a 500-page PDF must
resume, not restart), **human-in-the-loop interrupts** (contradiction
and merge approval — Principle 11's enforcement point), **typed state**,
and **replayability** for debugging.

**W2 — Contradiction Resolution** *(Phase 5+)* — inherently long-lived
and human-gated; may wait days for a decision. Checkpointing is the
whole point.

**W3 — Re-synthesis** *(Phase 6+)* — batch, resumable, partially
failing; regenerates syntheses marked stale.

**W4 — Corpus Backfill** *(Phase 2)* — 620 files, resumable, with
per-file failure isolation.

### 5.2 Does **not** use LangGraph

Parsing · chunking · hashing · embedding · vector search · graph queries
· the HTTP API · retrieval · CRUD.

These are ordinary functions. Wrapping them in a graph would add state
machinery to code that has no state, and would make them harder to test.

**Retrieval deliberately stays a plain composed pipeline** until there
is a real need for conditional multi-hop reasoning. Making retrieval a
graph on day one is the single most common way this kind of system
acquires ceremony without capability — and would push Forge toward the
"generic RAG chatbot" boundary.

### 5.3 Agent naming discipline

Nodes are named for their operation (`ContradictionAnalysisNode`), never
for a persona (`ResearchAgent`, `SuperSmartAgent`). If a node's
responsibility cannot be stated in one sentence, it is two nodes.

---

## 6. Retrieval architecture

Hybrid by default, because no single method is sufficient and three
cheap deterministic methods beat one expensive semantic one:

```
 QUERY
   |
   +-- lexical (BM25/FTS)      exact terms, names, code identifiers
   +-- vector (local embeds)   paraphrase, semantic similarity
   +-- graph traversal         "related to concept X", n-hop
   +-- metadata filter         kind, tier, confidence, date, source
   |
   v
 FUSION (reciprocal rank fusion — deterministic)
   |
   v
 OPTIONAL RERANK (local cross-encoder; off by default)
   |
   v
 RESULTS  — every hit carries its provenance tier
```

Two requirements that follow from the principles:

- **Every result is attributed.** A retrieval hit without a resolvable
  span is a bug, not a degraded result.
- **Retrieval must work with the LLM entirely disabled.** Search is not
  a semantic-reasoning task (Principle 7). Generation over retrieval is
  a separate, optional layer.

---

## 7. Provider abstraction

```python
class LLMProvider(Protocol):
    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...
    async def structured(
        self, req: CompletionRequest, schema: type[T]
    ) -> T: ...
    @property
    def capabilities(self) -> ProviderCapabilities: ...

class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[Vector]: ...
    @property
    def dimensions(self) -> int: ...
```

Rules:

- **Ollama is the default and the CI target.** No paid provider is
  required for any core path (Principle 9).
- **`MockProvider` returns deterministic fixtures**, so the full
  pipeline is testable offline with no model.
- **Model names never appear in business logic** — only in
  configuration, referenced by *role* (`extraction`, `analysis`,
  `embedding`), so a node asks for the extraction model, not for a
  specific checkpoint.
- **No API keys in code.** Environment/config only, validated at
  startup.
- **Structured output is schema-validated with bounded repair retries**,
  then a hard failure. A malformed model response must never become a
  silently-degraded write.
- **Capability negotiation:** providers differ in structured-output
  support and context length. Nodes declare requirements; the
  abstraction selects or fails loudly at startup rather than mid-run.

---

## 8. Proposed module layout

Assumes ADR-001 option (a) — engine alongside the vault. See audit §8/D1.

```
engine/
  forge/
    config/        typed settings, validation, provider registry
    models/        canonical model (pydantic + persistence schema)
    sources/       registry, hashing, change detection
    parsers/       markdown | pdf | repo | web   (deterministic)
    chunking/      structure-aware span construction
    embeddings/    provider-backed, cached
    stores/        relational | vector | graph | blob  (behind protocols)
    knowledge/     concept resolution, relationship discovery
    evolution/     change analysis, contradiction, supersession, staleness
    provenance/    tiers, floor rule, lineage, revision log
    retrieval/     hybrid search + fusion
    workflows/     LangGraph graphs, nodes, typed state
    llm/           provider abstraction + mock
    api/           FastAPI
    cli/
    mcp/
    observability/ structured logging, tracing, metrics
  tests/
    unit/  integration/  fixtures/  eval/
  migrations/
```

**Stores sit behind protocols** so the Phase-3 choice (SQLite + a local
vector index) can become the Phase-8 choice (Postgres + Qdrant + Neo4j)
without touching workflow code. Given the open technology questions in
[technology-decisions.md](./technology-decisions.md), that indirection is
justified rather than speculative.

---

## 9. Engineering quality requirements

| Area | Requirement |
|---|---|
| Typing | Full annotations; strict type-checking in CI |
| Unit tests | Every deterministic component; parsers/chunkers against golden fixtures |
| Integration tests | Full ingestion against `MockProvider`, offline, no network |
| Eval tests | Concept extraction and contradiction detection against a labeled set drawn from the existing corpus |
| Determinism | Same input + same fixtures → byte-identical model output |
| Logging | Structured, `run_id`-correlated, every LLM call logged with prompt version and token count |
| Tracing | Span per node; the ingestion graph must be inspectable end to end |
| Config | Validated at startup; fail fast on a missing provider or bad model role |
| Errors | Typed; node failures quarantine a source rather than corrupting the model |
| Migrations | Versioned for relational; derived stores rebuild instead |

**The corpus is the evaluation asset.** 620 files of curated
engineering knowledge, with a known concept vocabulary (32 patterns, 30
algorithms, 18 data structures, 11 technologies), is a far better test
set than anything synthetic — and it comes with ground-truth canonical
names already assigned.

---

## 10. Deployment

Docker Compose, local-first, no cloud dependency:

```
forge-api      FastAPI
forge-worker   LangGraph execution
ollama         local inference        (or host-native)
vector-store   \
graph-store     >  per technology-decisions.md
postgres       /
```

Phase-3 development should run with **no containers at all** — SQLite,
a local vector index, and host Ollama. Compose arrives when the store
choices are settled (Phase 4+), not before. Requiring a five-service
Compose stack to iterate on a chunking algorithm is friction with no
payoff.

---

## 11. Known risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Local models underperform at claim extraction and contradiction detection** — the two hardest semantic tasks, on the weakest models | Spike early (Phase 1/3). Design so quality degrades gracefully: fewer claims, not wrong ones. Provider abstraction allows opt-in stronger models |
| R2 | **Contradiction false positives** erode trust faster than misses | Conservative thresholds; contradictions are proposals for review, not silent model edits |
| R3 | **Concept fragmentation or over-merging** | Deterministic alias matching first; human approval above the merge threshold |
| R4 | **`RELATED_TO` noise** turns the graph into a mesh | Similarity threshold, score retained, excluded from reasoning traversals |
| R5 | **Derived state that Markdown can't express is lost on rebuild** | The core tension behind ADR-001; back up derived stores; do not treat as disposable until resolved |
| R6 | **Ingestion cost grows with corpus rather than change** | Hash short-circuit + span caching; enforce the "zero-token re-ingest" benchmark as a test |
| R7 | **Scope drift into RAG-chatbot shape** | Boundary checklist at every phase gate |

---

## Related

- [Current-state audit](./forge-current-state.md)
- [Technology decisions](./technology-decisions.md)
- [Canonical knowledge model](../knowledge-model/canonical-model.md)
- [ADR-001](../decisions/001-forge-knowledge-os.md)
- [Roadmap](../roadmap.md)
