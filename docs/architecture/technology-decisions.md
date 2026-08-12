# Forge — Technology Decisions

*Each candidate technology evaluated against the audited repository, with a recommendation, a deferral trigger where applicable, and an honest confidence level.*

**Status:** proposed · **Depends on:** [current-state audit](./forge-current-state.md), [target architecture](./target-architecture.md)

---

## 0. The constraint the audit imposes

The repository contains **zero application code** — no manifests, no
dependencies, no CI, no tests. There is therefore **no existing
implementation to preserve or migrate**, and the "existing repository
has priority over these defaults" rule resolves cleanly: it applies to
the *content substrate* (Markdown, Git, Obsidian, wikilinks, YAML
frontmatter), which is preserved absolutely, and not to the engine
stack, which is genuinely greenfield.

That freedom makes the main risk **over-provisioning**. The natural
failure here is standing up Postgres + Qdrant + Neo4j + Docker Compose
in week one and spending the project's early energy on infrastructure
plumbing instead of the provenance and evolution model — which is the
part that actually cannot be retrofitted.

Two rules follow:

1. **Adopt on evidence, not on the list.** Every deferred technology
   below has an explicit trigger that promotes it.
2. **Defer behind a protocol, not behind a rewrite.** Store choices sit
   behind interfaces (target architecture §8), so deferral costs an
   adapter, not an architecture change.

### Scale, measured

| Quantity | Estimate |
|---|---|
| Existing corpus | 620 files, 48,737 lines |
| Expected spans from corpus | ~8,000–15,000 (heading-level chunking) |
| Concepts bootstrappable from filenames | ~500 |
| Existing link edges | ~4,100 |
| Concurrent writers | **1** |
| Deployment target | one laptop |

This is a **small** dataset by every relevant measure. It fits
comfortably in SQLite and in memory. Sizing infrastructure for the
imagined 10,000-source future instead of the actual current one would be
the classic mistake.

---

## 1. Decisions at a glance

| Area | Decision | Confidence |
|---|---|---|
| Language | **Python 3.12+** | High |
| API | **FastAPI** | High |
| Orchestration | **LangGraph** (Phase 5, W1–W4 only) | High |
| LangChain | **Selective** — loaders/splitters only, not as a framework | Medium |
| Local inference | **Ollama** | High |
| Embeddings | **Local, via provider abstraction**; model by hardware tier | Medium |
| Relational | **SQLite now → Postgres on trigger** | High |
| Vector | **SQLite-based (sqlite-vec/LanceDB) → Qdrant on trigger** | Medium |
| Graph | **Relational adjacency → Neo4j on trigger** | **Low — see §5** |
| PDF | **pypdfium2 + pdfplumber**; PyMuPDF only if licensing accepted | Medium |
| Frontend | **CLI first**, web UI (Next.js/TS) at Phase 6 | High |
| Obsidian | **TypeScript plugin**, Phase 7 | High |
| MCP | **Python MCP server**, Phase 8 | High |
| Containers | **Docker Compose at Phase 4+**, not before | High |

---

## 2. Language and API

**Python 3.12+.** LangGraph, Ollama clients, embedding libraries, and
every document-parsing library live here. Modern typing generics and
better error messages matter for a codebase that will be heavily typed.

**FastAPI**, with Pydantic v2 doing double duty: HTTP contracts *and*
LLM structured-output schemas. One schema definition validating both the
API boundary and model responses is a genuine simplification, not a
convenience.

**Confidence: high.** No serious contender given the ecosystem.

---

## 3. Orchestration

**LangGraph — adopted, scoped to four workflows** (target architecture
§5): ingestion/evolution, contradiction resolution, re-synthesis,
corpus backfill. Justified by checkpointing, human-in-the-loop
interrupts, typed state, and replay.

**LangChain — selective use only.** Its document loaders and text
splitters are worth reusing. Its chains, agents, and retriever
abstractions are not: they hide control flow and make provenance
tracking harder, which is directly hostile to Principle 10.

> The specific hazard: LangChain retrievers return documents with
> metadata dictionaries. Forge needs claims bound to spans with
> enforced provenance tiers. Adopting the abstraction would mean
> fighting it at every boundary.

**Confidence: high** for LangGraph, **medium** for the LangChain
boundary — the line may need to move once the parsers are real.

---

## 4. LLM and embeddings

### 4.1 Ollama — adopted as default and CI target

Satisfies Principle 9 directly: no paid API on any core path. Also the
lowest-friction local runtime, with an OpenAI-compatible endpoint that
makes the provider abstraction simpler.

The abstraction (target architecture §7) supports OpenAI-compatible,
Anthropic, and Gemini providers as **opt-in**. None may become required.

### 4.2 Model roles, not model names

Configuration binds *roles* to models; business logic references roles.

| Role | Task | Difficulty |
|---|---|---|
| `extraction` | Claims from spans; structured output | Medium |
| `analysis` | Change analysis, contradiction detection | **Hard** |
| `resolution` | Concept matching among candidates | Easy–medium |
| `synthesis` | Aggregate generation | Medium |
| `embedding` | Vectors | N/A |

### 4.3 Hardware is unknown — tiers, not a pick

The repository records nothing about available hardware, so a single
recommendation would be a guess. Three tiers instead:

| Tier | Generation | Embedding |
|---|---|---|
| Modest (≤8 GB VRAM / Apple silicon 16 GB) | 7–8B instruct, quantized | `bge-small-en-v1.5` (384d) |
| Comfortable (12–24 GB) | 12–14B instruct | `bge-base-en-v1.5` / `nomic-embed-text` (768d) |
| Generous (≥32 GB) | 27–32B instruct | `bge-large` / `nomic-embed-text` |

**Embedding dimension is a schema-affecting choice** — changing it
requires re-embedding every span. Store `embedding_model` and
`dimensions` alongside vectors from day one so re-embedding is a
detectable, scriptable migration rather than silent corruption.

### 4.4 The open risk

**`analysis` is the hard role and the one most likely to disappoint on
local models.** Judging whether two claims genuinely contradict —
rather than differ in scope or vocabulary — is subtle reasoning.

Mitigation: spike this specific capability early (Phase 1/3), against
real pairs drawn from the existing corpus, before committing to
thresholds. Design so that weak performance yields *fewer detections*,
not *wrong ones* — a conservative detector that misses contradictions is
recoverable; one that manufactures them destroys trust in the model.

**Confidence: medium.** This is the largest technical unknown in the
project, and no amount of architecture removes it.

---

## 5. Storage

### 5.1 Relational — SQLite first

**Recommended: SQLite for Phases 1–5; Postgres when triggered.**

One writer, one laptop, ~15k spans. SQLite gives zero-setup local-first
operation, a single-file database that is trivially backed up and
inspected, FTS5 for the lexical arm of hybrid retrieval, and no service
to run while iterating.

**Promote to Postgres when:** concurrent writers appear, the API becomes
multi-user, or `pgvector` consolidation becomes attractive.

*Both are SQL behind the same protocol; the migration is real but
bounded.*

### 5.2 Vector — SQLite-based first, Qdrant on trigger

**Recommended: `sqlite-vec` or LanceDB for Phases 3–5.**

At ~15k vectors, brute-force cosine similarity takes milliseconds. An
HNSW index solves a problem this dataset does not have, and Qdrant adds
a service, a client, and a second consistency boundary to keep in sync
with the relational store.

**Promote to Qdrant when:** vectors exceed ~500k, filtered search
latency becomes a real complaint, or multi-tenancy arrives.

**Confidence: medium.** The trigger is clear, and the protocol boundary
keeps the switch cheap.

### 5.3 Graph — the genuinely contested decision

**Recommended for Phase 4: relational adjacency tables with recursive
CTEs, behind a `GraphStore` protocol. Re-evaluate Neo4j at the Phase-4
exit gate.**

**Confidence: low.** This is the weakest recommendation in the document
and is flagged for human decision.

Honest case for each:

| Relational adjacency | Neo4j |
|---|---|
| No extra service; local-first stays trivial | Cypher is dramatically better for multi-hop and path queries |
| One consistency boundary, one backup, one transaction | Purpose-built traversal performance |
| ~4k edges is nothing; CTEs are fast at this size | Native graph algorithms (centrality, community detection) — directly relevant to gap detection |
| Reified edges (`ClaimLink`, `EvidenceLink`) are *tables*, which is the natural relational shape | Idiomatic for the property-graph model the canonical model describes |

The real argument for Neo4j is not scale — it is **query expressiveness**.
Phase-9 questions ("which concepts connect this unresolved question to
evidence I already have?") are natural in Cypher and painful in
recursive SQL. The real argument against is that adding a JVM service to
a single-user local-first tool contradicts the deployment story, and
that a second store means two things to keep consistent.

**Deciding factor, to be measured rather than argued:** if Phase-4
queries routinely exceed 3 hops or need path-finding, adopt Neo4j. If
they are mostly 1–2 hop neighborhood lookups, relational wins on
simplicity. **That is measurable at the Phase-4 gate, and the decision
should be deferred to that measurement rather than made now.**

### 5.4 Blob store

Content-addressed local directory (`blobs/<sha256[:2]>/<sha256>`).
Originals are the one thing that cannot be regenerated. No database
involved.

---

## 6. Parsing

Deterministic parsing is Principle 7's most visible application: an LLM
must never be asked to extract text from a PDF.

| Format | Library | Note |
|---|---|---|
| Markdown | `markdown-it-py` + custom frontmatter | **Must strip fenced/inline code before link extraction** — audit §6.3; 240 code blocks contain `[[...]]` literals |
| PDF text | `pypdfium2` | Apache/BSD-licensed, fast, reliable text + layout |
| PDF tables | `pdfplumber` | Better table extraction when needed |
| Repos | `GitPython` + `tree-sitter` | Structure-aware code chunking |
| Web | `trafilatura` | Main-content extraction, no browser |
| HTML | `selectolax` | Fast parsing |

**PyMuPDF is deliberately not the default despite being the best
extractor**: it is AGPL-3.0 or commercially licensed. For a personal
local-first tool AGPL is likely acceptable, but it is a decision the
user should make knowingly rather than inherit from a dependency
choice. `pypdfium2` avoids the question entirely.

**Scanned PDFs / OCR: out of scope for the first version.** Detect and
report "no extractable text" rather than silently producing empty
documents.

---

## 7. Interfaces

**CLI first (Phase 2).** `typer` or `click`. The CLI is the fastest way
to exercise ingestion and is the interface that keeps working when
everything else is half-built.

**Web UI at Phase 6.** Next.js + TypeScript. The audit found **no
existing Forge interface** — "existing Forge interface if suitable" from
the brief's candidate stack does not apply, since Obsidian is a
third-party editor, not a Forge UI. The Phase-6 need is graph
visualization and provenance inspection, which is genuinely a web
problem.

Graph rendering: evaluate at Phase 6. Cytoscape.js and Sigma.js are the
realistic candidates; D3 offers control at a much higher cost. Not
decided here — it depends on how the graph explorer's interaction model
lands.

**Obsidian plugin at Phase 7.** TypeScript, Obsidian plugin API,
talking to the local Forge API. Constraint from the audit (P3): the
plugin may *display* engine data, but any content it writes into the
vault must remain meaningful as plain Markdown.

**MCP server at Phase 8.** Python MCP SDK, exposing retrieval and
knowledge-model queries as tools. Same core, addressed by an agent
instead of a person.

---

## 8. Deployment

**Docker Compose from Phase 4**, once there is more than one store to
run. Before that, local processes only: SQLite file, host Ollama, no
containers.

Compose services at Phase 4+: `forge-api`, `forge-worker`, `ollama`
(optional — host-native is often better for GPU access), plus whichever
stores have been promoted.

**No cloud dependency in any profile.** Cloud providers are opt-in
configuration, never infrastructure.

---

## 9. Testing and observability

| Concern | Choice |
|---|---|
| Test runner | `pytest` + `pytest-asyncio` |
| Type checking | `mypy --strict` or `pyright`, enforced in CI |
| Lint/format | `ruff` |
| Fixtures | Golden files for parsers/chunkers; recorded LLM responses for pipeline tests |
| Mock LLM | `MockProvider` — deterministic, offline, **the CI default** |
| Logging | `structlog`, JSON, `run_id`-correlated |
| Tracing | OpenTelemetry; a span per LangGraph node |
| Eval | Labeled set drawn from the existing corpus (§10) |

**CI must run with no models and no network.** If the suite needs
Ollama, it will be skipped, and the pipeline will rot.

---

## 10. The corpus as evaluation asset

Worth stating as a decision rather than an afterthought: the 620-file
vault is the **primary evaluation dataset**, because it comes with
ground truth the engine can be scored against —

- ~500 concepts with human-assigned canonical names (filenames);
- ~4,100 human-authored relationships (wikilinks);
- explicit `canonical: true` markers on 358 files;
- consistent heading structures for chunker validation;
- known duplicate-adjacent content (e.g. `Graph Traversal` vs `DFS`/`BFS`,
  `Binary Search` as both pattern and algorithm) for testing concept
  resolution against genuinely hard cases.

Concept extraction can be scored against the filename-derived concept
set. Relationship discovery can be scored against the existing wikilink
graph. **This is a far stronger evaluation position than most greenfield
projects start from**, and it is a direct dividend of the existing
corpus's discipline.

---

## 11. Explicitly rejected

| Technology | Why |
|---|---|
| LangChain as a framework | Hides control flow; hostile to provenance tracking |
| Cloud-only vector DBs (Pinecone, Weaviate Cloud) | Violates Principle 9 |
| Hosted embedding APIs as default | Same; opt-in only |
| Elasticsearch | Operationally heavy for one user; SQLite FTS5 suffices |
| Celery/Redis queues | LangGraph checkpointing covers resumability at this scale |
| Kubernetes | One laptop |
| A separate "agent framework" | Explicit workflows instead of agent abstractions |

---

## 12. Decisions deferred to a measurement

| Deferred | Trigger | Gate |
|---|---|---|
| Postgres | Concurrent writers / multi-user API | Phase 6 |
| Qdrant | >500k vectors or filtered-search latency | Phase 5 |
| **Neo4j** | **Queries routinely >3 hops or needing path-finding** | **Phase 4 — needs human decision** |
| Reranker model | Retrieval precision measured as insufficient | Phase 3 |
| Graph viz library | Interaction model settled | Phase 6 |
| PyMuPDF | User accepts AGPL | Phase 2 |
| OCR | Scanned PDFs actually appear | Post-MVP |

---

## Related

- [Current-state audit](./forge-current-state.md)
- [Target architecture](./target-architecture.md)
- [ADR-001](../decisions/001-forge-knowledge-os.md)
- [Roadmap](../roadmap.md)
