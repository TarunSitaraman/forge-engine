# Forge Knowledge OS — Roadmap

*Phased implementation plan for the Forge engine, with an explicit exit gate per phase.*

**Scope note:** this roadmap covers the **engine**. The root
[`ROADMAP.md`](../ROADMAP.md) covers the Markdown vault's own content
plans and remains valid — the two are separate tracks and should not be
merged.

---

## How phases work

- A phase ends at its **exit gate**, not when its tasks feel done. Every
  gate is a verifiable statement.
- The [boundary checklist](./product/competitive-boundary.md#boundary-review-checklist)
  runs at every gate.
- Phases 1–5 are strictly ordered — each builds on the last. Phases 6–8
  are interface work and can reorder or run in parallel.
- **Phase 10 does not start before Phase 1 is stable.** No polish on an
  unstable model.

---

## Phase 0 — Repository audit & architecture *(complete)*

**Delivered:** [current-state audit](./architecture/forge-current-state.md),
[vision](./product/vision.md),
[positioning](./product/product-positioning.md),
[competitive boundary](./product/competitive-boundary.md),
[target architecture](./architecture/target-architecture.md),
[technology decisions](./architecture/technology-decisions.md),
[canonical model](./knowledge-model/canonical-model.md),
[ADR-001](./decisions/001-forge-knowledge-os.md), this roadmap.

**Gate: passed.** The audit exists; no implementation has begun.

**Blocking on human decision before Phase 1:**
D1 (repository layout) and D2 (write-back policy) — audit §8, ADR-001.

---

## Phase 1 — Canonical knowledge model

The foundation. Everything else is built on this, so it is the phase
most worth slowing down for.

**Scope**
- Entity/relationship implementation: `Source`, `Document`, `Span`,
  `Concept`, `Claim`, `EvidenceLink`, `ClaimLink`, `Provenance`,
  `Revision`.
- Provenance tiers + **floor rule** enforced at write time.
- Revision log — append-only, from the first write.
- Persistence (SQLite) + migrations.
- Provider abstraction + `MockProvider`.
- Config with startup validation.
- **Frontmatter repair migration** — the 283 malformed `related:` fields
  (audit §6.2). First code committed, because nothing downstream can
  trust frontmatter until it lands.
- Spike: local-model claim extraction and contradiction detection
  against real corpus pairs (technology decisions §4.4).

**Gate**
- [ ] A claim cannot be persisted without provenance; the floor rule has
      tests proving synthesis cannot be written as `SOURCE_FACT`
- [ ] Supersession retains both states and writes a `Revision`
- [ ] All 283 `related:` fields parse as valid YAML; full-corpus
      frontmatter parse succeeds with zero errors
- [ ] Full test suite runs offline with no model
- [ ] Spike result recorded: is local `analysis` good enough, and at
      what thresholds?

---

## Phase 2 — Source ingestion infrastructure *(complete)*

**Delivered:** PDF (`pypdfium2`) and Markdown source adapters behind one
acquisition protocol; deterministic structure-aware chunking into spans
carrying page, section, line and character offsets; derivation-key caching;
optional LLM extraction with strict schemas and a verbatim-quote grounding
check; concept candidate matching that never merges; a proposal system with
approval state, safety classification and flag-gated reversible write-back;
lexical retrieval with filters and optional semantic re-rank.

**Gate: passed** — `bash scripts/validate_phase2.sh` (16/16). See
[phase-2-implementation.md](./architecture/phase-2-implementation.md).

**Deviation from the original scope:** the LLM-extraction, concept-matching and
proposal work listed below under Phases 4–5 was pulled forward, because
ingesting external sources without provenance-carrying candidates and a human
approval gate would have meant building the unsafe version first.

**Original scope**
- Source registry: hashing, change detection, dedup.
- Markdown parser (**code-fence-aware** — audit §6.3), PDF parser,
  blob store for originals.
- Structure-aware chunking → spans.
- Corpus backfill workflow (W4), resumable, per-file failure isolation.
- CLI: `forge ingest`, `forge status`, `forge sources`.

**Gate**
- [x] All vault files ingest; failures quarantined, never partial
- [x] Every span resolves to an exact source location
- [x] **Re-ingesting an unchanged source costs zero LLM calls**
- [x] Editing a file reprocesses only that source
- [x] A PDF ingests end to end with page-level provenance
- [x] Parsers pass tests against real PDF fixtures

---

## Phase 3 — Knowledge activation & retrieval *(complete)*

Scope shifted during planning: activation — turning approved proposals into
canonical knowledge — turned out to be the missing link, and embeddings became
a *measurement* rather than a deliverable.

**Delivered**
- Proposal activation (`APPROVED → ACTIVATED`) with deterministic identity,
  evidence links, provenance, and revisions
- Concept identity states plus a persisted user decision file for the vault's
  four real collisions
- Evidence-gated relationship activation over a five-type vocabulary
- A SQLite knowledge graph with bounded traversal and integrity diagnostics
- A labelled retrieval evaluation set (24 queries / 48 labels) and a metrics
  harness
- Embeddings built, measured, and **rejected** on the evidence

**Gate**
- [x] Approved proposals become canonical Concepts and Claims
- [x] Activation is idempotent across approve / re-index / re-activate
- [x] Every result carries a resolvable provenance chain
- [x] **Retrieval works with the LLM entirely disabled**
- [x] Re-embedding is detectable when the model changes (vectors are keyed by
      model id, so a model change invalidates rather than mixes)
- [x] Hybrid retrieval adopted **only if measured better** — it was not.
      Lexical R@10 = 0.650 beat semantic (0.601) and every swept fusion weight
      (0.524 / 0.496 / 0.428). See
      [retrieval baseline](./research/retrieval-baseline.md).

*Caveat carried forward:* the semantic measurement used a non-neural hashing
vectorizer, because no model could be downloaded in this environment. It shows
that vocabulary-overlap vectors do not help; it cannot speak to real
embeddings. Re-running the sweep with a neural model is a one-command job and
is the first thing to do when one is reachable.

---

## Phase 4 — Knowledge graph

Phase 3 delivered the graph *substrate* — storage, bounded traversal,
integrity, and the measurement that says SQLite suffices (0.24 ms neighbour
lookup at 5,000 nodes / 20,000 edges). Phase 4 is about *populating* it at
corpus scale.

**Scope**
- Concept resolution: alias/lexical/vector candidate narrowing, LLM only
  for genuine ambiguity.
- Relationship discovery beyond co-occurrence, over the typed vocabulary.
- Bootstrap concepts from filenames; seed edges from the 4,100 wikilinks.
- Deterministic retrieval improvements measured against the Phase 3 set:
  title/heading boosting, and alias-driven query expansion reusing the
  identity config.

**Gate**
- [ ] Concept extraction scored against the ~500 filename-derived
      ground-truth concepts
- [ ] Relationship discovery scored against the existing wikilink graph
- [ ] `Graph Traversal` vs `DFS`/`BFS` and `Binary Search`
      pattern-vs-algorithm resolve correctly (the known hard cases)
- [ ] `RELATED_TO` edges carry similarity scores and are excluded from
      reasoning traversals
- [x] **Neo4j decision made on measured hop-depth** — measured in Phase 3 and
      answered *no*; `scripts/measure_graph_scale.py` re-runs the measurement

---

## Phase 5 — LangGraph ingestion & evolution workflow

Where Forge stops being a RAG pipeline.

**Scope**
- W1 ingestion/evolution graph: typed state, checkpointing, conditional
  routing, retries, quarantine.
- Change analysis: known / supports / contradicts / refines / new.
- Contradiction detection → `Contradiction` entities.
- Supersession with history retention.
- Synthesis staleness marking (deterministic traversal).
- Approval gates for high-severity contradictions and above-threshold
  merges.
- W2 contradiction resolution workflow.

**Gate**
- [ ] **The MVP acceptance scenario passes end to end (§MVP below)**
- [ ] A second overlapping document updates the model rather than
      duplicating it
- [ ] A contradicting document produces a `Contradiction`, not a silent
      overwrite
- [ ] Interrupting mid-ingestion and resuming does not duplicate work
- [ ] Every model change traces to a `run_id` and a `Revision`
- [ ] No node is named for a persona

---

## Phase 6 — Knowledge exploration interface

**Scope**
- FastAPI read endpoints; graph explorer; concept → claims → evidence →
  source drill-down; provenance tier always visible; revision timeline.

**Gate**
- [ ] From any claim, reach the exact source span in one interaction
- [ ] Generated content is **visually distinguishable** from source
      evidence — a UI requirement derived from Principle 10
- [ ] The model is comprehensible with no chat interface present

---

## Phase 7 — Obsidian integration

**Scope**
- Plugin surfacing concepts, claims, contradictions, and related
  evidence for the current note; links back to the graph explorer.
- Write-back **only** if ADR-001 D2 permits, and only into a segregated,
  provenance-stamped namespace.

**Gate**
- [ ] Vault remains fully usable with the plugin disabled
- [ ] Anything written is valid, meaningful plain Markdown
- [ ] No plugin-only constructs introduced

---

## Phase 8 — MCP interface

**Scope**
- MCP server exposing retrieval and knowledge-model queries as tools;
  read-only initially.

**Gate**
- [ ] An external agent can query the model and receive provenance with
      every result
- [ ] Identical semantics to the HTTP API — no capability lives only in
      one interface

---

## Phase 9 — Research intelligence

Where the vision's questions become answerable.

**Scope**
- `Question` and `KnowledgeGap` entities; gap detection by deterministic
  graph queries.
- Belief queries ("what do I believe about X"), change queries ("what
  changed this month"), relevance-to-open-question retrieval.
- W3 re-synthesis workflow.

**Gate**
- [ ] All six vision questions answerable with sources and dissent
- [ ] Gap detection produces findings a human agrees are real gaps
- [ ] Syntheses auto-mark stale when constituent claims change

---

## Phase 10 — Polish, testing, documentation, release

**Scope**
- Coverage, performance, error-message quality, deployment docs,
  end-user documentation, backup/restore for derived stores.

**Gate**
- [ ] Fresh-machine setup works from documentation alone, with no paid
      API
- [ ] Full rebuild from Markdown reproduces the derived model
- [ ] Backup/restore covers state Markdown cannot express (ADR-001 R5)

---

## MVP — the vertical slice that proves the thesis

Delivered by the **end of Phase 5**. This is the acceptance test for
the entire foundation:

1. Add a PDF.
2. Forge extracts content — deterministically.
3. Forge identifies concepts using a local model.
4. Forge finds related existing concepts.
5. Forge identifies relationships.
6. Forge stores the knowledge model.
7. Forge stores provenance.
8. Forge displays the resulting graph.
9. User clicks a concept.
10. User sees the supporting source evidence.
11. User adds a second, overlapping document.
12. Forge detects the overlap.
13. **Forge updates the graph rather than creating duplicate notes.**

**Steps 11–13 are the MVP.** Steps 1–10 are a competent RAG pipeline
that many tools already deliver; the second document is where Forge
either maintains understanding or merely stores information. A
demonstration that stops at step 10 has not demonstrated the product.

Steps 8–10 need a viewer, which formally belongs to Phase 6 — a minimal
read-only graph view is pulled forward into Phase 5 for exactly this
reason, and nothing more.

---

## Sequencing risks

| Risk | Handling |
|---|---|
| Phase 1 feels slow with nothing demoable | Correct and intentional. Provenance and history cannot be retrofitted; a system that starts logging revisions at Phase 9 has no history for Phases 1–8 |
| Local model quality blocks Phase 5 | Spiked in Phase 1, not discovered in Phase 5 |
| Interfaces tempt early attention | Phases 6–8 sit behind the model deliberately |
| Store choices block progress | Everything is behind protocols; only Neo4j is a real gate (Phase 4) |
| Scope creep toward chatbot | Boundary checklist at every gate |

---

## Related

- [Vision](./product/vision.md) · [Positioning](./product/product-positioning.md) · [Competitive boundary](./product/competitive-boundary.md)
- [Current-state audit](./architecture/forge-current-state.md) · [Target architecture](./architecture/target-architecture.md) · [Technology decisions](./architecture/technology-decisions.md)
- [Canonical knowledge model](./knowledge-model/canonical-model.md)
- [ADR-001](./decisions/001-forge-knowledge-os.md)
