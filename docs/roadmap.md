# Forge Knowledge OS: Roadmap

*Phased implementation plan for the Forge engine, with an explicit exit gate per phase.*

**Scope note:** this roadmap covers the **engine**. The vault repository
keeps its own `ROADMAP.md` for the Markdown vault's content plans, which
remains valid: the two are separate tracks and should not be merged.

---

## How phases work

- A phase ends at its **exit gate**, not when its tasks feel done. Every
  gate is a verifiable statement.
- The [boundary checklist](./product/competitive-boundary.md#boundary-review-checklist)
  runs at every gate.
- Phases 1-5 are strictly ordered, each builds on the last. Phases 6-8
  are interface work and can reorder or run in parallel.
- **Phase 10 does not start before Phase 1 is stable.** No polish on an
  unstable model.

---

## Phase 0: Repository audit & architecture *(complete)*

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
D1 (repository layout) and D2 (write-back policy), audit §8, ADR-001.

---

## Phase 1: Canonical knowledge model

The foundation. Everything else is built on this, so it is the phase
most worth slowing down for.

**Scope**
- Entity/relationship implementation: `Source`, `Document`, `Span`,
  `Concept`, `Claim`, `EvidenceLink`, `ClaimLink`, `Provenance`,
  `Revision`.
- Provenance tiers + **floor rule** enforced at write time.
- Revision log, append-only, from the first write.
- Persistence (SQLite) + migrations.
- Provider abstraction + `MockProvider`.
- Config with startup validation.
- **Frontmatter repair migration.** The 283 malformed `related:` fields
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

## Phase 2: Source ingestion infrastructure *(complete)*

**Delivered:** PDF (`pypdfium2`) and Markdown source adapters behind one
acquisition protocol; deterministic structure-aware chunking into spans
carrying page, section, line and character offsets; derivation-key caching;
optional LLM extraction with strict schemas and a verbatim-quote grounding
check; concept candidate matching that never merges; a proposal system with
approval state, safety classification and flag-gated reversible write-back;
lexical retrieval with filters and optional semantic re-rank.

**Gate: passed** by `bash scripts/validate_phase2.sh` (16/16). See
[phase-2-implementation.md](./architecture/phase-2-implementation.md).

**Deviation from the original scope:** the LLM-extraction, concept-matching and
proposal work listed below under Phases 4-5 was pulled forward, because
ingesting external sources without provenance-carrying candidates and a human
approval gate would have meant building the unsafe version first.

**Original scope**
- Source registry: hashing, change detection, dedup.
- Markdown parser (**code-fence-aware**, audit §6.3), PDF parser,
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

## Phase 3: Knowledge activation & retrieval *(complete)*

Scope shifted during planning: activation, turning approved proposals into
canonical knowledge, turned out to be the missing link, and embeddings became
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
- [x] Hybrid retrieval adopted **only if measured better**. It was not.
      Lexical R@10 = 0.608 beat semantic (0.581) and every swept fusion weight
      (0.544 / 0.517 / 0.449). See
      [retrieval baseline](./research/retrieval-baseline.md).

*Caveat carried forward:* the semantic measurement used a non-neural hashing
vectorizer, because no model could be downloaded in this environment. It shows
that vocabulary-overlap vectors do not help; it cannot speak to real
embeddings. Re-running the sweep with a neural model is a one-command job and
is the first thing to do when one is reachable.

---

## Phase 4: Agentic knowledge evolution *(complete)*

The plan had Phase 4 as "knowledge graph population" and Phase 5 as "LangGraph
workflow". Those were swapped in execution, for a reason worth recording: the
capability that mattered was not *more* knowledge, it was Forge noticing when
new evidence **changes** knowledge it already holds. Graph population without
that is a bigger pile of facts nobody is maintaining.

**Delivered**
- A stateful LangGraph workflow: observe evidence → identify affected concepts
  → retrieve claims → assess → classify impact → propose → **pause for a
  human** → activate → revise
- Deterministic-first candidate narrowing, with a recorded reason per candidate
  and zero model calls
- Grounded semantic assessment: five classifications, no `CONTRADICTS`,
  citations verified against real stored spans, ungrounded output rejected
- Three evolution proposal types: corroborate, refine (supersede,
  non-destructively), flag as disputed (never retract)
- Checkpointing and resume across a real process restart
- A provider-agnostic layer: local Ollama, remote Ollama, cloud, mock: with
  no silent downgrade for knowledge mutation

**Gate**
- [x] LangGraph genuinely orchestrates; services remain plain Python
- [x] Typed, serializable, checkpointed state
- [x] Human interruption and resume, verified across a process restart
- [x] New evidence can change existing knowledge, only via approval
- [x] Candidate narrowing is deterministic-first (0 model calls, asserted)
- [x] Assessments are grounded in real spans; hallucinated citations rejected
- [x] Potential conflicts require human review
- [x] Provenance records provider, model, prompt version, schema version
- [x] Ollama, remote Ollama, and cloud all work through one abstraction
- [x] No provider is required for deterministic operation
- [x] Assessments cached; provider/model/prompt/schema changes invalidate
- [x] Duplicate execution is safe, 0 new entities, 0 model calls
- [x] CI is fully offline
- [x] **Real-model evaluation, local**: run 2026-08-14 on Qwen3 8B / RTX
      4050: 5/5 classifications, 1.00 structured-output validity, 1.00
      grounding, 0 false-positive conflicts. A passing smoke test, not a
      characterisation. See
      [provider availability](./research/provider-availability.md) §6.
- [x] **Real-model evaluation, cloud**: run 2026-09-03 against
      `openai/gpt-oss-120b` on Groq, 21 cases. Structured-output validity
      1.00, grounding 1.00, classification 0.76. The average hides the
      finding: SUPPORTS, REFINES and IRRELEVANT are each 100%, while
      **INSUFFICIENT_EVIDENCE is 2/6** and accounts for four of the five
      failures, the model reaches for a nearby label instead of declining.
      One of those four read a mechanism as support for an outcome and produced
      a `CLAIM_EVIDENCE` proposal, which is the error shape a provenance
      floor cannot catch: the citation is real, the reasoning is not. See
      [assessment quality](./research/assessment-quality.md).

*Deferred from the original Phase 4 scope, now the leading candidates for
Phase 5:* bootstrapping concepts from filenames, seeding edges from the ~4,100
wikilinks, relationship discovery beyond co-occurrence, and the two
deterministic retrieval improvements the Phase 3 miss analysis identified
(title/heading boosting, alias-driven query expansion).

---

## Phase 5: Real-model validation and graph population

Phase 4 delivered the workflow this phase was originally scoped to build, so
Phase 5 becomes the two things Phase 4 could not do: **prove the pipeline works
with a real model**, and populate the graph at corpus scale.

**Scope**
- Expand the assessment set well beyond 5 cases. The local smoke test passed
  5/5, which is consistent with a model that is right 60% of the time: the
  set is now the binding constraint on what can be claimed, not the model.
- Measure the **false-positive conflict rate** properly. Zero false positives
  on two adversarial cases is encouraging and is not a rate. This needs enough
  IRRELEVANT and INSUFFICIENT_EVIDENCE cases to put a real bound on it.
- Run the same set against a cloud model. Report as two rows, never averaged.
  They are different instruments.
- Act on the latency finding: 63 s/case locally, one call over the 120 s
  timeout. Raise the default timeout, and measure whether a larger assessment
  batch degrades accuracy, since per-call overhead now dominates.
- Expand the assessment set beyond 5 cases once a real model shows where it is
  weak.
- Bootstrap concepts from filenames; seed edges from the ~4,100 wikilinks
  (deferred from the original Phase 4 scope).
- Relationship discovery beyond co-occurrence.
- The two deterministic retrieval improvements the Phase 3 miss analysis
  identified: title/heading boosting and alias-driven query expansion.
- Evolution beyond claims: let new evidence refine a *concept* or a
  relationship, not only a claim.

**Gate**
- [x] Assessment metrics measured on a real model, cloud, 2026-09-03,
      21 cases against `openai/gpt-oss-120b`. **The local half is dropped
      rather than pending:** the 2026-08-14 Qwen3 8B result came from an
      RTX 4050 machine no longer in use, and current work runs against Groq.
      Two models on one hosted provider is the comparison available; a local
      row would be a different instrument and is not required to close this.
- [x] False-positive conflict rate measured, **2 of 18 non-conflict cases,
      11.1%** on the fitted set, *unchanged* by the 0.2.0 prompt fix that took
      classification 0.76 → 0.86: one false conflict was fixed and a different
      one created. **1 of 16 (6.2%) on the held-out set**, where the REFINES
      regression reproduced on a fresh case, confirming it as behaviour rather
      than noise. Held-out also showed cases the prompt cues describe and cases
      they do not scoring identically (3/5 each), so prompt instruction is not
      the binding constraint on this class.
      Conflict recall 2/3; the miss absorbed a contrary finding as a REFINES,
      which is the costlier direction since a refinement supersedes without a
      human looking. **Not** judged acceptable for promoting
      `POTENTIAL_CONFLICT` to an asserted `Contradiction`; human routing
      stays. See [assessment quality](./research/assessment-quality.md).
- [ ] A second overlapping document updates the model rather than duplicating it
- [ ] Interrupting mid-ingestion and resuming does not duplicate work *(already
      true for evolution; needs proving for ingestion)*
- [ ] Every model change traces to a workflow id and a `Revision` *(already true)*
- [x] **Graph populated deterministically, 2026-09-06.** `forge bootstrap
      --apply` over the vault: **545 concepts, 2,752 RELATED_TO edges, 0 LLM
      calls**, 125 navigation and template pages skipped. Graph stats: mean
      degree 10.1, max 163, 72 isolated nodes, neighbour query 0.5 ms, path
      query 91 ms. This was listed as "deferred from the original Phase 4
      scope" and was in fact already built; the box was stale, which is the
      failure mode this document warns about elsewhere. Recount, do not carry
      forward.
- [ ] Concept extraction scored against the 545 filename-derived concepts
      *(the reference set now exists; scoring needs a model run)*
- [ ] Retrieval improvements measured against the Phase 3 labelled set

*A structural corroboration pass exists but is off by default*, a second
question over any SUPPORTS/REFINES, measured 2026-09-05 at 13/18 with and
13/18 without on the held-out set (50% precision, 33% recall on the failures
it targets). Three attempts at this class have now failed, so the constraint
is read as the model's judgement rather than the prompting. `--corroborate`
enables it; see [assessment quality](./research/assessment-quality.md) §9.

*Still deliberately not built:* contradiction *detection* as an autonomous
capability. Phase 4's `POTENTIAL_CONFLICT` routes to a human by design, and
promoting it to an asserted `Contradiction` entity should wait until the
false-positive rate is measured.

---

## Phase 6: Knowledge exploration interface

**Scope**
- FastAPI read endpoints; graph explorer; concept → claims → evidence →
  source drill-down; provenance tier always visible; revision timeline.

**Gate**
- [ ] From any claim, reach the exact source span in one interaction
- [ ] Generated content is **visually distinguishable** from source
      evidence, a UI requirement derived from Principle 10
- [ ] The model is comprehensible with no chat interface present

---

## Phase 7: Obsidian integration

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

## Phase 8: MCP interface

**Scope**
- MCP server exposing retrieval and knowledge-model queries as tools;
  read-only initially.

**Gate**
- [ ] An external agent can query the model and receive provenance with
      every result
- [ ] Identical semantics to the HTTP API, no capability lives only in
      one interface

---

## Phase 9: Research intelligence

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

## Phase 10: Polish, testing, documentation, release

**Scope**
- Coverage, performance, error-message quality, deployment docs,
  end-user documentation, backup/restore for derived stores.

**Gate**
- [ ] Fresh-machine setup works from documentation alone, with no paid
      API
- [ ] Full rebuild from Markdown reproduces the derived model
- [ ] Backup/restore covers state Markdown cannot express (ADR-001 R5)

---

## MVP: the vertical slice that proves the thesis

Delivered by the **end of Phase 5**. This is the acceptance test for
the entire foundation:

1. Add a PDF.
2. Forge extracts content, deterministically.
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

**Steps 11-13 are the MVP.** Steps 1-10 are a competent RAG pipeline
that many tools already deliver; the second document is where Forge
either maintains understanding or merely stores information. A
demonstration that stops at step 10 has not demonstrated the product.

Steps 8-10 need a viewer, which formally belongs to Phase 6: a minimal
read-only graph view is pulled forward into Phase 5 for exactly this
reason, and nothing more.

---

## Sequencing risks

| Risk | Handling |
|---|---|
| Phase 1 feels slow with nothing demoable | Correct and intentional. Provenance and history cannot be retrofitted; a system that starts logging revisions at Phase 9 has no history for Phases 1-8 |
| Local model quality blocks Phase 5 | Spiked in Phase 1, not discovered in Phase 5 |
| Interfaces tempt early attention | Phases 6-8 sit behind the model deliberately |
| Store choices block progress | Everything is behind protocols; only Neo4j is a real gate (Phase 4) |
| Scope creep toward chatbot | Boundary checklist at every gate |

---

## Related

- [Vision](./product/vision.md), [Positioning](./product/product-positioning.md), [Competitive boundary](./product/competitive-boundary.md)
- [Current-state audit](./architecture/forge-current-state.md), [Target architecture](./architecture/target-architecture.md), [Technology decisions](./architecture/technology-decisions.md)
- [Canonical knowledge model](./knowledge-model/canonical-model.md)
- [ADR-001](./decisions/001-forge-knowledge-os.md)
