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

## Phase 1: Canonical knowledge model *(complete)*

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
- [x] A claim cannot be persisted without provenance; the floor rule has
      tests proving synthesis cannot be written as `SOURCE_FACT`.
      `tests/unit/test_provenance.py`; the rule is a pydantic validator, so a
      violating object cannot be constructed at all.
- [x] Supersession retains both states and writes a `Revision`.
      `tests/unit/test_revision_and_model.py`.
- [x] All 283 `related:` fields parse as valid YAML; full-corpus frontmatter
      parse succeeds with zero errors. `forge diagnostics frontmatter` on the
      vault reports 407 files with frontmatter, 407 valid, 0 invalid, over 892
      `related:` entries (2026-09-08; the 283 was the pre-repair audit figure).
- [x] Full test suite runs offline with no model. 1,435 pass with no provider
      configured, and `CALLS.count` is asserted rather than assumed.
- [x] Spike result recorded: is local `analysis` good enough, and at what
      thresholds? [`docs/research/local-model-capability-spike.md`](research/local-model-capability-spike.md).

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
- [x] **A second overlapping document updates the model rather than
      duplicating it, 2026-09-06.** MVP steps 11-13, asserted end to end in
      `tests/integration/test_mvp_second_document.py`: ingest, activate,
      ingest an overlapping document, activate again, then check the shape of
      the graph on the other side.

      Detection was already covered; **step 13 was not**, and the two are
      different claims. An unchanged concept count is also what a refused
      duplicate looks like, so the tests assert the mechanism: the second
      document raises `CONCEPT_MATCH` with `match_candidate` and **no**
      `NEW_CONCEPT`, activation registers the name as an alias of the existing
      concept, the id survives, revisions are never lost, and the second
      source and its spans are still stored so its evidence stays citable.

      **The boundary, measured rather than assumed:** matching is lexical, so
      `Hybrid Retrieval` against a stored `Hybrid Search` does not match and is
      proposed as a new concept. The duplicate never reaches the graph, because
      it is a pending proposal and nothing activates without approval. So the
      defensible claim is **the graph never duplicates without a human
      approving the duplicate**, not that overlap detection is complete.
      Embeddings would widen it and are off by default.
- [x] **Interrupting mid-ingestion and resuming does not duplicate work,
      2026-09-06.** Proving it found a defect, so it was not true when this box
      was written.

      `tests/integration/test_ingestion_resume.py` states the gate as an
      equality rather than an absence: **a partial run followed by a resume
      must leave the same store as one uninterrupted run.** Weaker phrasings
      ("no duplicate spans", "no crash") pass for a system that silently drops
      the work it was interrupted during.

      It failed. An interrupted run left **4 proposals where an uninterrupted
      one left 3**, plus an extra revision. Cause: extraction is cached, so the
      resumed run re-derived nothing and cost no model calls, but it still
      reached `_propose`, and by then the matcher had learned about the
      proposals the first run made. A source that had proposed `Test Concept`
      as NEW_CONCEPT came back and raised a CONCEPT_MATCH **against its own
      earlier proposal**. One source, two live proposals, one name.

      Fixed by making `_propose` idempotent per source: names this source has
      already proposed are skipped, with rejected proposals excluded so a
      human's "no" is not re-asked. Three of the six tests fail without the
      fix. The remaining tests also pin that a resumed document costs zero
      model calls, that ingesting deterministically and extracting later still
      works (the silent-no-op defect `_ingest_one` documents), and that
      extraction over stored spans does not re-chunk or bump the document
      version.
- [x] **Every model change traces to a workflow id and a `Revision`,
      2026-09-08.** This box carried the annotation "(already true)" and was
      not. The Revision half held; the workflow half did not, and no test in
      the suite so much as named `workflow_run_id`. Writing the four that do
      (`TestEveryModelChangeIsTraceable`) failed two of them immediately:
      `supersede_claim` wrote both of its revisions with no workflow id, so a
      **refinement**, the model's most consequential change, was the one that
      could not be traced back to the run that made it. The id is now stamped
      from the entity's own provenance inside `_append`, rather than passed by
      each of the fourteen callers that write a revision, because a parameter
      every caller must remember is one a caller will forget. Supersession
      passes its own: the run that retires a claim is the one that produced
      the replacement, not the one that created the claim.
- [x] **Graph populated deterministically, 2026-09-06.** `forge bootstrap
      --apply` over the vault: **545 concepts, 2,752 RELATED_TO edges, 0 LLM
      calls**, 125 navigation and template pages skipped. Graph stats: mean
      degree 10.1, max 163, 72 isolated nodes, neighbour query 0.5 ms, path
      query 91 ms. This was listed as "deferred from the original Phase 4
      scope" and was in fact already built; the box was stale, which is the
      failure mode this document warns about elsewhere. Recount, do not carry
      forward.
- [ ] Concept extraction scored against the 545 filename-derived concepts.
      **The harness is built and verified offline, 2026-09-06; only the model
      run is outstanding.** `scripts/concept_extraction_eval.py` scores
      extraction against the vault's own page names, which nobody labelled for
      this purpose, and reports three numbers rather than recall:
      **self-recovery** (does the page about X yield X?), **junk rate** against
      the same 25 observed-junk strings `extraction-eval` uses, so the two are
      quotable side by side, and **off-vocabulary rate**, which is reported and
      explicitly not called junk. Global recall is not reported: a page about
      B-trees is not supposed to mention 544 other concepts, so "3 of 545"
      would be arithmetic rather than a finding. One command when the key is
      next available:

          python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
          .venv/bin/python scripts/concept_extraction_eval.py \
              --vault ~/forge --provider cloud --limit 40 --sleep 20 --json \
              > phase5-concept-eval.json

      Both flags in that command are there because the shorter version failed.
      `--vault` because vault resolution looks upward from the working
      directory: run this from inside the engine checkout and it resolves the
      *repository*, derives 23 concepts from `docs/`, and reports a
      self-recovery rate over architecture notes in exactly the format it uses
      for the vault. That case is refused by name now, and the vault and its
      concept count print before the run rather than in the report at the end
      of it. The venv because the script imports the engine from source, so the
      interpreter running it needs the package's dependencies; a bare `python3`
      that is not the one `forge` runs on fails on `structlog`.

      **`--sleep` is not optional, and this line said so only after somebody
      tried the command.** It defaults to 0, and the version printed here
      until 2026-09-08 omitted it. A free hosted tier limits tokens per minute
      rather than requests: with `max_tokens` reserved against an 8,000 TPM
      budget the ceiling is about two calls a minute however they are spaced,
      so an unpaced run spends its budget in the first few seconds and 429s
      for the rest. Each failed page is recorded incomplete rather than
      crashing, so the result would have been a report full of holes that took
      an hour to produce. The script now prints its call budget and expected
      wall clock before starting, and warns when cloud is paired with no
      pacing. At `--sleep 20` the 240 calls are about 80 minutes.

      The harness was re-run offline against the vault on 2026-09-08, after a
      week of extractor changes: 544 concept pages, seeded sample, and every
      adversarial probe came back dropped. What remains is the model run,
      which needs a key.

      Building it produced one result already, out of the offline mode, and a
      fix. The scripted provider pairs every verbatim quote with an
      adversarial probe, the same line with its token order reversed, and
      every probe must come back dropped. Ten did not.

      **Cause: `_grounded` fell back to a longest-common-subsequence *ratio*
      over the whole span.** A ratio is scale-free, so a short quote against a
      span that repeats its tokens is cheap to satisfy: the quote's words need
      only appear in ascending order somewhere, and five near-identical
      `assert sorted(result) == sorted(...)` lines supply as many ascending
      positions as the quote has words. The fully reversed quote scored 1.000
      and was accepted as evidence. Every negative case ever written for
      `_grounded` was prose, and prose does not repeat its tokens that way.

      **Fixed 2026-09-06:** the match is now constrained to a window roughly
      the length of the quote, which encodes what a quote is, a contiguous
      passage. Re-measured over all 545 pages, 1,536 probes: **10 survived
      before, 0 after**, and reverting the window reproduces the 10, so the
      offline check is sensitive rather than merely green. The eval exits
      non-zero on a survivor.

      Two things worth carrying forward. The window factor is **not
      knife-edge**, and the first version of its comment claimed it was: the
      defect returns only at factor 12, where the window is the whole span
      again. And **11 of the first pass's apparent survivors were the harness,
      not the check**: probes built by reversing *words* barely permute
      *tokens* when one word holds most of them, so probes are now built at
      the token level and degenerate lines are skipped rather than counted
      against the check.

      **This does not move the assessment numbers.** That path grounds on span
      ids, not quoted text, so the classification and false-positive-conflict
      figures above are untouched. See
      [extraction against the vault](./research/extraction-against-the-vault.md).
- [x] **Retrieval improvements measured against the Phase 3 labelled set.**
      Both deterministic improvements the Phase 3 miss analysis proposed are
      resolved, and neither shipped as an improvement.

      **Title/heading boosting: measured, and a regression.** Over 670 sources
      and 8,133 spans, every boost scores below plain lexical; the shipped 1.25
      cost 0.062 of R@10 and 0.200 of `fuzzy_concept`, the hardest category. It
      is inert on the categories it was meant to help, because it can only fire
      where BM25 already ranks the page first. **Defaulted to 1.0 on
      2026-09-06**, and the docstring beside it, which cited a superseded sweep
      over a corpus a quarter the size, was corrected. The research note that
      the finding "might not transfer to answering" was also wrong and is
      withdrawn: `ask()` and the evaluator issue the identical
      `SearchService.search` call.

      **Alias-driven query expansion: not attempted, and correctly so.** The
      vault contains zero `aliases:` frontmatter keys across 671 files and zero
      aliased wikilinks across 4,703 links. There is nothing to expand from.
      That is corpus work, not engine work, and should be measured against
      `fuzzy_concept` once the aliases exist.

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

## Phase 6: Knowledge exploration interface *(complete)*

**Scope**
- FastAPI read endpoints; graph explorer; concept → claims → evidence →
  source drill-down; provenance tier always visible; revision timeline.

**Delivered, 2026-09-07**
- `forge serve`: a read-only HTTP API (`forge.api`) and a graph explorer, both
  behind the `api` extra so indexing a vault never needs a web framework.
- Endpoints for concepts, claims, evidence, spans, sources, revisions, lexical
  search, graph neighbours and bounded path search, plus `/stats`.
- A single-file explorer served at `/`, vanilla JS over that JSON API. No build
  step and no framework: it is a client of the API and nothing else, so a view
  that needs data the API does not publish is a gap in the API.

**Gate**
- [x] **From any claim, reach the exact source span in one interaction.**
      `GET /claims/{id}` returns the claim *and* its evidence chain, with each
      span's verbatim text, its citation and its source locator. The test
      asserts the request count, not merely that the data is reachable: an API
      that returned span ids and made the client fetch each one would satisfy a
      weaker reading of this and fail the gate.
- [x] **Generated content is visually distinguishable from source evidence.**
      Every payload that can carry provenance does, and the explorer keys its
      styling on it. Colour is not the only signal: model-derived objects also
      carry a dashed left edge and the word "model" spelled out, and the
      overview states the mapping rather than leaving a reader to infer it.

      Building this found a real instance of the failure the gate exists to
      prevent. `Derivation` serializes lower-case (`model`) while
      `ProvenanceTier` serializes upper-case (`EXTRACTED_CLAIM`), and the
      explorer's first version compared `derivation === "MODEL"`, which never
      matched. Model-derived content was marked only when its tier happened to
      give it away. Caught by the gate test, fixed with a case-insensitive
      comparison, and the serialized casing of all three fields a client styles
      on is now pinned by its own test.
- [x] **The model is comprehensible with no chat interface present.** There is
      no prompt box, asserted structurally rather than by searching for the
      word "chat": no `<textarea>`, no `<form>`, exactly one `<input>` (the
      concept name filter), and nothing on the page issues anything but a GET.
      A word search would fail on the page's own comment explaining the absence
      and pass on a prompt box named something else.

**Read-only is enforced, not promised.** A test walks the OpenAPI schema and
fails on any non-GET route. Knowledge changes through proposal and activation,
which require a human decision; an HTTP write path would be a second way in
without that gate. The API also makes **zero model calls**, asserted across
every route the way the rest of the engine asserts it, and `/stats` publishes
the counter so a reader can see it.

**One defect found on the first smoke run, worth recording.** `sqlite3`
connections are bound to their creating thread and FastAPI runs sync endpoints
in a threadpool, so a single shared store raised `ProgrammingError` on the
first route that touched the database. Fixed by opening a connection per
request rather than passing `check_same_thread=False`, which silences the check
without making the connection safe to share. Two tests pin it, including one
that issues eight concurrent requests.

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

## Phase 8: MCP interface *(complete)*

**Scope**
- MCP server exposing retrieval and knowledge-model queries as tools;
  read-only initially.

**Delivered, 2026-09-07**
- `forge mcp`: 14 read-only tools over stdio, behind the `mcp` extra.
- **`forge.api.queries`, a single service layer both interfaces call.** The
  HTTP routes were rewritten as wrappers over it in the same change, so there
  is one implementation of every query rather than two that agree today.

**Gate**
- [x] **An external agent can query the model and receive provenance with
      every result.** The tools return the same models the HTTP API does, so
      the MCP *output schema* carries provenance as part of the published
      contract rather than as a convention an agent has to discover. Derived
      knowledge carries a provenance tier, quoted source material carries a
      trust tier and a locator, a revision carries the cause that triggered it.
      A test sweeps every capability and fails on any record that arrives
      unattributable.

      Two gaps were closed to make that true rather than nearly true. A path
      result published only edge *types*, so an agent was told two concepts
      were `RELATED_TO` with no way to tell a human-authored wikilink from a
      model's guess; every hop now carries its own provenance and rationale.
      Search hits and spans now carry the source's trust tier for the same
      reason.

      The server's `instructions` tell an agent how to read the distinction
      before it calls anything, because a distinction the agent flattens when
      reporting is one that was not kept.
- [x] **Identical semantics to the HTTP API, no capability lives only in one
      interface.** `queries.CAPABILITIES` names each capability once; the HTTP
      layer stamps it as the route's `operation_id` and the MCP layer names the
      tool the same. One test compares all three sets, so a capability added to
      one interface and forgotten in the other fails the suite.

      A second test compares the **payloads**: same store, same arguments, both
      interfaces, asserting the JSON is equal across twelve capabilities. The
      registry check alone would pass for two interfaces that share names and
      disagree about meaning. It excludes `neighbor_query_ms` and
      `path_query_ms`, which `GraphMetrics` measures at call time and which
      therefore differ between any two calls, including two to the same
      interface.

      `health` is the one deliberate asymmetry: it is how the HTTP interface is
      operated, not something an agent can ask about the knowledge model, and
      MCP has its own liveness semantics. The test names the exclusion rather
      than letting the sets quietly differ.

**Read-only, and asserted.** No tool is named for a write, and the tools make
zero model calls. An agent is the caller you least want holding a write path:
knowledge changes through proposal and activation, which require a human
decision.

**One test drives the real transport.** It spawns `forge mcp` and speaks the
protocol to it, because the failure most likely to break an agent integration
silently is not in the tools: **stdout is the protocol channel**, so one stray
`print` during startup corrupts the stream and every client sees a parse error
rather than a Forge problem. The CLI writes diagnostics to stderr for that
reason, and that test is what checks it.

---

## Phase 9: Research intelligence *(complete)*

Where the vision's questions become answerable.

**Scope**
- `Question` and `KnowledgeGap` entities; gap detection by deterministic
  graph queries.
- Belief queries ("what do I believe about X"), change queries ("what
  changed this month"), relevance-to-open-question retrieval.
- W3 re-synthesis workflow.

**Delivered, 2026-09-07**
- `Question` and `Synthesis` entities, schema v5, with a tested v4 upgrade.
- `KnowledgeGap` **computed, never stored**: a stored gap is wrong the moment
  the missing claim arrives and nothing would notice. The cost is that a user
  cannot yet dismiss a gap they disagree with, which is a real follow-up.
- `forge.research`: belief, change and gap queries plus question-scoped
  retrieval. Zero model calls anywhere in the package.
- `forge question add|list`, `forge gaps`, `forge changes`, `forge belief`, and
  eight new capabilities in **both** the HTTP API and MCP, which the Phase 8
  parity tests now hold to 22.

**Gate**
- [x] **All six vision questions answerable with sources and dissent.** One
      test per question, named for it.

      "Dissent" needed care. Forge has no `CONTRADICTS` edge on purpose: Phase
      4 produces `POTENTIAL_CONFLICT` and routes it to a human rather than
      asserting a contradiction a model detected. So dissent is reported as the
      three real things it can be, a disputed claim, a superseded one, and a
      conflict proposal awaiting review, and never as a verdict.

      There is also **no confidence score**, which the vision's table asks for
      as "claims with confidence". `forge.evolution.impact` refuses to invent
      one, and it is right: a number a model emits about its own certainty is
      not a measurement. A belief returns every held claim with how it was
      derived, which is strictly more information than a single score. Building
      this found that the Phase 6 API had been publishing a `confidence` field
      read off a `Provenance` that has no such attribute, so it was `null` in
      every response ever served, quietly suggesting the system calibrates.
      Removed.
- [x] **Gap detection produces findings a human agrees are real gaps.**
      *Closed 2026-09-07 by reading them, and the answer was no until the rule
      was changed.* Five
      deterministic rules, each tested to fire on exactly the structure it
      claims and stay silent otherwise. Run over the real vault graph:

      | finding | count |
      | ---- | ---- |
      | `concept_without_claims` | 544 of 545 |
      | `isolated_concept` | **72** (see below; 71 of these were wrong) |
      | `claim_with_single_source` | 1 |

      That run changed the design. 544 of 545 is *one* fact about the state of
      extraction, not 544 gaps, and printing it buries the 72 isolated concepts
      underneath it, which are individually actionable. A kind that applies to
      most of its population is now summarised in one line, with the instances
      still counted and still listable by naming that kind. A second run found
      the corner: at a population of one, any finding is trivially "100%
      saturated", so a legitimate single-source finding was summarised away.
      There is now a minimum population.

      **The 72 were then read, one at a time, and 71 were wrong.** Every
      reported page was checked against the vault's own links: 115 links point
      at those 72 pages, and not one comes from a page the graph counts.
      `isolated_concept` was reporting graph degree, and edges only run between
      concept pages, and the `_index.md` hubs that do the linking in a
      hub-and-spoke vault are deliberately excluded as navigation, so all 658 of
      their links are dropped. The most-linked page on the "isolated" list had
      **12** inbound links.

      Worse than noisy: a genuinely unreferenced page
      (`DSA/Forge Engineering Constitution.md`) was linked from the index page
      its four siblings are linked from, exactly as the vault's conventions
      require, and stayed on the list, because that index is navigation. A
      finding a user cannot clear by doing the right thing teaches them to
      ignore the report.

      Fixed by counting inbound links over **every** page during bootstrap
      (schema v6, `concept_inbound_links`) and reading that count instead of
      degree. Isolation is about arriving, not leaving: a page with twenty
      outgoing links that nothing points at is exactly as unreachable as one
      with none, so an intermediate version requiring degree 0 *and* inbound 0
      was rejected for hiding 52 real cases. The corpus now reports **53
      findings, all true**: 26 cheat sheets and 8 templates nothing links to,
      14 problem pages missing from their pattern's index, one interview guide,
      the losing side of three decided name collisions, and one orphaned page.
      A store that predates the count falls back to degree and says so in the
      finding's own text.
- [x] **Syntheses auto-mark stale when constituent claims change.** Computed
      deterministically, never by a model: the store snapshots a fingerprint of
      every constituent claim at generation, and a later write compares.

      The fingerprint is status, tier, statement and supersession, not a hash
      of the whole claim: `created_at` and provenance bookkeeping move without
      changing what a claim asserts, and staleness that fires on those would
      cry wolf until nobody read it. A test pins both directions, including
      that an identical rewrite does *not* stale anything, which matters
      because re-ingestion rewrites claims constantly.

      Writing the tests found the hole: `supersede_claim` writes through raw
      SQL rather than `put_claim`, so it bypassed the hook entirely. A
      superseded claim is the clearest case for staling work written from it,
      and it was the one case that missed. Fixed and pinned.
      `recheck_synthesis_staleness` also re-derives the invariant from the data
      alone, for the paths no hook can see: a restore, an external edit, a bug
      in the hook.

**Not built: the W3 re-synthesis workflow.** Staleness detection is the gated
half and is done; *generating* a replacement synthesis needs a model, a prompt,
and an eval to know whether the output is worth trusting. Shipping generation
without measuring it is the thing this project has repeatedly refused to do,
and there is no reason to start here. `list_syntheses --stale` is the queue W3
would consume.

---

## Phase 10: Polish, testing, documentation, release *(complete)*

**Scope**
- Coverage, performance, error-message quality, deployment docs,
  end-user documentation, backup/restore for derived stores.

**Gate**
- [x] **Fresh-machine setup works from documentation alone, with no paid API.**
      Verified 2026-09-07 by installing into an empty virtualenv and following
      the docs on a vault that did not exist beforehand. It did not work the
      first time, and the four things that stopped it are the useful output of
      this gate:

      1. **`forge --version` was `No such option`.** Found in the first minute,
         which is roughly how long it takes a new user to find it.
      2. **A vault had to contain `.git`.** A plain folder of Markdown notes is
         not a git repository, so the first documented command failed before
         reaching anything Forge does. `.obsidian` and `.forge` are now
         accepted as equally deliberate markers. The guarantee is unchanged: an
         *unmarked* directory is still not a vault, because silently indexing
         the wrong directory is the failure that rule exists to prevent.
      3. **`bootstrap` reported `edges: 0` and said nothing about why.** The
         one wikilink in the test vault, `[[Vector Databases]]` against
         `vector-databases.md`, is a `renamed_candidate`, and acting on it
         would be the engine guessing what the user meant. The conservatism is
         right; the silence was not. Bootstrap now reports what it declined and
         points at `forge diagnostics`.
      4. **The documented install was `pip install -e ".[dev]"`**, which is the
         contributor path, not the user path. There is now a Quickstart written
         from the verified transcript.

      A fifth was found by re-running the walkthrough after building the gate
      itself: `forge backup` raised `NameError: utc_now` on its first line,
      because a lint autofix had removed the import as unused before the
      command that needed it existed. Every unit test passed, because they call
      `create_backup` directly. Only running the command found it, which is the
      whole argument for the clean-room exercise. There is now a CLI-level test.
- [x] **Full rebuild from Markdown reproduces the derived model.** Two rebuilds
      from the same vault into two empty stores produce identical concept ids,
      link ids and counts. What that really tests is whether anything
      non-deterministic leaked into the derivation: a timestamp in an id, a set
      iterated in hash order. Re-running into a populated store is idempotent
      rather than additive.
- [x] **Backup/restore covers state Markdown cannot express (ADR-001 R5).**
      `forge backup` / `forge restore`, using SQLite's own backup API so the
      copy is transactionally consistent under a concurrent writer rather than
      a file copied out from under a live WAL.

      A test states the gap before closing it: a rebuild is *expected* to lose
      a question the user asked, and asserted to. What a rebuild cannot
      reproduce is a proposal you rejected and why, the revision log, your
      questions, syntheses and their staleness, and the wording of a
      model-derived claim.

      Restore refuses three things rather than guessing, because each destroys
      knowledge quietly: a backup whose contents do not match its manifest
      checksum, one written by a newer schema than the build understands, and
      an existing store without `--force`. It also clears stale `-wal` sidecars,
      without which a restore appears to succeed and then serves the old data
      back.

      **`docs/cli.md` said `.forge/` "can be deleted at any time: `forge index`
      rebuilds it".** That was wrong, and R5 is the risk it was understating.
      Corrected.

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
