# Test Strategy

*435 tests, 92% coverage, no model required. What is tested, why it is tested that way, and what is deliberately not tested.*

```bash
pytest                                    # full suite, offline, ~20s
pytest tests/unit -q                      # fast
pytest --cov=forge --cov-report=term      # coverage
```

---

## Principles

**1. The suite runs offline with no model.** `MockProvider` is the default.
If the suite needed Ollama it would be skipped in CI and would rot. Anything
genuinely needing a live model is marked `requires_model` and skipped. Nothing
currently is.

**2. Real corpus over synthetic fixtures.** Integration tests run against the
actual 630-file vault, and PDF tests run against real PDFs. Synthetic examples
cannot demonstrate that the parser survives 555 code blocks of Python literals,
that link resolution copes with three real stem collisions, or that a truncated
PDF fails cleanly. Ideal inputs prove very little.

**3. Fixtures reproduce real defect shapes verbatim.** Where a small vault is
needed, its files carry the exact malformed strings found in the corpus:
copied, not invented:

```yaml
related: [[Pattern Index]], [[Template Index]]      # FM001, from DFS Cheat Sheet.md
related: [[[Python - Greedy]], [[Heap]]]            # FM002, from Greedy.md
related: [[Binary Search Tree]], [[Tree Traversal]  # FM008, from AVL Tree.md
```

**4. Invariants are tested where they are enforced.** Provenance tests
construct domain objects directly rather than going through a service. If a
violating object can be built, it can be persisted, so the domain layer is the
right place to assert.

**5. Bugs found during development become tests.** Every false positive listed
in §"Regression tests" is a mistake made and then locked down.

---

## Running the corpus tests

42 integration tests run against the Markdown vault the engine was built
against. That vault is a **separate, private repository**, so by default these
tests **skip** and the suite reports `1,205 passed, 42 skipped`.

```bash
python -m pytest tests                              # 1,205 passed, 42 skipped
FORGE_TEST_VAULT=/path/to/forge python -m pytest tests   # 1,247 passed
```

`FORGE_TEST_VAULT` takes a checkout of that vault. It is deliberately **not**
`FORGE_VAULT_PATH`: these tests assert against that specific corpus, file
counts, particular paths, the recorded collision decisions, so aiming them at
some other vault would fail on content rather than on a defect, and the two
variables must not be confusable. A path that is set but is not a vault raises
rather than skipping: the caller asked for something specific, and silently
testing nothing is how coverage disappears.

Skipping is the honest default, not a weakened suite. What it costs is real
though. These are the tests that validate the engine on material that actually
drifted, so run them with the vault before trusting a change to indexing, link
resolution, or diagnostics.

## Layout

```
tests/
  conftest.py                      fixture vault, real vault, call-counter reset
  fixtures/vault/                  synthetic vault with real defect shapes
  unit/
    test_frontmatter.py            parsing, diagnostics, repair safety
    test_markdown.py               code-fence masking, links, headings
    test_links.py                  resolution, classification, hard cases
    test_provenance.py             floor rule, derivation, evidence, link types
    test_revision_and_model.py     revision shapes, supersession, entity validation
    test_storage.py                protocol conformance, enforcement, history
    test_change_detection.py       hashing, change classification
    test_llm_provider.py           abstraction, structured output, failure modes
    test_spike.py                  spike harness honesty
    test_activation.py             proposal activation, idempotency, identity, gating
    test_graph.py                  bounded traversal, evidence chains, integrity
    test_evaluation.py             retrieval metrics, dataset, embeddings
    test_evolution.py              narrowing, assessment, impact, proposals, activation
    test_providers.py              ollama, cloud, mock; credentials; no silent downgrade
  integration/
    test_real_corpus.py            everything, against the real vault
    test_pipeline_and_cli.py       end-to-end pipeline + every CLI command
    test_phase2_ingestion.py       ingestion, cost control, ambiguity, CLI
    test_phase3_activation.py      the activation loop through the CLI, batch ops
    test_phase4_workflow.py        LangGraph interrupt/resume, routing, CLI
```

`reset_call_counter` is autouse, so **every test starts with `CALLS.count == 0`**
and "made no model calls" is assertable anywhere.

---

## What the exit criteria are proved by

| Criterion | Test |
|---|---|
| Corpus unchanged | `test_indexing_does_not_modify_the_vault` (git porcelain diff), `test_no_markdown_file_mtime_changes`, `test_cli_never_writes_to_the_vault` (byte comparison) |
| Deterministic indexing | `test_same_corpus_produces_same_index` (fingerprint equality), `test_discovery_order_is_stable` |
| Deterministic indexing *across platforms* | Not a test, **observed**. The same commit indexed on Windows (CRLF checkout, ASUS) and Linux (LF) produced the identical fingerprint `600dd093…` over 642 files, 2026-08-17. `text_hash` normalizes CRLF/CR to LF before hashing for exactly this reason, so a checkout's line endings are not a content change. Worth re-checking whenever hashing or discovery changes: a platform-dependent fingerprint would make derived state non-shareable and provenance non-comparable between machines. |
| Unresolved links reported | `test_unresolved_links_are_reported` |
| Malformed frontmatter reported | `test_frontmatter_defects_are_found`, `test_every_parse_error_has_a_verified_repair` |
| Zero LLM calls on re-index | `test_reindexing_unchanged_corpus_costs_zero_llm_calls` |
| Single-file change isolation | `test_editing_one_file_marks_only_that_file`, `test_editing_a_file_reprocesses_only_it` |
| Provenance floor enforced | `TestFloorRule` (7 tests) |
| Revision history works | `TestSupersession`, `TestRevisionShapes` |
| Provider abstraction usable | `TestProtocol`, `TestStructuredOutput` |
| Runs without a paid API | the entire suite |

---

## Known-hard cases

Preserved as required, and they are genuinely hard rather than ceremonial.

**Graph Traversal / DFS / BFS** must never collapse onto one another:

```python
def test_dfs_and_bfs_resolve_to_their_own_files(real_index):
    # links from Graph Traversal.md resolve to the correct distinct files
```

Also asserted: the three files index distinctly and carry three distinct
hashes.

**Stem collisions.** `Heap`, `Binary Search`, and `Trie` each exist twice in
the real corpus (pattern vs algorithm/data-structure), producing 180 of the 282
unresolved link occurrences. The tests assert these are classified `AMBIGUOUS`
with `resolved_path is None`, the engine must never pick one of two canonical
homes on the user's behalf.

**Code-fence hazard, on real data:**

```python
def test_python_literals_do_not_become_links(real_index):
    # no link target anywhere in the corpus is numeric
```

---

## Regression tests for bugs made during development

| Bug | Test |
|---|---|
| `%20` compared against the filesystem without decoding (Phase 0 false positive) | `test_url_encoded_link_resolves` |
| `](../personal-agent/)` directory links reported broken, 62 false positives | `test_directory_link_is_not_broken` |
| CRLF vs LF hashing as a content change | `test_crlf_and_lf_hash_identically` |
| Deterministic-id separator collision `("a","bc")` vs `("ab","c")` | `test_deterministic_id_separator_is_unambiguous` |
| Spike schemas accepting `{}` as success (would inflate every result) | `test_schema_violations_are_recorded_not_hidden` |
| `ProvenanceViolation` swallowed by pydantic's `ValidationError` | `test_violation_is_not_a_pydantic_validation_error` |
| Truncated wikilink `[[X]` leaving 18 files unrepairable | `TestTruncatedWikilinks` |

---

## Deliberately not tested

- **Live Ollama inference.** Not available in CI. The adapter's failure path is
  tested against a dead port, which is the state most developers hit first; its
  success path is covered by the spike harness against the mock. This is the
  suite's largest genuine gap and is stated rather than papered over.
- **Concept/claim extraction *quality*.** The extraction path is fully tested for correctness (schemas, grounding, provenance, failure modes) against a scripted provider, but whether a real local model extracts *good* concepts is unmeasured, no model was reachable here.
- **Performance.** The corpus indexes in ~0.7s. A threshold test would be noise
  at this size.
- **`logging.py` internals** (57% covered). Configuration plumbing; its one
  real bug, a captured `sys.stderr` surviving into a closed stream, was found
  by the CLI tests and fixed at the root.

---

## Phase 2 additions

| Module | Tests |
|---|---|
| `tests/unit/test_pdf_adapter.py` | Real PDF fixtures: normal, multipage, empty pages, image-only, malformed, non-PDF |
| `tests/unit/test_phase2_units.py` | Markdown adapter, registry, chunking, derivation keys, extraction, embeddings |
| `tests/unit/test_proposals.py` | Lifecycle, safety classification, matching, write-back |
| `tests/unit/test_retrieval.py` | Lexical search, filters, semantic path via a fake embedding provider |
| `tests/integration/test_phase2_ingestion.py` | End-to-end ingestion, cost control, overlap, ambiguity, CLI |

**PDF fixtures are real PDFs**, generated from raw PDF syntax by
`scripts/make_pdf_fixtures.py` and committed. A PDF parser tested against mocks
tests nothing. The fixtures deliberately include the failure cases: an
image-only page (`OCR_REQUIRED`), a truncated file, and a text file with a
`.pdf` extension.

**The semantic path is tested without a model.** `scripted_extractor` and
`FakeEmbeddings` drive the production code through the real provider
interfaces. Everything except the model itself is the shipped path.

### Phase 2 regression tests for bugs found during development

| Bug | Test |
|---|---|
| `INSERT OR REPLACE` cascading DELETE destroyed prior documents and spans on re-ingest | `test_modification_creates_a_new_document_version` |
| FTS5 query injection: `quote"inside` raised `unterminated string` | `test_operator_characters_are_neutralized`, `test_embedded_quotes_are_doubled` |
| PDF line cursor advanced on skipped blank lines, drifting every citation | `test_line_numbers_index_into_the_extracted_text` |
| Directory links and `%20` targets reported broken (Phase 1) | `test_directory_link_is_not_broken`, `test_url_encoded_link_resolves` |
| A dropped ungrounded claim failed the whole span and blocked caching | `test_ungrounded_quote_is_dropped_and_reported` |
| 120-char span floor silently discarded short meaningful sections | `test_short_spans_are_still_extracted` |
| `Settings.load` kwargs collided with `**overrides` | exercised by every `Settings.load(state_dir=...)` call |

---

## Phase 3 additions

| Module | Tests |
|---|---|
| `tests/unit/test_activation.py` | Concept/claim activation, origin tracking, the approve→activate→reindex→activate idempotency cycle, FAILED reporting, ambiguity protection, identity round-trip, relationship gating |
| `tests/unit/test_graph.py` | Neighbour direction, type filters, depth and node-budget bounds, cycle termination, shortest path, evidence chains, metrics, all nine integrity codes |
| `tests/unit/test_evaluation.py` | Hand-computed Recall@k / Precision@k / MRR, dataset validation, embedding determinism and cache invalidation, evaluator degradation |
| `tests/integration/test_phase3_activation.py` | The loop through the CLI, batch approval guards, identity commands, embeddings, `retrieval-eval`, and a `git status` proof that the real vault is untouched |

Three properties are asserted repeatedly rather than once, because they are the
ones a future change is most likely to break quietly:

- **Metrics are checked against hand-computed values**, never against whatever
  the implementation currently returns. A measuring instrument that agrees with
  itself measures nothing.
- **Degradation is asserted as hard as the happy path.** A missing embedding
  model must produce an explicit, reported absence: `test_semantic_is_skipped_with_an_explicit_note_when_unavailable`
  fails if the runner silently falls back and looks successful.
- **Diagnostics never repair.** `test_integrity_check_repairs_nothing` counts
  entities before and after a check that finds problems.

### Phase 3 regression tests for bugs found during development

| Bug | Test |
|---|---|
| `concepts.canonical_name UNIQUE` made namespaced concepts impossible | `TestIdentityConfig` round-trip; schema v3 migration |
| Migration ran the schema script before reshaping the table (`no such column: namespace`) | exercised by every v2→v3 store open |
| A resolved collision still refused to activate ("match target is not a canonical concept") | `TestAmbiguityProtection` |
| The ambiguity index double-counted paths, reporting "4 canonical homes" for Heap | `test_a_model_supplied_namespace_is_ignored`, demo step 16 |
| `retrieval-eval` resolved its default dataset relative to the working directory | `test_eval_runs_the_labelled_set_and_reports_metrics` |

## Phase 4 additions

| Module | Tests |
|---|---|
| `tests/unit/test_evolution.py` | Candidate narrowing and its selectors, claim retrieval bounds, all five assessment classifications, grounding rejection, every failure mode, cache hit and four invalidation paths, impact precedence, proposal generation per class, and the three activation behaviours |
| `tests/unit/test_providers.py` | Provider selection, remote Ollama, cloud wire formats (both), credential handling, and the no-silent-downgrade rule |
| `tests/integration/test_phase4_workflow.py` | The full loop through LangGraph: interrupt, checkpoint, resume across a simulated process restart, conditional routing, provider mismatch on resume, idempotency, observability, and the CLI |

Four properties are asserted repeatedly rather than once:

- **Deterministic steps spend nothing.** `CALLS.count == 0` after narrowing,
  retrieval, impact classification, and activation. "Deterministic first" is a
  claim that decays silently if nothing checks it.
- **Ungrounded output is rejected, never repaired.** Two shapes are tested: an
  invented span id, and a *real* span the model was not shown. The second is
  the subtle one, citing real-but-unshown evidence is still fabrication with
  respect to the question asked.
- **A failure never becomes an empty success.** Three provider failures, each
  asserted not to report success.
- **Nothing is overwritten.** Every activation test also checks the prior state
  is still retrievable.

**CI is fully offline.** Every model call goes through `MockProvider` over the
real `LLMProvider` interface, and the cloud provider's HTTP is stubbed with
`httpx.MockTransport`: no test touches a network.

### Phase 4 regression tests for bugs found during development

| Bug | Test |
|---|---|
| `AssessmentRecord.to_dict` omitted `derivation_key`, so state failed to round-trip on resume | `test_run_round_trips_through_storage` |
| Disputed claims were excluded from retrieval, faking idempotency and making DISPUTED terminal | `test_disputed_claims_are_still_examined` |
| Resuming with nothing decided completed the run, treating a resume as consent | `test_resuming_without_deciding_pauses_again` |
| The no-candidates path skipped `classify_impact`, losing the NEW_KNOWLEDGE finding | `test_unrelated_evidence_never_reaches_the_model` |
| A provider-change warning was discarded when state was projected onto the run | `test_a_provider_change_can_be_accepted_explicitly` |

---

## Coverage

89% overall across 737 tests. `llm/ollama.py` and `embeddings/ollama_embeddings.py`
are lowest because their success paths need a live server; `domain/`, where the
invariants live, is 98-100%.

---

## Related

- [CLI usage](./cli.md)
- [Phase 1 implementation architecture](./architecture/phase-1-implementation.md)
- [Phase 3 implementation architecture](./architecture/phase-3-implementation.md)
- [Retrieval baseline](./research/retrieval-baseline.md)
