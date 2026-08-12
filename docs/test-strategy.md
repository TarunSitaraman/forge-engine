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
genuinely needing a live model is marked `requires_model` and skipped — nothing
currently is.

**2. Real corpus over synthetic fixtures.** Integration tests run against the
actual 630-file vault, and PDF tests run against real PDFs. Synthetic examples
cannot demonstrate that the parser survives 555 code blocks of Python literals,
that link resolution copes with three real stem collisions, or that a truncated
PDF fails cleanly. Ideal inputs prove very little.

**3. Fixtures reproduce real defect shapes verbatim.** Where a small vault is
needed, its files carry the exact malformed strings found in the corpus —
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
  integration/
    test_real_corpus.py            everything, against the real vault
    test_pipeline_and_cli.py       end-to-end pipeline + every CLI command
```

`reset_call_counter` is autouse, so **every test starts with `CALLS.count == 0`**
and "made no model calls" is assertable anywhere.

---

## What the exit criteria are proved by

| Criterion | Test |
|---|---|
| Corpus unchanged | `test_indexing_does_not_modify_the_vault` (git porcelain diff), `test_no_markdown_file_mtime_changes`, `test_cli_never_writes_to_the_vault` (byte comparison) |
| Deterministic indexing | `test_same_corpus_produces_same_index` (fingerprint equality), `test_discovery_order_is_stable` |
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
with `resolved_path is None` — the engine must never pick one of two canonical
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
| `](../personal-agent/)` directory links reported broken — 62 false positives | `test_directory_link_is_not_broken` |
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
- **Concept/claim extraction *quality*.** The extraction path is fully tested for correctness (schemas, grounding, provenance, failure modes) against a scripted provider, but whether a real local model extracts *good* concepts is unmeasured — no model was reachable here.
- **Performance.** The corpus indexes in ~0.7s. A threshold test would be noise
  at this size.
- **`logging.py` internals** (57% covered). Configuration plumbing; its one
  real bug — a captured `sys.stderr` surviving into a closed stream — was found
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

## Coverage

92% overall. `llm/ollama.py` sits at 64% because its success paths need a live
server; every other module is 90%+, and `domain/` — where the invariants live —
is 98–100%.

---

## Related

- [CLI usage](./cli.md)
- [Phase 1 implementation architecture](./architecture/phase-1-implementation.md)
