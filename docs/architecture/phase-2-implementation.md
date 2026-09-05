# Phase 2: External Knowledge Ingestion & Evidence Foundation

*What was built, how it behaves, and what it deliberately refuses to do. Describes the implementation as it exists.*

**Status:** implemented, **Tests:** 435 passing, 92% coverage, **LLM required:** no
**Validate:** `bash scripts/validate_phase2.sh` (16/16)

---

## 1. What Phase 2 is

External sources can now enter the canonical knowledge model with full
provenance, and anything the engine *believes* about them is a proposal
awaiting a human decision.

Delivered:

- **PDF and Markdown source adapters** behind one acquisition protocol
- **Deterministic chunking** into spans carrying page, section, line, and
  character offsets
- **Derivation-key caching** so unchanged work is never redone
- **Optional LLM extraction** of concept and claim candidates, with strict
  schemas and a grounding check
- **Concept candidate matching** that produces candidates and never merges
- **A proposal system** with approval state, safety classification, and
  flag-gated, reversible write-back
- **Lexical retrieval** with metadata filters and optional semantic re-ranking

Not built, deliberately: contradiction detection, synthesis, autonomous
research, graph or vector databases, web frontend, Obsidian plugin, MCP.

---

## 2. Ingestion architecture

```
 EXTERNAL SOURCE (.pdf / .md)
        |
   AdapterRegistry.for_path
        |
   SourceAdapter.acquire         deterministic — no LLM
        |  -> AcquisitionResult { text, blocks[], hash, metadata }
        |
   change detection (content hash vs stored Source)
        |-- unchanged --> STOP. zero work, zero tokens.
        |
   Source + Document registration
        |
   chunking.build_spans          deterministic — structure-aware
        |  -> Span[] with page / heading_path / lines / char offsets
        |
   persist (spans + FTS index, one transaction)
        |
   === optional, everything above already succeeded ===
        |
   CandidateExtractor.extract    strict schemas, grounding check
        |  -> ConceptCandidate[], ClaimCandidate[]
        |
   ConceptMatcher.match          NEW / MATCH_CANDIDATE / AMBIGUOUS
        |
   ProposalService.create        PENDING, awaiting a human
```

**The optional stage is genuinely optional.** Extraction runs after the
deterministic pipeline has already committed its spans, so a missing model, an
unreachable Ollama, or a malformed response can never cost you the ingest.
`test_extraction_failure_does_not_lose_the_ingest` asserts exactly this.

### Package layout added

```
engine/forge/
  sources/       base.py (protocol) · pdf_adapter.py · markdown_adapter.py · registry.py
  ingestion/     pipeline.py · chunking.py · derivation.py · report.py
  extraction/    extractor.py · schemas.py · prompts.py
  matching/      matcher.py
  proposals/     service.py · metadata_repair.py · apply.py
  retrieval/     search.py
  embeddings/    base.py · ollama_embeddings.py
```

Adapters do acquisition **only**, no concept extraction, no relationship
discovery, no LLM orchestration. That is what makes a future web or repository
adapter a self-contained addition.

---

## 3. PDF parser behaviour

`pypdfium2` (Apache/BSD, chosen in Phase 0 over PyMuPDF to avoid the AGPL
question). No OCR.

| Input | Outcome |
|---|---|
| Normal text PDF | `INGESTED` with blocks, pages, offsets |
| Multi-page | `INGESTED`, page boundaries preserved |
| Page with no text | `INGESTED` + warning naming the page |
| Image-only / scanned | **`OCR_REQUIRED`**, never a silent empty success |
| Malformed / truncated | `PARSE_FAILED` with the parser's reason |
| Not a PDF | `PARSE_FAILED` |
| Missing file | `NOT_FOUND` |

Failures are **returned as values, not raised**, so ingesting a directory
reports per-file outcomes instead of aborting on the first bad file.

### What is exact, and what is a heuristic

Stated precisely because over-claiming here would poison every downstream
citation:

- **Page numbers: exact.** Taken from document structure.
- **Character offsets and line numbers: exact**, relative to the text this
  parser extracted. `test_char_offsets_index_into_the_extracted_text` verifies
  every block resolves back to its own characters.
- **Headings: a heuristic.** PDF has no heading concept. Headings are inferred
  from per-character font size against the document's median body size
  (`HEADING_SIZE_RATIO = 1.15`), and `metadata["heading_detection"]` records
  `"font-size heuristic"` so a consumer knows what it is looking at.

**Content hash is computed over the extracted text, not the raw bytes.**
Re-saving a PDF changes its bytes constantly (producer strings, timestamps);
hashing bytes would report content changes that did not happen.

---

## 4. Span model

Phase 1's `Span` gained four **optional** fields, `page`, `page_span`,
`char_start`, `char_end`. Phase 1 Markdown spans set none of them, `make_id` is
unchanged, and existing span identities are stable.

```python
Span(
  page=2, page_span=None,
  heading_path=("Retrieval Augmented Generation", "Chunking Strategy"),
  start_line=4, end_line=6, char_start=130, char_end=255,
  locator="p.2 L4-L6",
)
span.citation()   # "p.2 | Retrieval Augmented Generation > Chunking Strategy"
```

`page` is `None` for Markdown rather than faked as page 1, absent is
information.

Token offsets were deliberately not implemented: they would bind provenance to
a tokenizer version, and character offsets already answer the question.

### Chunking

`chunking.build_spans`, three rules in priority order:

1. **Never cross a structural boundary.** Headings and pages end a chunk.
2. **Never split a sentence** when a boundary exists; an unsplittable oversized
   block is returned whole, because an oversized span is an inefficiency while
   a truncated one is a provenance error.
3. **Deterministic identity.** Content-identical documents produce identical
   span ids.

Lone headings are merged forward, so a bare title never becomes a span with a
location and no content. Not tuned for a vector database: no overlap windows,
no token targets. Those are retrieval-quality decisions with nothing yet to
measure against.

---

## 5. Change detection and cost control

The hash comparison happens **before** any parsing or persistence. An unchanged
source short-circuits with zero tokens and zero writes.

For everything expensive, the **derivation key**:

```
sha256( source content hash
      + processor version
      + model id
      + prompt version
      + schema version )
```

Change any component and the key changes and the work is redone; change none
and the cached result is reused. `test_every_component_invalidates` asserts all
five independently.

**Only useful outcomes are cached.** A `provider_unavailable` result is never
stored, caching it would make a run permanently model-free once Ollama came
back up.

| Scenario | Behaviour |
|---|---|
| Unchanged source | `UNCHANGED`, 0 LLM calls, 0 writes |
| Modified source | new Document version; prior version **retained** |
| Deleted source | derived state invalidated, `INVALIDATE` revision written |
| Prompt/model/schema changed | cache miss, re-extracted |

### A data-loss bug found and fixed here

Phase 1 used `INSERT OR REPLACE` for sources and documents. SQLite implements
REPLACE as **DELETE-then-INSERT**, which fires `ON DELETE CASCADE`, so
re-ingesting a *modified* source silently destroyed all of its prior documents
and spans. That is precisely the historical provenance Phase 2 must preserve.

All writes now use `INSERT … ON CONFLICT(id) DO UPDATE`.
`test_modification_creates_a_new_document_version` locks it down.

---

## 6. LLM extraction

Optional, provider-agnostic, off unless `--extract` is passed.

**Strict schemas.** Every response model is `extra="forbid"` with required,
non-empty primary collections. The Phase 1 spike taught this: schemas whose
fields all have defaults accept `{}`, so a model that produced nothing scores as
a success. Here that would silently write empty knowledge.

**Malformed output is a failure, not a repair opportunity:**

| Outcome | Meaning |
|---|---|
| `SUCCEEDED` | every attempted span produced valid output |
| `PARTIAL` | some spans succeeded, some failed |
| `FAILED` | no span produced valid output |
| `SKIPPED_NO_PROVIDER` | no model available, ingestion still succeeded |
| `SKIPPED_CACHED` | unchanged source, nothing to do |

**The grounding check.** Every claim must quote its span verbatim. The quote is
checked against the actual span text (exact substring, then a word-overlap
fallback for whitespace/case normalization at `QUOTE_GROUNDING_THRESHOLD = 0.6`).
A quote that is not there means the claim is **dropped and the drop reported**.

A dropped claim does *not* fail the span, the filter rejecting a fabricated
quote is the system working, and treating it as failure would discard the
span's good candidates and prevent caching a correct result.

**Tier discipline.** `extraction_provenance` raises if asked for `SOURCE_FACT`
or `USER_ASSERTION`. A model cannot assert source facts, and it cannot speak as
the user. The request is rejected rather than silently downgraded, quietly
fixing it would hide the bug.

Cost control: `max_spans` (default 12) caps calls per document, longest spans
first. The floor for "worth a call" is 40 characters, an earlier 120-character
threshold silently discarded short but meaningful sections, such as a two-line
definition of a data structure, which then never appeared as a candidate at all.

---

## 7. Concept matching

Three outcomes and no fourth: `NEW_CONCEPT`, `MATCH_CANDIDATE`, `AMBIGUOUS`.
There is deliberately no `MERGED`.

Signals, in descending confidence:

| # | Signal | Result |
|---|---|---|
| 1 | **Vault collision** | `AMBIGUOUS`, dominates everything |
| 2 | Exact canonical name | `MATCH_CANDIDATE` |
| 3 | **Already proposed** from another source | `MATCH_CANDIDATE` |
| 4 | Registered alias | `MATCH_CANDIDATE` |
| 5 | Normalized name | `MATCH_CANDIDATE` |
| 6 | Lexical ≥ 0.86 / embedding ≥ 0.88 | `MATCH_CANDIDATE`, or `AMBIGUOUS` on a near tie |
| - | nothing | `NEW_CONCEPT` |

**Signal 1 exists because of this corpus.** `Heap`, `Binary Search`, and `Trie`
each have two canonical homes in the vault: a pattern and an
algorithm/data-structure, accounting for 180 of its 282 unresolved links. Any
matcher that picks one is wrong half the time and never says so. So the
ambiguity index is built from the **vault filesystem** (not from ingested
sources, which may not include those files at all), and a colliding name is
ambiguous before any scoring runs.

**Signal 3 exists because Phase 2 creates no Concepts.** Extraction produces
proposals, so `list_concepts()` is always empty and every source would otherwise
rediscover the same concept as brand new. Matching against pending proposals is
what makes "existing concepts detected" true before any approval workflow runs.

`MatchResult.best` returns `None` unless the kind is `MATCH_CANDIDATE`, so a
caller cannot accidentally treat an unresolved collision as resolved.

Near ties (within `AMBIGUITY_MARGIN = 0.05`) are ambiguous: a 0.87-vs-0.86 win
is noise, not a decision.

---

## 8. Proposal system

| Field | Purpose |
|---|---|
| `type` | `metadata_repair`, `new_concept`, `concept_match`, `new_claim` |
| `status` | `pending` → `approved` / `rejected` / `superseded` |
| `safety` | `deterministic_verified`, `deterministic_unverified`, `model_generated`, `ambiguous` |
| `operation` | action, target, before, after, details |
| `evidence_span_ids` | required for anything model-derived |
| `provenance` | full Phase 1 provenance record |

**Safety is derived, not asserted.** The domain layer rejects a model-derived
proposal claiming `DETERMINISTIC_VERIFIED`, and rejects a model-derived proposal
citing no evidence. "The model was confident" can never be mistaken for
"software checked this".

**Identity is deterministic**, derived from what the proposal *is*. Without
that, every re-ingest would resurrect proposals the user already rejected as
fresh `PENDING` ones. `test_rejected_proposals_are_not_resurrected` covers it.

Status transitions are validated (a proposal cannot be approved twice) and every
change writes a `Revision`.

### The 283 metadata repairs

Phase 1's verified frontmatter repairs are surfaced one proposal **per line**,
not per file: a file with two malformed fields is two independently reviewable
decisions. Each shows the file, current and proposed metadata, the reason, the
affected wikilinks, and its safety class. All 283 classify as
`DETERMINISTIC_VERIFIED`; none is applied.

### Write-back: off by default

`forge proposals approve <id>` records a decision and **writes nothing**. With
`--apply`:

1. **Backup first** into `.forge/backups/<timestamp>/<path>`, reversible.
2. **A `Revision` is recorded** against the Source with before and after.
3. **The exact diff is shown.**
4. **Refusals are conservative:** not approved, not `DETERMINISTIC_VERIFIED`,
   unsupported operation, file missing: all refused with a reason.
5. **Staleness is checked.** The target line must still match what the proposal
   recorded; if the file changed underneath, the application is refused rather
   than clobbering the user's edit.
6. **Only the named file is touched**, `test_only_the_named_file_is_touched`.

---

## 9. Retrieval

Lexical (SQLite FTS5/BM25) plus metadata, source, page, and heading filters, and
concept/document/span lookup. Every hit carries the full chain back to its
source; a hit that cannot resolve to a document and source is a bug, not a
degraded result.

No chatbot, no conversational interface, no generation over retrieved text.

FTS5 syntax never leaks from user input: tokens are quoted and embedded quotes
doubled. Without that, a query like `quote"inside` produced
`sqlite3.OperationalError: unterminated string`: the same shape as SQL
injection, in a query language the caller never sees.

### Embeddings and the degradation mode

Optional throughout, documented rather than silent:

| With embeddings | Without |
|---|---|
| Lexical + semantic re-rank | Lexical only |
| Semantic concept candidates | Exact / alias / normalized / lexical only |

`SearchService.degradation_note()` reports which state you are in, and the CLI
prints it. An embedding backend that fails mid-query degrades to the lexical
ordering rather than raising. No Qdrant, vectors live in SQLite, where
brute-force cosine over a few thousand spans is milliseconds.

---

## 10. Observability

Structured logs plus typed `SourceReport` / `IngestionReport` records, exposing
per source: status, duration, spans, pages, extraction status, concepts, claims,
proposals, LLM calls, and cache hits/misses/writes. `--json` on every command.

Not an observability platform.

---

## 11. Known limitations

| # | Limitation |
|---|---|
| L1 | **No local model was available in this environment**, so extraction was validated against a scripted provider through the real `LLMProvider` interface. Real-model behaviour remains unmeasured (see [the spike](../research/local-model-capability-spike.md)) |
| L2 | No OCR. Image-only PDFs report `OCR_REQUIRED` and stop |
| L3 | PDF heading detection is a font-size heuristic; PDFs with uniform typography yield no headings |
| L4 | Chunking is untuned for retrieval quality, no labelled set exists yet |
| L5 | Extraction caps at `max_spans` per document, so long PDFs are sampled, not exhaustively processed |
| L6 | Nothing creates a `Concept` or `Claim` entity yet; Phase 2 produces proposals. Accepting a proposal into the model is Phase 3 |
| L7 | Embedding re-rank weighting (0.5/0.5) is arbitrary and untuned |

---

## Related

- [CLI usage](../cli.md), [Test strategy](../test-strategy.md)
- [Phase 1 implementation](./phase-1-implementation.md)
- [Canonical knowledge model](../knowledge-model/canonical-model.md)
- [ADR-001](../decisions/001-forge-knowledge-os.md), D2 segregated write-back
- [Roadmap](../roadmap.md)
