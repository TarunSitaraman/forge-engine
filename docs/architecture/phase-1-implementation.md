# Phase 1: Implementation Architecture

*What was actually built, how it behaves, and where the extension points are. Describes the implementation as it exists, not as it was planned.*

**Status:** implemented, **Tests:** 243 passing, 95% coverage, **LLM required:** no

---

## 1. What Phase 1 is

The canonical knowledge foundation. Concretely:

- a typed domain model with **structurally enforced provenance**,
- append-only **revision tracking**,
- a **deterministic corpus indexer** that never calls a model,
- **safe frontmatter parsing** that diagnoses defects and proposes repairs
  without applying them,
- **hash-based change detection** so unchanged content costs nothing,
- a **provider abstraction** with an offline mock and an Ollama adapter,
- a **CLI** that demonstrates all of it.

**Not** in Phase 1: concept extraction, claim extraction, embeddings,
retrieval, graph queries, contradiction detection, LangGraph workflows.
Those are Phases 2-5. The store has `concepts` and `claims` tables and
they are deliberately empty after indexing, a fact asserted by
`test_no_claims_are_created_during_indexing`.

---

## 2. Package layout

```
engine/forge/
  config.py        typed settings from env, validated at startup
  logging.py       structlog, run_id-correlated
  ids.py           deterministic + time-ordered identity, content hashing
  domain/          THE MODEL — no I/O, no storage, no LLM
    enums.py         closed vocabularies + tier strength ordering
    provenance.py    Provenance + the floor rule (enforced here)
    entities.py      Source, Document, Span, Concept, Claim, *Link
    revision.py      append-only history records
    validation.py    cross-entity invariants
  parsing/         deterministic text processing
    markdown.py      headings, wikilinks, tags, code-fence masking
    frontmatter.py   safe YAML + diagnostics + repair proposals
    links.py         link resolution and classification
  corpus/          deterministic corpus analysis (no LLM reachable)
    indexer.py       walk, hash, parse, span
    model.py         index records
    diagnostics.py   frontmatter + link reports
    stats.py         computed statistics
    conventions.py   machine-readable convention conflict
    pipeline.py      index -> detect changes -> persist -> report
  storage/         protocols + SQLite implementation
  llm/             provider abstraction, mock, Ollama
  spike/           capability spike harness
  cli/             typer CLI
```

**The dependency rule:** `domain/` imports nothing from `storage/`,
`llm/`, or `corpus/`. `corpus/` cannot reach `llm/`. That second one is
not a convention. It is why "indexing makes zero LLM calls" is a
structural property rather than a promise.

---

## 3. The domain model

Nine entities, exactly as approved. `Contradiction`, `Synthesis`,
`Question`, and `KnowledgeGap` are **not** implemented.

| Entity | Identity | Notes |
|---|---|---|
| `Source` | `blake2b(locator)` | Identity follows path, not content, so edits don't create new sources |
| `Document` | `blake2b(source_id, content_hash)` | New content ⇒ new document; re-parsing identical content ⇒ same id |
| `Span` | `blake2b(document_id, ordinal, locator)` | The provenance atom |
| `Concept` | `blake2b(casefold(name))` | Case-insensitive canonical identity |
| `Claim` | `blake2b(statement, source_ref)` | |
| `EvidenceLink` | `blake2b(claim, span, relation)` | Reified Claim→Span |
| `ClaimLink` | `blake2b(from, to, type)` | Reified, carries own provenance |
| `Provenance` | embedded, frozen | Not a table |
| `Revision` | time-ordered random | Append-only |

### Why identity is deterministic

Re-indexing an unchanged vault must produce a byte-identical index. Random
ids make that impossible. Every id derived from corpus content is a
BLAKE2b digest over namespaced parts, with `\x00` separators so
`("a","bc")` cannot collide with `("ab","c")`: a test asserts exactly that.

### Content hashing normalizes line endings

`text_hash` converts CRLF→LF before hashing. The corpus was authored on
Windows; without this, the same logical content hashes differently
depending on the checkout and every file reads as modified.

---

## 4. Provenance: how enforcement works

Five tiers, with an explicit strength ordering in `enums.TIER_STRENGTH`:

```
SOURCE_FACT (4) > EXTRACTED_CLAIM (3) = USER_ASSERTION (3)
                > MODEL_INFERENCE (2) > SYNTHESIS (1)
```

`USER_ASSERTION` is ranked alongside `EXTRACTED_CLAIM`, and the reasoning
is recorded in the code: ranking it below `MODEL_INFERENCE` would let a
model's guess outrank the user; ranking it at the top would let unsourced
belief launder itself into quotable evidence.

### The floor rule

Enforced in a pydantic validator on `Provenance`, so **a violating object
cannot be constructed**, and therefore cannot be persisted by any route:

```python
Provenance(
    tier=ProvenanceTier.SOURCE_FACT,
    derivation=Derivation.DETERMINISTIC,
    agent="t",
    inputs=(ProvenanceInput(..., tier=ProvenanceTier.SYNTHESIS),),
)
# ProvenanceViolation: cannot assert tier SOURCE_FACT from inputs whose
# weakest tier is SYNTHESIS
```

Three further rules ride along in the same validator:

1. `derivation=MODEL` requires a `model_id`, unattributable model output
   is not storable.
2. `derivation=DETERMINISTIC` must **not** carry a `model_id`.
3. `derivation=MODEL` can never produce `SOURCE_FACT`.

### `ProvenanceViolation` is not a `ValueError`

Pydantic folds `ValueError` raised inside validators into its own
`ValidationError`, which would bury a provenance breach among ordinary
field errors. Inheriting from `Exception` makes it propagate unwrapped and
separately catchable. There is a test asserting it is *not* a
`ValidationError`.

### Two more places the rules are structural

- **`EvidenceRelation.QUOTES` cannot be model-derived.** A quote asserts
  verbatim text, a deterministic check. A model claiming it cannot be
  trusted to be verbatim.
- **Semantic link types cannot be deterministic.** `SUPPORTS`,
  `CONTRADICTS`, `REFINES`, `PART_OF`, ... require judgement, so
  `Derivation.DETERMINISTIC` is rejected for them. Only `MENTIONS`,
  `DERIVED_FROM`, `PRECEDES`, `RELATED_TO`, `ABOUT` may be asserted by
  ordinary code. This makes Principle 7 checkable rather than cultural.
- **`RELATED_TO` requires an explicit `score`.** It is the edge that turns
  a knowledge graph into an untyped mesh; an unscored one is rejected.

### Evidence requirement

`validate_claim` is called by `SqliteStore.put_claim`, so the rule holds at
the storage boundary regardless of caller: a claim whose tier is not
`USER_ASSERTION` must have at least one `EvidenceLink`, or the write fails.

---

## 5. Revision model

Append-only. Four operations: `create`, `change`, `supersede`, `invalidate`
(plus `merge`/`split` reserved).

Shape rules are validated: `CREATE` must carry `after` and no `before`;
`CHANGE`/`SUPERSEDE` need both; `INVALIDATE` must carry the `before` state
it invalidated. Revisions are frozen.

**Supersession is non-destructive.** `supersede_claim` marks the old claim
`SUPERSEDED`, sets `superseded_by` and `valid_to`, keeps its statement
verbatim, writes a `SUPERSEDE` revision holding both states, and writes a
`CREATE` revision for the replacement. Deleting a source likewise writes an
`INVALIDATE` revision carrying the prior state, history survives deletion.

Ordering uses a monotonic `seq` column, not timestamps: two revisions
written in the same millisecond must still be totally ordered.

Storage-agnostic by construction, a `Revision` is a plain record of
`(entity, op, before, after, cause)`. Nothing about it presumes a graph
database.

---

## 6. Deterministic parsing

### Code-fence masking (the hazard that matters)

The corpus contains 555 fenced code blocks, many with Python literals like
`[[0,1],[1,0]]`. A naive wikilink regex reads those as links, the Phase 0
audit's first pass did exactly that.

`mask_code` blanks fenced blocks and inline code **while preserving line
count**, so every subsequent line number stays accurate for span
construction. An integration test asserts no numeric-looking link targets
survive anywhere in the real corpus.

### Frontmatter: three defect shapes, not two

Phase 0 characterized two. Implementation found a third.

| Code | Shape | Count | Severity |
|---|---|---:|---|
| `FM001` | `related: [[A]], [[B]]` → YAML ParserError | 68 | error |
| `FM002` | `related: [[[A]], [[B]]]` → nested lists | 215 | warning |
| **`FM008`** | **`related: [[A]], [[B]` → truncated final link** | **18** | **warning** |
| `FM003` | no frontmatter at all | 268 | info |
| `FM005` | duplicate key | 0 | warning |

`FM008` is new. The final wikilink is missing one closing bracket, which is
why those 18 files initially had *no* verified repair: the extractor
requires `]]` and the leftover `]` tripped the residue check that prevents
lossy rewrites. Handling it explicitly, rather than loosening the residue
check, keeps the safety property and makes the defect visible as its own
class.

**Nothing is repaired automatically.** Proposals are generated, applied
*in memory*, re-parsed, and only marked `verified: true` if the result
parses to a mapping with no nested-list values. Files on disk are untouched
(ADR-001 D2).

A repair is refused when wikilinks do not account for the whole value,
a mechanical rewrite could silently drop mixed-in content.

### Reading broken metadata without repairing it

`extract_wikilink_values` recovers link names by text extraction, so the
`related:` graph is usable **today**, across all 301 malformed files,
without touching the corpus. This is what makes "diagnose, don't mutate"
practical rather than merely principled.

---

## 7. Link resolution

Eight statuses, resolved in descending confidence: exact stem → exact path
→ case-insensitive stem → case-insensitive path → path mismatch →
normalized (rename) → close matches as *candidates*.

**Ambiguity is never resolved by guessing.** Where several files share a
stem, status is `AMBIGUOUS`, `resolved_path` is `None`, and all candidates
are reported. This matters because the real corpus has three genuine
collisions, `Heap`, `Binary Search`, `Trie`: each existing as both a
pattern and an algorithm/data-structure, accounting for **180 of the 282
unresolved link occurrences**. That is a measured violation of the vault's
own "one canonical home per concept" rule, and picking a side on the user's
behalf would silently rewrite their corpus's meaning.

Two false-positive classes are handled explicitly, both discovered by
getting them wrong first:

- **URL-encoded targets** (`DSA/00_Index/DSA%20Home.md`) are decoded before
  resolution. This was the Phase 0 audit's own false positive.
- **Directory links** (`](../personal-agent/)`) resolve against a directory
  index. Fixing this removed 62 spurious "missing" links.

---

## 8. Indexing and change detection

```
discover (sorted walk, symlinks skipped, dot-dirs skipped)
  -> read bytes -> text_hash (CRLF-normalized)
  -> parse_markdown -> parse_frontmatter -> resolve links
  -> IndexedFile
```

`build_index` catches per-file exceptions and logs them, so one malformed
file cannot abort a 629-file run.

`detect_changes` compares `{path: content_hash}` against the store and
classifies each source `NEW` / `MODIFIED` / `UNCHANGED` / `DELETED`.
`requires_processing` is `new + modified`, when it is empty, there is
provably no work, and therefore no model calls.

### Spans

Heading-delimited, deterministic. Content before the first heading becomes
a preamble span so no bytes are unattributable. Heading paths are tracked
via a level stack, giving each span a breadcrumb like
`("RAG", "Architecture", "Chunking")`. The real corpus yields **7,003
spans from 629 files**.

### Excluded from indexing

`.git`, `.forge`, `.obsidian`, `engine`, `tests`, `scripts`, `docker`, plus
all dot-directories. **Known consequence:** `.obsidian-config/README.md` is
not indexed because it lives in a dot-directory. That is correct default
behaviour (Obsidian ignores dot-dirs too) but it is a real file in the
repo, so it is recorded here rather than left as a silent discrepancy.

---

## 9. Storage

SQLite via stdlib `sqlite3`, behind protocols in `storage/base.py`.

**Why SQLite:** stdlib (no dependency, no service, no container),
transactional (provenance and revisions commit atomically with the objects
they describe), single deletable file, and replaceable. Phase 1 must run on
a laptop with nothing installed.

**Why not Neo4j/Qdrant/Postgres:** the corpus is 629 files and 7,003 spans
with one writer. Standing up three services for that would be
infrastructure for an imagined future rather than the actual present.
Promotion triggers are recorded in
[technology-decisions.md §12](./technology-decisions.md).

Entities are stored as JSON documents with lookup fields promoted to
columns. During Phase 1's high-churn period this means a model change does
not require a schema migration.

**Derived state is disposable.** `store.reset()` drops everything;
`test_reset_then_reindex_reproduces_state` asserts that re-indexing
reproduces identical counts and an identical fingerprint.

---

## 10. LLM abstraction

Nothing above `llm/` names a provider. Configuration binds **roles**
(`extraction`, `analysis`, `resolution`, `synthesis`) to models; code asks
for a role. All four roles must be bound or configuration fails at startup.

- **`MockProvider`.** Deterministic, offline, the CI default.
- **`OllamaProvider`.** Local HTTP, no key, no account. Structured output
  uses Ollama's `format` parameter, with one bounded repair retry that
  feeds the model its own invalid output plus the validation error, then a
  hard failure.
- **`extract_json`** recovers JSON from fences and prose deterministically.
  Local models wrap JSON constantly; recovering is a *parsing* problem, so
  software does it rather than spending another model call.
- **`CALLS`** counts every outbound call, making "this made zero LLM calls"
  assertable in tests and observable in `forge index` output.

`ProviderUnavailable` is deliberately distinct from `LLMError`: "no model
is running" and "the model answered badly" need different fixes.

---

## 11. Extension points for Phase 2+

| Need | Seam |
|---|---|
| New source type (PDF, repo, web) | Add a parser; `Source.kind` already enumerates them |
| Embeddings | `EmbeddingProvider` alongside `LLMProvider`; `Span.embedding_ref` reserved |
| Vector store | New protocol in `storage/base.py` |
| Graph store | Implement `KnowledgeStore`, nothing upstream changes |
| Concept extraction | Write into `concepts`/`claims`; tables and validation exist |
| LangGraph workflows | `Provenance.workflow_run_id` and `Revision.workflow_run_id` are already threaded |
| Contradiction/Synthesis | New entities; the tier ordering already supports them |

---

## Related

- [CLI usage](../cli.md), [Test strategy](../test-strategy.md)
- [Canonical knowledge model](../knowledge-model/canonical-model.md)
- [Target architecture](./target-architecture.md), [Technology decisions](./technology-decisions.md)
- [ADR-001](../decisions/001-forge-knowledge-os.md)
- [Local model capability spike](../research/local-model-capability-spike.md)
