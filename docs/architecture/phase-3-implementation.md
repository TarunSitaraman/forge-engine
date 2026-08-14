# Phase 3 — Knowledge Activation & Retrieval

*What was built, how it behaves, and what it deliberately refuses to do.
Describes the implementation as it exists.*

**Status:** implemented · **Tests:** 595 passing, 89% coverage · **LLM required:** no
**Validate:** `bash scripts/validate_phase3.sh` (17/17)

---

## 1. What Phase 3 is

Phase 2 could *propose* knowledge. Phase 3 closes the loop: an approved
proposal becomes canonical knowledge that can be traversed, cited, and
retrieved — and every step of that path is reversible, idempotent, and
traceable back to a page of a source document.

Delivered:

- **Proposal activation** — `APPROVED → ACTIVATED`, creating canonical
  Concepts and Claims with evidence links, provenance, and revisions
- **Deterministic idempotency** — approve, activate, re-index, activate again;
  nothing duplicates
- **Concept identity states** and a **persisted user decision file** for the
  known vault collisions (Heap, Binary Search, Trie)
- **Evidence-gated relationship activation** over a five-type vocabulary
- **A SQLite knowledge graph** with bounded traversal, and measurements that
  justify not adopting a graph database
- **Graph integrity diagnostics** — nine codes, report-only
- **A labelled retrieval evaluation set** (24 queries / 48 labels) and a
  metrics harness (Recall@5/@10, Precision@5, MRR)
- **A measured lexical / semantic / hybrid comparison** with a swept fusion
  weight — and the honest conclusion that **hybrid was rejected**
- **Batch proposal review** with a guard on ambiguous proposals

Not built, deliberately: contradiction detection, synthesis, autonomous
research, LangGraph, Neo4j, Qdrant, web frontend, Obsidian plugin, MCP.

---

## 2. The activation path

```
 PENDING proposal
        |
   human decision (CLI / API) --- REJECTED ---> terminal
        |                    \--- SUPERSEDED --> terminal
   APPROVED
        |
   ProposalActivator.activate
        |
        |-- deterministic id: Concept.make_id(name, namespace)
        |                     Claim.make_id(statement, subject)
        |
        |-- already exists? --> ALREADY_ACTIVE (records the link, creates nothing)
        |-- unresolved collision? --> REFUSED   (with the command that fixes it)
        |-- persistence raised? --> FAILED      (proposal stays APPROVED, retryable)
        |
   put_concept / put_claim(+EvidenceLink)   one transaction
        |
   Revision written (append-only)
        |
   proposal.activate(entity_type, entity_id)
   ACTIVATED
```

Four outcomes, and **none of them is silent**. `ActivationOutcome` is
`CREATED | ALREADY_ACTIVE | REFUSED | FAILED`, and a report carries the reason
string for every one. A proposal that fails to persist is never reported as
activated — it stays `APPROVED` so the same command retries it once the cause
is fixed.

**Every activated entity keeps its origin.** `Concept.origin_proposal_id`,
`Concept.origin_span_ids`, and `Claim.origin_proposal_id` are what let
`forge concept <name>` answer *"which proposal created this?"* and *"which
source span caused this claim to exist?"* — the two questions the brief names.

### Idempotency, concretely

Identity is derived, not allocated: `Concept.make_id` is a BLAKE2b digest over
`("concept", canonical_name, namespace)`, so the same proposal always computes
the same primary key. Activating twice hits the `ALREADY_ACTIVE` path.

The test that matters is `test_activation.py::TestIdempotency`, which runs the
exact cycle the brief specifies — *approve, approve again, re-index, activate
again* — and asserts zero duplicate concepts, claims, evidence links, and
revisions. Re-indexing between activations is the important part: it proves
identity survives a full rebuild of derived state.

### Provenance on activated entities

An activated Concept or Claim inherits `MODEL_INFERENCE` from the proposal that
produced it — activation is a *transcription* of a decision, not a new source
of truth, so it may not upgrade the tier.

One subtlety worth recording. The `EvidenceLink` created alongside a Claim
carries **deterministic** provenance while the Claim itself carries model
provenance. This looks inconsistent and is not: the link asserts only "this
quote appears in this span", which the activator verifies in code by string
comparison before writing it. The provenance floor rule from Phase 1 correctly
rejected the first attempt, where the link claimed `QUOTES` on model
provenance — the rule caught a real modelling error rather than being worked
around.

---

## 3. Concept identity

Similarity never merges concepts. Phase 2 established that; Phase 3 adds the
states and the place where a human records the answer.

| State | Meaning |
|---|---|
| `EXACT_MATCH` | The name matches a canonical concept exactly. |
| `ALIAS_MATCH` | The name is a user-registered alias. |
| `RESOLVED_BY_USER` | A collision the user has explicitly decided. |
| `NEW` | Nothing known — a genuinely new concept. |
| `AMBIGUOUS` | Several canonical homes; **Forge refuses to pick**. |

There is deliberately no `MERGED`.

### The decision file

`config/concept-identity.yaml` is knowledge configuration, not matcher code:

```yaml
version: 1
collisions:
- name: Heap
  # no `default:` key -> still ambiguous, deliberately.
  # Add `default: data-structure/Heap` to decide it.
  identities:
  - canonical_name: Heap
    kind: pattern
    namespace: pattern
    vault_path: DSA/01_Patterns/Heap.md
  - canonical_name: Heap
    kind: data_structure
    namespace: data-structure
    vault_path: DSA/03_DataStructures/Heap.md
aliases: {}
```

`forge identity scaffold` generates it from collisions **actually present in
the vault** — currently four: `Binary Search`, `Heap`, `Trie`, and
`weekly-review` — and leaves every one undecided. It suggests a namespace
from the containing folder — `01_Patterns` → `pattern` — because the corpus
already encodes the distinction there. It does not invent vocabulary, and
re-scaffolding preserves existing decisions rather than resetting them.

`forge identity decide Heap data-structure/Heap` records the choice. From then
on the matcher checks the user's decision **before** the vault-collision check,
so a decided name resolves cleanly while an undecided one stays ambiguous.

**No LLM is involved anywhere in this path**, as the brief requires.

A related guarantee, tested in
`test_phase3_activation.py::test_a_model_supplied_namespace_is_ignored`: a
proposal that carries a `namespace` in its details cannot create a namespaced
concept. Namespacing comes only from the identity config. Otherwise the model
would be choosing the vocabulary this whole mechanism exists to let the user
choose.

---

## 4. Relationships

Five types, and nothing else: `RELATED_TO`, `PART_OF`, `DEPENDS_ON`,
`IMPLEMENTS`, `EXPLAINS`.

Relationship spam is the failure mode that turns a knowledge graph into an
untyped mesh, so the gate is deliberately strict:

- **Co-occurrence requires ≥ 2 shared spans.** One shared span is a
  coincidence; the rejection message says so.
- **`RELATED_TO` requires an explicit score.** The domain model rejects an
  unscored one (`ClaimLink._check`).
- **Name matching requires ≥ 6 characters** of normalized text, so short names
  don't match inside unrelated words.
- **Deterministic code may not assert a semantic edge type.** `DEPENDS_ON` and
  friends require judgement, so a `DETERMINISTIC` derivation carrying one is a
  `ProvenanceViolation`.

Measured on the demo corpus: **3 candidates considered, 1 created, 2 rejected**
— both rejections for having only a single shared span. That ratio is the
feature working.

---

## 5. The graph — and why SQLite is enough

The graph is an indexed adjacency table (`claim_links`) plus bounded traversal
in `forge/graph/graph.py`. Every walk takes a depth limit **and** a node
budget, neither of which can be disabled: `get_related_concepts` clamps to
`DEFAULT_MAX_DEPTH = 3`, `find_path` caps at 6, and both stop at
`DEFAULT_NODE_BUDGET = 500` nodes.

`find_path` returning `None` means *"no path within this bound"* — deliberately
not *"no path"*, which a bounded search cannot establish. The CLI prints
exactly that: `no path within 3 hops (this does not prove none exists)`.

### Measured, at a scale Forge has not reached

"It is fast because it is small" is not an argument. `scripts/measure_graph_scale.py`
builds a synthetic graph an order of magnitude larger than the real one and
measures the operations the product actually performs:

| Graph | Nodes | Edges | Neighbour lookup | Bounded path (d≤3) | Depth-3 neighbourhood |
|---|---:|---:|---:|---:|---:|
| Demo (real) | 3–4 | 1 | 0.03 ms | 0.06 ms | — |
| Synthetic | 5,000 | 19,991 | **0.24 ms** | **17.0 ms** | 14.5 ms (budget-capped at 408) |

At 5,000 concepts — roughly eight times the vault's document count — with a
branching factor of 8.0, neighbour lookup is a quarter of a millisecond and a
depth-3 path search is 17 ms. **No graph database is justified.** The condition
that would change that is stated in
[`../research/retrieval-baseline.md`](../research/retrieval-baseline.md) §8.

### Integrity diagnostics

Nine codes, all deterministic, all **report-only** — the same discipline as the
Phase 1 frontmatter diagnostics, for the same reason: an automatic repair to a
knowledge graph is an unreviewed change to what the user believes.

| Code | Severity | Meaning |
|---|---|---|
| GR001 | error | Relationship whose source entity does not exist |
| GR002 | error | Relationship whose target entity does not exist |
| GR003 | warning | Duplicate relationship (same pair, same type) |
| GR004 | warning | Relationship type outside the supported vocabulary |
| GR005 | warning | Relationship with no provenance agent |
| GR006 | error | EvidenceLink pointing at a deleted span |
| GR007 | error | Claim whose tier requires evidence but has none |
| GR008 | warning | Concept with no origin proposal and no claims |
| GR009 | error | Relationship connecting an entity to itself |

GR007 has two lines of defence, and the tests assert both: the store *refuses*
to write an unevidenced non-user claim at all, and the diagnostic catches the
state if it arrives later by corruption.

---

## 6. Retrieval

Lexical FTS5/BM25 remains the only retrieval path in production.

The full measurement lives in
[`../research/retrieval-baseline.md`](../research/retrieval-baseline.md). The
short version:

| Method | R@5 | R@10 | MRR | Verdict |
|---|---:|---:|---:|---|
| lexical | 0.406 | 0.608 | 0.471 | **baseline, adopted** |
| semantic | 0.301 | 0.581 | 0.342 | regression |
| hybrid (w=0.25 / 0.5 / 0.75) | 0.378 / 0.364 / 0.279 | 0.544 / 0.517 / 0.449 | 0.337 / 0.337 / 0.336 | regression at every weight |

*(Re-measured after the Phase 0–4 branches were merged, which added seven
technology docs to the corpus and moved lexical R@10 from 0.650 to 0.608
without any retrieval code changing. The verdict is unchanged; the mechanism
is discussed in the baseline document.)*

Fusion weight was **swept, not chosen** — `DEFAULT_FUSION_WEIGHTS =
(0.0, 0.25, 0.5, 0.75, 1.0)`, with the endpoints as anchors that must reproduce
the pure methods. An earlier run of the same sweep had w=0.25 beating lexical
on R@5 while losing on R@10 — reporting one metric, or picking one weight a
priori, would have manufactured a false win out of it.

The semantic row was produced by `HashingEmbeddingProvider` — a hashed
bag-of-features vector, **explicitly not a neural embedding** — because no
model could be downloaded in this environment. That limitation, and exactly
what it does and does not license anyone to conclude, is documented in
[`../research/local-model-capability-spike.md`](../research/local-model-capability-spike.md)
and in §4 of the retrieval baseline.

---

## 7. Review ergonomics

`forge proposals list` filters by type, status, safety, and source.
`forge proposals approve-all` adds batch approval with two guards:

- **Dry run by default.** `--no-dry-run` is required to actually decide
  anything; the default prints what *would* be approved.
- **Ambiguous proposals need `--include-ambiguous`.** Bulk-approving an
  ambiguous semantic proposal is approving a decision nobody made, so it takes
  an explicit flag and exits 2 otherwise.

Safety class stays derived from provenance and evidence — it is never
something a model asserts about its own output.

---

## 8. Write-back

Unchanged from Phase 2 and deliberately conservative. **Activating a Concept
or Claim writes nothing to Markdown.** Canonical knowledge lives in the derived
store and exists independently of the vault files.

The only write-back path remains flag-gated metadata repair, with backups,
revisions, and refusal of ambiguous changes.
`test_phase3_activation.py::TestNoWriteBack` runs the whole Phase 3 loop
against the real repository and asserts `git status --porcelain` is byte-identical
before and after.

---

## 9. Storage changes

Schema v2 → **v3**, migrated in place:

- `concepts` gained `namespace`, `origin_proposal_id`, `origin_span_ids`
- `UNIQUE(canonical_name)` became `UNIQUE(canonical_name, IFNULL(namespace,''))`
  — the old constraint made namespaced concepts impossible
- `claims` gained `origin_proposal_id`
- `proposals` gained `activated_entity_type`, `activated_entity_id`,
  `activated_at`

The migration runs in two parts: `_migrate_structure` reshapes tables *before*
the schema script executes, `_migrate_derived` runs after. Doing it in one pass
failed with `no such column: namespace`, because `CREATE TABLE IF NOT EXISTS`
will not add a column to a table that already exists.

---

## 10. Known limitations

- **The semantic measurement used a non-neural vectorizer.** It shows that
  vocabulary-overlap vectors do not help. It cannot speak to real embeddings,
  and specifically cannot speak to the `fuzzy_concept` category where they
  would matter most.
- **24 evaluation queries is a small set.** Differences under 0.01 are treated
  as noise, and the comparison output says so.
- **The real graph is tiny** because canonical concepts only exist where
  evidence was activated, and activation requires human approval. The scale
  measurement compensates, but it is synthetic.
- **Relationship discovery is co-occurrence only.** `PART_OF`, `DEPENDS_ON`,
  `IMPLEMENTS`, and `EXPLAINS` are supported by the model and the graph but
  have no automatic discovery path — they are created by explicit decision.
- **`find_path` bounds are conservative.** Depth 3 by default, hard-capped at
  6. On a sparse graph, most pairs return "no path within the bound".

---

## Related

- [`phase-2-implementation.md`](phase-2-implementation.md) — ingestion and the proposal system
- [`../research/retrieval-baseline.md`](../research/retrieval-baseline.md) — the full retrieval measurement
- [`../research/local-model-capability-spike.md`](../research/local-model-capability-spike.md) — why no neural model was available
- [`../knowledge-model/canonical-model.md`](../knowledge-model/canonical-model.md) — the entity model being activated into
- [`../cli.md`](../cli.md) — command reference
