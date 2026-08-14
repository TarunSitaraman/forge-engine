# Retrieval Baseline — Measured, Not Assumed

*Phase 3. Every number on this page was produced by
`forge retrieval-eval` against the labelled set in
`tests/fixtures/eval/retrieval-v1.yaml`, run over the real Forge vault. Nothing
here is estimated, and nothing was tuned against the labels.*

**Headline: lexical search is the best retrieval method Forge currently has.
Embeddings were built, measured, and rejected. Hybrid fusion was swept across
four weights and every one of them regressed. No vector database is
justified.**

---

## 1. Why this document exists

The Phase 3 brief forbids claiming a retrieval improvement without measured
evidence, and forbids assuming a fusion weight. Both are easy rules to violate
by accident: "add embeddings, blend 50/50, ship" is the default move in this
part of the industry, and it produces a system nobody can defend.

So retrieval quality is treated the way a compiler team treats performance: a
fixed benchmark, a recorded baseline, and a rule that a change ships only if
the benchmark says it helped.

---

## 2. The evaluation set

`tests/fixtures/eval/retrieval-v1.yaml` — **24 queries, 48 labels**, hand-built
and version-pinned.

| Category | Queries | What it probes |
|---|---:|---|
| `exact_concept` | 5 | The user knows the term and types it exactly. |
| `fuzzy_concept` | 5 | The user describes the idea in *other words* — the hardest case, and the one embeddings are supposed to fix. |
| `related_concept` | 3 | The answer spans several related documents. |
| `dsa` | 4 | The flagship section, where retrieval is used most. |
| `technology` | 3 | Canonical technology references. |
| `project` | 4 | Project knowledge packs. |

Two properties make the set worth trusting:

* **Labels are verified against the filesystem** on every run
  (`EvalDataset.verify_labels`), and reported as `label_rot` in the JSON
  output. A reorganized vault surfaces as a data problem, not as a mysterious
  drop in recall.
* **Ranking is scored at the source-document level**, not the span level. A
  document chunked into ten spans occupies one rank, not ten. Span-level
  ranking would have inflated precision for exactly the files that happen to
  be long.

Selection methodology, including which queries were deliberately made hard, is
documented in the header of the YAML file itself.

### Known limits of this set

* **24 queries is small.** A difference under 0.01 is treated as noise by
  `metrics.compare`, and the verdict text says so. Do not read a 0.005 gain as
  a win.
* **It was written by the person who built the corpus.** That biases toward
  queries a Forge user would plausibly type, which is the point, but it is not
  a neutral sample of all possible questions.
* **It measures retrieval, not answers.** There is no generation step in Phase
  3, so nothing here says anything about answer quality.

---

## 3. Results

Run: `forge retrieval-eval --methods lexical,semantic,hybrid --detail`
over 645 sources / 1692 spans of the real vault.

| Method | R@5 | R@10 | P@5 | MRR | Misses | Latency |
|---|---:|---:|---:|---:|---:|---:|
| **lexical (FTS5/BM25)** | **0.406** | **0.608** | **0.158** | **0.471** | **6** | **18.7 ms/q** |
| semantic (hashing-v1-256c) | 0.301 | 0.581 | 0.133 | 0.342 | 6 | 244.6 ms/q |
| hybrid (w=0.25) | 0.378 | 0.544 | 0.158 | 0.337 | 7 | 259.0 ms/q |
| hybrid (w=0.50) | 0.364 | 0.517 | 0.158 | 0.337 | 7 | 264.4 ms/q |
| hybrid (w=0.75) | 0.279 | 0.449 | 0.125 | 0.336 | 9 | 264.3 ms/q |

`w` is the share of the fused score given to the semantic signal; the
remainder goes to lexical. Both scores are min-max normalized before fusion —
BM25 and cosine live on incomparable scales, and blending them raw would let
whichever has the wider range dominate regardless of the weight.

**Deltas against the lexical baseline:**

| Candidate | R@5 | R@10 | P@5 | MRR | Verdict |
|---|---:|---:|---:|---:|---|
| semantic | −0.104 | −0.028 | −0.025 | −0.128 | regression |
| hybrid (w=0.25) | −0.028 | −0.064 | −0.000 | −0.134 | regression |
| hybrid (w=0.50) | −0.042 | −0.092 | −0.000 | −0.133 | regression |
| hybrid (w=0.75) | −0.126 | −0.160 | −0.033 | −0.135 | regression |

**Decision: hybrid retrieval is NOT adopted.** Lexical remains the default and
only retrieval path. The embedding code stays in the tree, off by default,
because the measurement must be repeatable when a real neural model becomes
available.

The sweep is monotone in the wrong direction: every point of weight moved from
lexical to semantic costs recall. w=0.25 is the least-bad hybrid and still
loses 0.126 of R@10.

Two notes on why the sweep is reported in full rather than as a single
verdict. First, an earlier run of this same sweep — before the Phase 3
documentation was written into the vault — had w=0.25 *beating* the baseline
on R@5 (+0.014) and P@5 (+0.008) while losing badly on R@10 and MRR. Reporting
only R@5, or picking one weight a priori, would have manufactured a win out of
that run. Second, that earlier run differed from this one only because writing
these documents added files to the corpus being searched; on a 24-query set,
that is enough to move R@10 by 0.02. Both facts argue for the same discipline:
sweep the parameter, report four metrics, and treat small deltas as noise.

**Third data point, and the sharpest one.** Merging seven new canonical
technology docs into the vault (Kubernetes, FastAPI, Node/Express, PostgreSQL,
React, Redis, Supabase) moved lexical R@10 from 0.650 to **0.608** and added a
sixth miss, without a single line of retrieval code changing. The query that
broke is `fuzzy-task-ordering` — "task ordering with dependencies", whose
target is `Topological Sort.md`. It sat at rank 9; the new infrastructure docs
discuss scheduling and dependency ordering often enough to push it to rank 11.

`fuzzy_concept` R@10 fell from 0.300 to 0.100 on that one query alone, which is
what a 24-query set does: it is sensitive enough to detect real movement and
too small to be stable. Both facts are true at once and both matter.

The lesson is not "the corpus got worse". It is that **BM25 degrades as a
corpus grows denser in a topic**, and it degrades first exactly where it was
already weakest — paraphrase. That is a much stronger argument for the two
deterministic fixes in §5 than any of the earlier runs made, because this time
the regression was observed rather than predicted.

---

## 4. The critical caveat about "semantic"

The semantic row above was produced by `HashingEmbeddingProvider`
(`hashing-v1-256c`): a hashed bag of word tokens and character 4-grams with
sublinear term frequency and L2 normalization.

**It is a lexical-statistical representation. It is not a neural sentence
embedding and it does not capture meaning.** Two passages that share no
vocabulary score near zero no matter how synonymous they are.

Why it was used at all: no neural model could be obtained in this environment.
The sandbox network policy denies `ollama.com:443` and `huggingface.co`, which
is documented with a reproducible transcript in
[`local-model-capability-spike.md`](local-model-capability-spike.md). The
options were to (a) ship the embedding pathway unmeasured, (b) fabricate
numbers, or (c) measure the pathway end to end with an honest, clearly-labelled
non-neural vectorizer. (c) is what happened. The storage, cache-invalidation,
fusion, and evaluation code paths are all genuinely exercised; only the model
is a stand-in.

**What this means for reading the table:** the semantic row is evidence that
*this particular vectorizer* does not help. It is **not** evidence that neural
embeddings would not help. Those are different claims and this document makes
only the first. The specific gap it cannot speak to is the `fuzzy_concept`
category, which is precisely where real embeddings should earn their place —
and precisely where a vocabulary-overlap vectorizer has nothing to offer.

The re-measurement is one command once a model is reachable:

```bash
ollama pull nomic-embed-text
forge embeddings build --provider ollama
forge retrieval-eval --methods lexical,semantic,hybrid --provider ollama
```

If that run shows an improvement, hybrid gets adopted then, on that evidence.

---

## 5. Where lexical retrieval actually fails

Per-category results for the lexical baseline:

| Category | R@5 | R@10 | MRR |
|---|---:|---:|---:|
| technology | 0.833 | 0.833 | 0.833 |
| project | 0.500 | 0.875 | 0.708 |
| exact_concept | 0.467 | 0.733 | 0.457 |
| related_concept | 0.300 | 0.644 | 0.319 |
| dsa | 0.375 | 0.625 | 0.577 |
| **fuzzy_concept** | **0.100** | **0.100** | **0.083** |

The failure is concentrated and unsurprising: **paraphrase**. The six
complete misses, from the per-query detail:

| Query | Category | Target | First hit |
|---|---|---|---|
| "disjoint sets" | fuzzy | `01_Patterns/Union Find.md` | never |
| "measure similarity between documents using an index" | fuzzy | `Technologies/Docs/vector-databases.md` | never |
| "grounding an LLM in your own documents" | fuzzy | `Technologies/Docs/rag.md` | rank 14 |
| "vector databases" | exact | `Technologies/Docs/vector-databases.md` | rank 12 |
| "task ordering with dependencies" | fuzzy | `01_Patterns/Topological Sort.md` | rank 11 |
| "off-by-one errors" | dsa | `08_Mistakes/Off-by-One.md` | rank 17 |

Four of the five are vocabulary mismatches — the vault says "Union Find", the
user says "disjoint sets". The `vector-databases` case is different and more
interesting: the term appears so often *across* the corpus (every RAG and
LangChain document mentions it) that the canonical page loses to pages that
merely discuss it. That is a BM25 saturation problem, not a semantic one, and
it is fixable deterministically — by boosting title matches — which is a
cheaper and more defensible fix than a vector store.

**Concrete implication for Phase 4+:** the two highest-value retrieval
improvements available are (1) title/heading boosting, and (2) a user-curated
alias table wired into query expansion, reusing the identity config that
already exists. Both are deterministic, both are testable against this same
set, and both should be measured before any vector database is reconsidered.

---

## 6. Cost of the alternatives

| Option | Latency | Extra dependency | Measured benefit |
|---|---:|---|---|
| Lexical (current) | 13.8 ms/q | none (SQLite FTS5, stdlib) | baseline |
| + hashing embeddings | ~200 ms/q | none | **negative** |
| + neural embeddings | untested here | model download, ~100–500 MB | **unknown** |
| + vector database | untested here | Qdrant/pgvector service | **unknown** |

A 15× latency increase for a measured regression is not a trade-off worth
making. Brute-force cosine over 1626 vectors in SQLite is what makes the
semantic row slow, and a vector database would fix *that* — but fixing the
speed of a method that returns worse results is not an improvement.

---

## 7. How to reproduce

```bash
cd engine
FORGE_VAULT_PATH=/path/to/forge FORGE_STATE_DIR=/tmp/eval \
  python -m forge.cli.main ingest /path/to/forge
FORGE_VAULT_PATH=/path/to/forge FORGE_STATE_DIR=/tmp/eval \
  python -m forge.cli.main embeddings build --provider hashing
FORGE_VAULT_PATH=/path/to/forge FORGE_STATE_DIR=/tmp/eval \
  python -m forge.cli.main retrieval-eval --methods lexical,semantic,hybrid --detail
```

Ingestion is deterministic and makes zero LLM calls. The evaluation makes zero
LLM calls. Both are reproducible offline.

---

## 8. What would change this decision

State the conditions in advance, so the decision can be revisited on evidence
rather than on enthusiasm:

* A real neural embedding model shows a **positive R@10 and MRR delta** on this
  same set, at some swept weight. Then hybrid is adopted at that weight.
* The evaluation set grows past ~100 queries and the noise floor drops enough
  that smaller deltas become readable.
* Retrieval latency at the current scale exceeds ~200 ms/query for the lexical
  path. It is 13.8 ms today; the corpus would need to grow by more than an
  order of magnitude.

Until one of those happens, the answer to "should Forge add a vector
database?" is **no, and here is the measurement.**

---

## Related

* [`local-model-capability-spike.md`](local-model-capability-spike.md) — why no
  neural model was available, with the transcript.
* [`../architecture/technology-decisions.md`](../architecture/technology-decisions.md)
  — the standing no-vector-database position this measurement supports.
* [`../architecture/phase-3-implementation.md`](../architecture/phase-3-implementation.md)
  — how retrieval, the graph, and activation fit together.
