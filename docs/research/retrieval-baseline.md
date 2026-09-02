# Retrieval Baseline — Measured, Not Assumed

*Phase 3. Every number on this page was produced by
`forge retrieval-eval` against the labelled set in
`forge/evaluation/data/retrieval-v1.yaml`, run over the real Forge vault. Nothing
here is estimated, and nothing was tuned against the labels.*

**Headline (Phase 3, since superseded — see §0 below): lexical search is the
best retrieval method Forge currently has. Embeddings were built, measured, and
rejected. Hybrid fusion was swept across four weights and every one of them
regressed. No vector database is justified.**

---

## 0. The fuzzy_concept ceiling is a ranking problem, not a retrieval one (2026-09-02)

Three independent methods now put `fuzzy_concept` R@10 at **exactly 0.300**:
lexical, hybrid over hashing vectors at every fusion weight, and hybrid over
spaCy static vectors at every fusion weight. A number that stable under
unrelated changes is a property of the system, not of any one method.

### First, a negative result: static word vectors are much worse than hashing

`SpacyEmbeddingProvider` (mean-pooled 300-d `en_core_web_md` vectors) was added
to test whether *semantic* matching beats *vocabulary overlap*. It does not —
not this kind of semantic matching:

| Method | R@5 | R@10 | P@5 | MRR | Misses |
|---|---:|---:|---:|---:|---:|
| lexical | **0.468** | **0.662** | 0.167 | 0.519 | **4** |
| semantic (spacy) | 0.231 | 0.260 | 0.100 | 0.246 | 15 |
| hybrid (w=0.25) | 0.496 | 0.601 | 0.183 | **0.554** | 5 |
| hybrid (w=0.50) | 0.432 | 0.594 | 0.158 | 0.551 | 6 |
| hybrid (w=0.75) | 0.289 | 0.414 | 0.133 | 0.323 | 10 |

Every weight regresses, and semantic alone misses 15 of 24 queries. Compare
the *hashing* provider, which is not even semantic and reaches R@5 = 0.551 at
w=0.50. Mean-pooling static vectors over a whole span averages it toward the
corpus mean, so spans stop being distinguishable; the hashing vector keeps rare
discriminating terms and beats it comfortably.

One thing does behave as predicted. Semantic-alone raises `fuzzy_concept` R@5
from 0.100 to 0.200, and `hybrid(w=0.75)` reaches 0.300 — the best fuzzy R@5
measured — while destroying `project` (0.750 → 0.250) and `technology`
(1.000 → 0.333). **The semantic signal helps exactly where predicted and hurts
everywhere else.** It is a real signal buried in a bad instrument.

### Then the diagnostic that matters

Are the `fuzzy_concept` targets even retrievable? Every one was probed at
depth 3000 against lexical search. `label_rot` is empty; all five queries
return 550–650 distinct documents.

| Query target | Rank |
|---|---:|
| `DSA/01_Patterns/Sliding Window.md` | 1 |
| `Technologies/Docs/rag.md` | 8 |
| `DSA/01_Patterns/Topological Sort.md` | 15 |
| `DSA/09_CheatSheets/Sliding Window Cheat Sheet.md` | 35 |
| `Technologies/Docs/vector-databases.md` | 112 |
| `DSA/03_DataStructures/Disjoint Set.md` | 211 |
| `DSA/01_Patterns/Union Find.md` | 231 |

**Nothing is missing. Everything is mis-ranked.** Indexing and chunking are
exonerated: the candidate set contains every labelled document. BM25 places
them 100–200 positions too low because the queries deliberately share almost no
vocabulary with them — `"keeping track of which items belong to the same group
as they merge"` has no term in common with `Union Find`.

### What this rules out

**A cross-encoder re-ranker over the top 30 cannot fix this.** It can only
reorder what it is given, and four of the seven targets sit at ranks 35, 112,
211 and 231. Re-ranking a shortlist that excludes the answer changes nothing.
To reach rank 231 the shortlist would have to be ~250 deep, and a cross-encoder
scoring 250 candidates per query is a different cost proposition from scoring
30 — that trade needs measuring before it is assumed.

`RETRIEVAL_DEPTH = 30` is also not the binding constraint in the way it first
appears. Raising it to 250 would let those documents *into* the evaluation
window, but they would still rank below 30 and contribute nothing to R@5 or
R@10. The depth is a symptom; the ranking is the disease.

### What this points at

A **bi-encoder over the whole index** — a transformer sentence encoder scoring
every document independently of its lexical rank. That is the one instrument
tested here that can move a document from rank 231 into the top 10, because it
never sees rank 231 in the first place. Static vectors were too weak an
instrument to settle it, and the fact that the fuzzy-specific signal survived
even in them is the encouraging part of an otherwise negative result.

Left unmeasured: this environment blocks `huggingface.co` and `ollama.com`
(403 at the proxy), so no transformer encoder can be obtained here. See
`docs/plans/neural-embeddings.md`.

---

## 0. Title boosting measured 2026-09-02 — a shipped setting that costs recall

**The heading/filename boost was already in the code, already shipping, and had
never been scored.** `SearchQuery.title_boost` existed from Phase 3, the
answering service passed `TITLE_BOOST = 1.25`, and no run of the labelled set
had ever exercised it — `retrieval-eval` had no way to sweep it. The value was
chosen from a single convincing anecdote, recorded in `search.py`: asking *"what
is retrieval augmented generation?"* ranked four Prompt-Library spans above
`Technologies/Docs/rag.md`, the canonical page, which missed the top eight
entirely.

The anecdote is real. It does not generalise.

Measured over 670 sources / 8,133 spans, `hashing-v1-256c`, same labelled set:

| Method | R@5 | R@10 | P@5 | MRR | Misses | Latency |
|---|---:|---:|---:|---:|---:|---:|
| lexical (FTS5/BM25) | 0.468 | **0.662** | 0.167 | 0.519 | **4** | **8.3 ms/q** |
| title (b=1.25) *— the shipped value* | 0.468 | 0.600 | 0.167 | 0.532 | 5 | 9.6 ms/q |
| title (b=1.5) | 0.468 | 0.621 | 0.167 | 0.514 | 5 | 9.5 ms/q |
| title (b=2) | 0.447 | 0.579 | 0.158 | 0.502 | 5 | 9.7 ms/q |
| title (b=3) | 0.406 | 0.558 | 0.142 | 0.491 | 6 | 9.5 ms/q |
| hybrid (w=0.25) | 0.524 | 0.662 | 0.183 | 0.547 | 4 | 617 ms/q |
| **hybrid (w=0.50)** | **0.551** | **0.685** | **0.200** | 0.536 | 4 | 623 ms/q |
| hybrid (w=0.75) | 0.511 | 0.678 | 0.192 | **0.569** | 4 | 636 ms/q |

**Every title boost is a regression, and the shipped 1.25 is the mildest of
them.** It buys nothing at R@5 — identical to lexical — costs 0.0625 of R@10,
and turns a hit into a fifth miss. The comparison harness classifies all four
as `regression` without being asked to.

### Where the damage is, and why

Per-category R@10 isolates it:

| Category | lexical | b=1.25 | change |
|---|---:|---:|---|
| `exact_concept` | 0.933 | 0.933 | — |
| `technology` | 1.000 | 1.000 | — |
| `project` | 0.750 | 0.750 | — |
| `related_concept` | 0.578 | 0.578 | — |
| `dsa` | 0.500 | 0.375 | **−0.125** |
| `fuzzy_concept` | 0.300 | 0.100 | **−0.200** |

The boost is inert on the categories it was meant to help and destructive on
the two hardest. That is mechanically what it should do, in hindsight: it can
only fire when the query's words appear in a filename or heading, which is
precisely the case where BM25 already ranks that page well. It adds nothing
there, and the multiplier it applies to those already-winning pages pushes down
the differently-worded pages that answer a `fuzzy_concept` query. `fuzzy_concept`
loses two thirds of its recall.

**A boost that can only reward pages the baseline already finds is not a
retrieval improvement; it is a re-ranking of the results you did not need
re-ranked, paid for out of the results you did.**

### What this does not settle

The eval measures **document recall**. The answering service uses the boost for
**span selection** inside an answer, which is a different question, and the
anecdote that motivated it was about span ordering. It is possible the boost
earns its keep there and is still wrong here. What is no longer possible is
shipping it as though it were measured — and a setting that costs 20 points of
`fuzzy_concept` recall needs a better defence than one query.

The honest options are to default `title_boost` to 1.0 and keep the field for
callers who want it, or to keep 1.25 in answering *only*, with this table cited
next to it. Left as a decision, not silently changed.

### Alias-driven query expansion: not attempted, and why

The Phase 3 miss analysis proposed two deterministic improvements. This is the
other one, and **it has no data to run on**: the vault contains zero `aliases:`
frontmatter keys across all 671 files, and zero aliased wikilinks
(`[[target|shown]]`) across all 4,703 links. `config/concept-identity.yaml`
records collision *decisions*, not surface forms.

Expansion needs a source of alternative names before it can be built. That is
corpus work — adding `aliases:` to canonical pages, which Obsidian already
supports natively — not engine work, and it should be measured against
`fuzzy_concept` when it exists, since that is the category with room to move
(R@5 = 0.100).

---

## 0. Re-baselined 2026-09-01 — and the rejection no longer holds

**Everything in §2 onward was measured before the engine was split out of the
vault.** The full sweep has been re-run on the current corpus, same command,
same labelled set, same embedder. The conclusion changed.

Measured at engine `b3218bf`, over 643 sources / 7,118 spans, with
`hashing-v1-256c`. Re-running produces byte-identical scores — the hashing
provider is deterministic, so this is reproducible rather than a single sample.

| Method | R@5 | R@10 | P@5 | MRR | Misses | Latency |
|---|---:|---:|---:|---:|---:|---:|
| lexical (FTS5/BM25) | 0.510 | 0.685 | 0.192 | 0.535 | 4 | **11.1 ms/q** |
| semantic (hashing-v1-256c) | 0.532 | 0.643 | 0.208 | 0.563 | 5 | 769 ms/q |
| hybrid (w=0.25) | 0.565 | 0.693 | 0.208 | 0.566 | 4 | 775 ms/q |
| **hybrid (w=0.50)** | **0.574** | **0.699** | **0.217** | **0.604** | **4** | 860 ms/q |
| hybrid (w=0.75) | 0.567 | 0.678 | 0.217 | 0.571 | 4 | 891 ms/q |

**Then, every fusion weight regressed. Now, every fusion weight improves.**
At `w=0.50` hybrid beats lexical on all four quality metrics — R@5 +0.064,
R@10 +0.014, P@5 +0.025, MRR +0.070. The headline claim this document has
carried since Phase 3, *"hybrid fusion was swept across four weights and every
one of them regressed"*, is no longer true of this system.

### What changed, and what did not

Held constant: the labelled set (24 queries, 48 labels, unedited), the
embedder (`hashing-v1-256c`), the command, and the fusion weights.

Changed — **two things at once, and they are not separable from these runs**:

| | Then | Now |
|---|---:|---:|
| Sources | 645 | 643 |
| Spans | 1,692 | **7,118** |

The file count barely moved. The span count is **4.2×**, which is the chunker,
not the corpus. The engine's own `docs/` tree also left in the split, removing
~20 documents that had been competing for these queries.

So the reversal is real *for the system as it stands*, and attributing it to
either cause would be guesswork. A run that isolates them would re-chunk the
pre-split corpus at the current settings, which is possible and has not been
done.

### What this does not license

**The latency is now the argument, not the quality.** Hybrid costs 860 ms/q
against lexical's 11 ms/q — **78× slower** for +0.064 R@5. Nothing about
`forge search` being interactive has changed, and §7's reasoning about what a
vector store would cost to operate stands untouched.

`hashing-v1-256c` remains a hashed bag of tokens and character 4-grams, not a
learned embedding. It is a floor, not a representative of what embeddings can
do — which cuts both ways now: a real model was never the thing being
rejected, and it is still unmeasured. §9 has the command.

**24 queries and 48 labels is a small set.** A 0.064 difference on it is a
handful of documents moving rank. Before any of this justifies shipping a
vector store, the set needs to be big enough for the difference to survive.

The rest of this document is left exactly as measured, with its original
numbers and its original conclusion, because it is a dated record of what was
true of that corpus.

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

`forge/evaluation/data/retrieval-v1.yaml` — **24 queries, 48 labels**, hand-built
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
  forge ingest /path/to/forge
FORGE_VAULT_PATH=/path/to/forge FORGE_STATE_DIR=/tmp/eval \
  forge embeddings build --provider hashing
FORGE_VAULT_PATH=/path/to/forge FORGE_STATE_DIR=/tmp/eval \
  forge retrieval-eval --methods lexical,semantic,hybrid --detail
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


---

## Semantic retrieval was a re-rank, not a retriever (2026-08-28)

**Found while wiring `forge ask`.** `SearchService` ran the lexical query
first and used embeddings only to reorder those hits. A span BM25 never
returned could not be recovered no matter how well it matched in embedding
space, so the semantic signal could improve the *order* of an answer set but
never its *contents*.

That is exactly the shape of the failure in the labelled set. Per-category
lexical recall:

| Category | R@10 |
|---|---:|
| technology | 1.000 |
| exact_concept | 0.933 |
| project | 0.750 |
| related_concept | 0.644 |
| dsa | 0.500 |
| **fuzzy_concept** | **0.100** |

Nearly all the loss sits in one category, and its queries are the ones that
describe a concept without naming it — *"stopping a language model from making
things up by giving it real passages"* for `rag.md`, *"technique for finding
the best contiguous subarray without recomputing"* for `Sliding Window.md`.
Those share almost no vocabulary with their targets, so BM25 never retrieved
them and re-ranking had nothing to fix.

**The evaluation and the product were measuring different systems.**
`RetrievalEvaluator` scores the query against *every* stored vector; the
production path scored only what lexical found. Any hybrid number from the eval
therefore described a retriever `forge search` did not implement.

Fixed: the candidate set is now the union of lexical hits and the strongest
direct vector matches, and the two signals are fused on one normalised scale.

**A second defect fell out of the same code.** Lexical scores were normalised
only for spans that had a vector, and spans without one kept their raw BM25
value — around 19 against a fused maximum of 1.0. Any span missing an embedding
therefore outranked every span with a good semantic match. This alone would
make hybrid score worse than lexical, independent of embedding quality, and it
is consistent with the measured result that hybrid degrades as semantic weight
rises.

**Not yet measured:** whether real embeddings now lift `fuzzy_concept`. The
stored vectors are still `hashing-v1-256c`, a hashed bag of tokens and
character 4-grams, which is not semantic at all — so the union retrieval has
nothing meaningful to union in. That measurement needs an embedding model, and
is the next step:

```powershell
ollama pull nomic-embed-text
forge embeddings build --provider ollama
forge retrieval-eval --methods lexical,semantic,hybrid --detail
```

Embedding ~7,000 spans is cheap in a way generation is not — this is a good use
of the local GPU, unlike extraction.

### `nomic-embed-text` needs asymmetric prefixes

Verified against Nomic's model card: stored text must be prefixed
`search_document: ` and queries `search_query: `. The provider applied neither,
which produces perfectly valid vectors sitting in the wrong region of the
space — nothing errors, and no stub test notices. Same class of defect as the
cloud provider's message ordering. Now applied, and the prefix scheme is part
of the model id (`nomic-embed-text+prefixed`) so prefixed and unprefixed
vectors can never share a derivation-cache entry.


## Real embeddings, measured (2026-08-28)

`nomic-embed-text` over 1,724 ingestion spans, whole vault, on the ASUS.

| Method | R@5 | R@10 | P@5 | MRR | misses |
|---|---:|---:|---:|---:|---:|
| lexical | 0.406 | 0.588 | 0.158 | 0.427 | 6 |
| **semantic** | **0.503** | **0.662** | **0.192** | **0.457** | **5** |
| hybrid(w=0.25) | 0.336 | 0.608 | 0.142 | 0.353 | 6 |
| hybrid(w=0.5) | 0.281 | 0.504 | 0.117 | 0.406 | 8 |
| hybrid(w=0.75) | 0.315 | 0.433 | 0.133 | 0.271 | 10 |

**Semantic beats lexical on every metric** — +0.097 R@5, +0.075 R@10, +0.030
MRR. That is the first evidence embeddings earn their place on this corpus, and
it replaces the earlier negative result, which was measured against a hashed
bag of tokens rather than a real model.

**Hybrid scored below both signals it blends, which is impossible.** A convex
combination of two rankings cannot be worse than either input unless the inputs
are not what they claim. They were not:

`_hybrid` built both score maps with `dict(...)` over lists sorted by
*descending* score. `dict` keeps the **last** pair, so every document collapsed
to its **worst** span — while `lexical` and `semantic` went through
`_to_documents`, which correctly keeps the best. Hybrid was blending each
document's weakest evidence against the other methods' strongest, and weighting
that signal harder is exactly why it degraded as `w` rose.

Fixed with an explicit `_best_per_key`. **The lesson is small and sharp:
`dict(pairs)` is a silent min when the pairs arrive sorted descending.**

A second defect fixed in the same pass: the evaluator embedded the query with
no task prefix while the stored spans were embedded as documents, so the two
sat in different regions of `nomic-embed-text`'s space. Semantic won *despite*
that handicap; the numbers above are a floor.

### Re-measured after both fixes (2026-08-28)

| Method | R@5 | R@10 | P@5 | MRR | misses |
|---|---:|---:|---:|---:|---:|
| lexical | 0.406 | 0.588 | 0.158 | 0.427 | 6 |
| semantic | **0.628** | 0.733 | **0.242** | **0.653** | 4 |
| hybrid(w=0.25) | 0.447 | 0.747 | 0.175 | 0.515 | **3** |
| hybrid(w=0.5) | 0.551 | 0.760 | 0.200 | 0.582 | **3** |
| hybrid(w=0.75) | 0.607 | **0.774** | 0.225 | 0.626 | 4 |

**Hybrid now beats lexical at every weight**, where before it lost at every
weight. The contradiction is gone, which is the confirmation that the min-per-
document aggregation was the cause rather than a coincidence.

**The query-prefix fix was worth more than the fusion fix.** Semantic alone
moved +0.125 R@5 and **+0.196 MRR** on nothing but embedding the query as
`search_query:` instead of as a document. A one-line omission was costing about
a third of the ranking quality, and it produced no error, no warning, and no
failing test — the vectors were valid the whole time.

**Lexical is now the worst option on every metric**, so `forge ask` no longer
defaults to it. It uses `w=0.75`: answering hands the model eight passages at
once, so getting the right document *into* that set (R@10 0.774 vs 0.733)
matters more than its rank within it, while the residual lexical weight still
catches exact-term queries an embedding blurs. Semantic-alone wins R@5, P@5 and
MRR and is within noise on 24 queries — this is the better available choice,
not a tuned optimum.

**`fuzzy_concept` is still the weak category: R@10 0.200.** It improved from
0.100 but remains far below every other category, and it is the one the whole
embeddings exercise was aimed at. Five queries is too few to conclude much, and
the honest reading is that describing a concept without naming it is still hard
here — worth its own investigation rather than another round of weight tuning.
