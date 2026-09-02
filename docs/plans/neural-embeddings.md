# Plan — Neural embeddings, and the fuzzy_concept ceiling

*Written 2026-09-02 for an implementing agent working on a machine with
network access. Every number below was measured, not estimated; the commands
that produced them are given so each can be reproduced before it is trusted.*

---

## 0. Why this work exists

Forge has an embedding pathway that is fully built — storage, cache
invalidation, fusion, evaluation — and has **never been run with a neural
model**. The provider it has been measured with, `hashing-v1-256c`, says of
itself in `engine/forge/embeddings/hashing.py`:

> *"It is not a neural sentence embedding, and it does not capture meaning.
> Two passages that share no vocabulary will score near zero no matter how
> synonymous they are."*

It exists because the environment the engine was built in blocked
`ollama.com` and `huggingface.co`, so a real model could not be obtained.
`OllamaEmbeddingProvider` was written as the intended production provider and
has never been exercised against the labelled set.

So every "semantic" and "hybrid" number Forge has ever reported is really
*character-and-token overlap in vector form*. **That is the gap this plan
closes.**

## 1. The measurements this starts from

Reproduce these first. If they do not match, stop and find out why before
changing anything — everything downstream is a delta against them.

```bash
export FORGE_VAULT_PATH=/path/to/forge
forge index
forge embeddings build --provider hashing
forge retrieval-eval --methods lexical,title,hybrid --provider hashing
```

Corpus at time of writing: **670 sources, 8,133 spans, 671 Markdown files,
~57,600 words.** Labelled set: 24 queries, 48 labels, 6 categories.

| Method | R@5 | R@10 | P@5 | MRR | Misses | Latency |
|---|---:|---:|---:|---:|---:|---:|
| lexical (FTS5/BM25) | 0.468 | 0.662 | 0.167 | 0.519 | 4 | **8.3 ms/q** |
| title (b=1.25) | 0.468 | 0.600 | 0.167 | 0.532 | 5 | 9.6 ms/q |
| hybrid (w=0.25) | 0.524 | 0.662 | 0.183 | 0.547 | 4 | 617 ms/q |
| **hybrid (w=0.50)** | **0.551** | **0.685** | **0.200** | 0.536 | 4 | 623 ms/q |
| hybrid (w=0.75) | 0.511 | 0.678 | 0.192 | **0.569** | 4 | 636 ms/q |

## 2. The hypothesis, stated so it can fail

Per-category recall is where the real signal is. `--json` gives it:

**R@5**

| Category | lexical | hybrid(w=0.5) |
|---|---:|---:|
| technology | 1.000 | 1.000 |
| project | 0.750 | 0.750 |
| exact_concept | 0.467 | 0.667 |
| dsa | 0.375 | 0.500 |
| related_concept | 0.300 | 0.133 |
| **fuzzy_concept** | **0.100** | **0.300** |

**R@10**

| Category | lexical | hybrid(w=0.25) | hybrid(w=0.5) | hybrid(w=0.75) |
|---|---:|---:|---:|---:|
| exact_concept | 0.933 | 0.933 | 0.933 | 1.000 |
| dsa | 0.500 | 0.625 | 0.625 | 0.625 |
| **fuzzy_concept** | **0.300** | **0.300** | **0.300** | **0.300** |

**`fuzzy_concept` R@10 is 0.300 under every method and every fusion weight.**
Hybrid moves hits from ranks 6–10 up into the top 5 — R@5 triples, 0.100 to
0.300 — but it never retrieves a document lexical did not already have. It
reorders; it does not discover.

That is precisely the behaviour the hashing provider's own docstring predicts.
A bag-of-features vector cannot connect two passages with no shared
vocabulary, and `fuzzy_concept` is *defined* as "the same idea phrased as a
user would say it, never using the page's exact title."

> **Hypothesis: a real neural embedder breaks the 0.300 `fuzzy_concept` R@10
> ceiling. Nothing else measured so far can.**

**How it fails:** if `nomic-embed-text` leaves `fuzzy_concept` R@10 at or
near 0.300, then vocabulary is not the bottleneck — the relevant documents
are absent from the candidate set for some other reason (chunking, retrieval
depth, or labels naming documents the index does not contain). In that case
**stop and diagnose rather than trying more models**; §6 says how.

This is the whole point of the exercise. A result either way is worth having,
and a negative one is worth publishing.

## 3. Prerequisites

**This cannot be done in the engine's build sandbox.** `huggingface.co` and
`ollama.com` both return 403 at the proxy, verified 2026-09-02. `pypi.org` and
GitHub release assets *are* reachable, which is how the spaCy model in §6 was
obtained — so if a transformer encoder is ever published as a GitHub release
wheel, that route works. Until then this needs a machine with open network
access.


```bash
brew install ollama          # or the installer from ollama.com
ollama serve                 # leave running
ollama pull nomic-embed-text # 274 MB, 768 dims
ollama list                  # confirm it is there
```

Verify the engine can see it before building anything — this is the check
that fails informatively rather than 8,000 silent zero-vectors later:

```bash
forge embeddings status --provider ollama
```

`available` must be `True` and `dimensions` non-zero. `dimensions: 0` means
no successful call has happened yet; the provider discovers dimensionality
from the first response rather than hardcoding it.

## 4. Tasks

### Task 1 — Make the embedding model selectable *(required; ~30 min)*

`engine/forge/cli/phase3.py:46` hardcodes the model:

```python
def _embedding_provider(settings: Any, name: str):
    if name == "hashing":
        return HashingEmbeddingProvider()
    if name == "ollama":
        return OllamaEmbeddingProvider(settings.llm.base_url)   # model not passed
    return NullEmbeddingProvider()
```

`OllamaEmbeddingProvider` accepts a `model` argument and defaults to
`nomic-embed-text`. Nothing can override it, so **comparing two embedding
models is currently impossible** — which is why Task 3 exists and why this
comes first.

- Thread an `embed_model: str | None` parameter through `_embedding_provider`.
- Add `--embed-model` to `forge embeddings build`, `forge embeddings status`
  and `forge retrieval-eval`. Default `None` — keep the provider's own
  default so existing invocations behave identically.
- Unit test: passing `--embed-model X` constructs a provider whose
  `model_id` contains `X`. No network needed; assert on the constructed
  object.

**Do not** change `DEFAULT_EMBED_MODEL`. The default stays `nomic-embed-text`
until a measurement says otherwise — that is the decision Task 3 produces.

### Task 2 — Build neural vectors and measure *(the main event; ~1 h)*

```bash
forge embeddings build --provider ollama --embed-model nomic-embed-text
forge retrieval-eval --methods lexical,hybrid --provider ollama \
    --embed-model nomic-embed-text
forge retrieval-eval --methods lexical,hybrid --provider ollama \
    --embed-model nomic-embed-text --json > /tmp/nomic.json
```

Expect the build to take minutes, not seconds: 8,133 spans at batch size 16.

Record, for every method: R@5, R@10, P@5, MRR, misses, latency, **and the
per-category breakdown** from the JSON. The per-category table is the
deliverable — the headline numbers will move for uninteresting reasons and
the category rows are what test the hypothesis.

**Vectors are namespaced by `model_id`, so this does not destroy the hashing
baseline.** `hashing-v1-256c` and `nomic-embed-text+prefixed` coexist in the
store; `forge embeddings status` lists what is present. You can re-run either
comparison at any time without rebuilding. Do not "clean up" the old vectors.

### Task 3 — Compare embedding models *(~1 h, only after Task 2)*

Only if Task 2 shows neural embeddings help. Same protocol, one build and one
eval per model:

| Model | Dims | Size | Note |
|---|---:|---:|---|
| `nomic-embed-text` | 768 | 274 MB | Baseline. Needs task prefixes — already handled. |
| `mxbai-embed-large` | 1024 | 670 MB | Generally stronger on retrieval benchmarks. |
| `bge-m3` | 1024 | 2.2 GB | Strong, much heavier. Try last. |

Report as separate rows. **Never average across models — they are different
instruments**, the same rule the assessment work already follows for local
vs cloud.

### Task 4 — Fix `forge ask`, which cannot use neural vectors at all *(bug; ~15 min)*

`engine/forge/cli/phase3.py:520`:

```python
answer = Answerer(
    SearchService(store, embeddings=_embedding_provider(settings, "hashing")),
    ...
).ask(question, semantic=semantic)
```

The provider is **hardcoded to `"hashing"`**. `forge ask --semantic` therefore
uses the non-neural embedder no matter what has been built, and no flag can
change it. Whatever Task 2 proves about retrieval quality, the answering path
cannot benefit from it.

- Add `--provider` (and `--embed-model`) to `forge ask`, defaulting to
  `hashing` so current behaviour is unchanged until measured.
- Test that the flag reaches the constructed `SearchService`.

Worth doing regardless of the Task 2 outcome: a hardcoded dependency that no
flag can reach is a defect on its own terms.

### Task 5 — Cross-encoder re-ranking *(~half day; independent of 1–4)*

The standard modern stack, and it should be **cheaper** than the current
623 ms hybrid rather than more expensive, because it scores a fixed 30
candidates instead of comparing against every stored vector.

- Retrieve top-30 lexically (8.3 ms).
- Re-rank those 30 with a cross-encoder — `BAAI/bge-reranker-base` or
  `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`.
- Add it as a `rerank` method in `RetrievalEvaluator`, alongside `title` and
  `hybrid`. Follow the pattern added in commit `02fdfdf`: sweep it, give it a
  no-op anchor, and assert the anchor reproduces the baseline exactly.
- Add `sentence-transformers` as an **optional extra**, never a core
  dependency — the same rule `agent` and `tui` follow. `forge index` on a
  fresh clone must not pull PyTorch.

A cross-encoder can only re-rank what retrieval already found, so **it cannot
break the `fuzzy_concept` R@10 ceiling either.** Expect it to help MRR and
P@5, not R@10. If it does raise R@10, something is wrong with the
understanding above and that is worth chasing.

## 5. Rules

**Never tune against the labelled set.** From `runner.py`: *"Nothing here
tunes retrieval. The set is for measuring; optimizing against it would make
the numbers meaningless."* Pick model and weight on reasoning, measure once,
report. If you find yourself trying values to see which scores best, you are
building a number that means nothing.

**Report negative results as prominently as positive ones.**
`docs/research/retrieval-baseline.md` is the record. It already carries two
findings that reversed prior conclusions — embeddings rejected then accepted,
and a shipped title boost measured as a regression. A third reversal is
normal and belongs there in the same voice.

**One variable at a time.** The last re-baseline could not attribute its
cause because the corpus and the chunker changed together, and the document
says so rather than guessing. Do not repeat that: change the embedder, hold
everything else.

**The vault is read-only to the engine.** Nothing here writes to it. Derived
state lives in `.forge/` and is rebuildable.

## 6. The diagnostic in §6 has already been run — read this before starting

This section originally said what to check *if* the hypothesis failed. Two of
those checks have since been done, and they change the plan. Full write-up in
`docs/research/retrieval-baseline.md`.

**A third method was tried and failed.** `SpacyEmbeddingProvider` (mean-pooled
300-d `en_core_web_md` vectors) is in the tree, wired to `--provider spacy`,
and measured. It is far worse than hashing — semantic alone misses 15 of 24
queries — because mean-pooling static vectors over a span averages it toward
the corpus mean. It left `fuzzy_concept` R@10 at 0.300, the third method to do
so. **Do not spend time on static word vectors.**

**But the ceiling is now explained.** Every `fuzzy_concept` target was probed
against lexical search at depth 3000. `label_rot` is empty and every labelled
document is present in the candidate set:

| Target | Rank |
|---|---:|
| `DSA/01_Patterns/Sliding Window.md` | 1 |
| `Technologies/Docs/rag.md` | 8 |
| `DSA/01_Patterns/Topological Sort.md` | 15 |
| `DSA/09_CheatSheets/Sliding Window Cheat Sheet.md` | 35 |
| `Technologies/Docs/vector-databases.md` | 112 |
| `DSA/03_DataStructures/Disjoint Set.md` | 211 |
| `DSA/01_Patterns/Union Find.md` | 231 |

Nothing is missing; everything is mis-ranked. **Indexing and chunking are
exonerated** — do not go looking there. BM25 puts these 100–200 positions too
low because the queries share almost no vocabulary with their targets, which is
what `fuzzy_concept` was designed to do.

**Consequences for the tasks below:**

1. **Task 5 (cross-encoder) will not break the ceiling and should be
   re-scoped.** A re-ranker only reorders what it is handed, and four targets
   sit below rank 30. Re-ranking the top 30 cannot surface a document at rank
   231. Expect it to improve MRR and P@5 — worth having — but do not expect
   R@10 to move, and do not treat a flat R@10 as a bug. If you want it to have
   a chance at the ceiling, the shortlist has to be ~250 deep, and scoring 250
   candidates per query is a different cost proposition that needs measuring
   rather than assuming.
2. **Task 2 is still the right experiment, and now for a sharper reason.** A
   bi-encoder scores every document independently of its lexical rank, so it
   is the only instrument tested that *can* move a document from 231 into the
   top 10. It never sees rank 231.
3. **Raising `RETRIEVAL_DEPTH` is not a fix.** It admits those documents to the
   evaluation window but they still rank below 30 and contribute nothing to
   R@5 or R@10. Depth is a symptom.

The encouraging part of the negative result: even in a bad instrument, the
fuzzy-specific signal survived — semantic-alone raised `fuzzy_concept` R@5 from
0.100 to 0.200, and `hybrid(w=0.75)` reached 0.300, the best fuzzy R@5
measured, while wrecking every other category. The signal is real and the
instrument was too blunt.

## 7. Acceptance

- [ ] Baseline in §1 reproduced before any change
- [ ] `--embed-model` works on build, status and eval; unit-tested
- [ ] Neural vectors built; `embeddings status` shows non-zero dimensions
- [ ] Full sweep run with `--provider ollama`, per-category JSON captured
- [ ] `fuzzy_concept` R@10 reported against the 0.300 ceiling, either way
- [ ] `docs/research/retrieval-baseline.md` updated with a dated section,
      the exact command, the corpus size it was measured on, and the verdict
- [ ] `forge ask` no longer hardcodes the hashing provider
- [ ] `python -m pytest tests` green — 1,076 passed, 42 skipped at
      commit `02fdfdf`; recount rather than quoting this
- [ ] Hashing vectors still in the store, so the comparison stays reproducible

## 8. Two open decisions this does not settle

Both are recorded in `docs/research/retrieval-baseline.md` and are the
owner's calls, not the implementing agent's:

1. **`TITLE_BOOST`** — the answering service passes 1.25; measured as a
   regression at every value (−0.0625 R@10, −0.200 on `fuzzy_concept`). It
   was measured on *document recall* and used for *span selection*, so the
   evidence is suggestive rather than decisive for that path.
2. **Hybrid as the retrieval default** — best recall available, at ~75× the
   latency. Neural embeddings change both sides of that trade, so re-open it
   only after Task 2.

## 9. What not to do

- **Do not fine-tune a language model on this corpus.** ~78K tokens is
  roughly 0.0008% of GPT-2 small's training data. It is enough to shift
  style, not to install knowledge, and a fine-tune teaches a model that a
  *shape* of statement is likely rather than that it is *true* — producing
  confident, unattributable claims. That is the exact failure the provenance
  floor exists to prevent.
- **Do not train a GNN on the concept graph.** ~500 concepts and 4,703 edges
  is too small, and no task is defined that it would serve.
- **Do not train anything supervised on the labelled set.** 24 queries and
  48 labels is an evaluation set, not training data. Using it as both
  destroys its only purpose.
- **Do not replace the hashing provider.** It is the zero-dependency path
  that keeps the embedding pipeline testable offline and in CI. It should
  stop being the *default* for quality claims, not stop existing.
