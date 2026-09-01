# Extraction Cost — Measured, and a Runbook for Doing It

*What it actually costs to turn the vault into concepts and claims, which
subsets are worth spending that on, and how to run it without babysitting.*

**Headline: the call count is settled — 3,372 calls for the whole vault, 196
for `Technologies/Docs/`. The per-call latency is not. The 63 s/call borrowed
from the Phase 4 assessment eval understates real extraction by roughly 3.6×
(§2 — this was stated as 7× until 2026-08-19; see the correction there). The
leading suspect was that Qwen3 reasons before every answer because nothing
tells it not to — that has now been tested, and turning reasoning off buys 2.5×
at the cost of a false-positive conflict, so it is rejected (§2).
Extraction is resumable at document granularity, so it does not need to finish
in one sitting.**

---

## 1. The correction that produced this document

An earlier estimate put full-vault extraction at ~236 hours and concluded it
was infeasible. That number was wrong, and the reason is worth recording
because it is an easy mistake to repeat.

Forge chunks text in **two different places**:

| Command | Purpose | Spans over the vault |
|---|---|---:|
| `forge index` | corpus indexing / retrieval | **7,237** |
| `forge ingest` | the ingestion pipeline | **1,697** |

Extraction runs inside the **ingestion** pipeline, over ingestion spans. The
earlier estimate counted `forge index` spans — a 4× overcount — and then
applied the per-call latency to it. The two figures come from different
chunkers with different targets, and they are not interchangeable.

Correct method: count ingestion spans that clear `MIN_SPAN_CHARS = 40`, apply
the per-document `max_spans` cap, multiply by **2 calls/span** (`_concepts`
then `_claims`), multiply by measured latency.

---

## 2. Measured cost

Inputs, all measured rather than assumed:

| Quantity | Value | Source |
|---|---|---|
| Ingestion spans, whole vault | 1,697 | `forge ingest .` |
| Spans ≥ 40 chars, after the 12/doc cap | **1,686** | counted from the store |
| Model calls per span | **2** | `extraction/extractor.py` |
| Local latency | **63 s/call** | `provider-availability.md` §6 |

| Scope | Spans | Calls | At 63 s/call |
|---|---:|---:|---:|
| **Whole vault** | 1,686 | 3,372 | 59 h |
| `Projects/` | 160 | 320 | 5.6 h |
| `Technologies/Docs/` | 98 | 196 | 3.4 h |
| `DSA/01_Patterns/` | 94 | 188 | 3.3 h |
| `Courses/` | 37 | 74 | 1.3 h |

### The 63 s/call figure does not survive contact with extraction

**It was measured on a different task.** It comes from the Phase 4 *assessment*
eval — short prompts, small outputs. Extraction sends a full span and asks for
up to 15 concepts or 10 claims. The first real extraction run, on the same
hardware and model, produced:

| Span | Wall time (2 calls) | Implied per call |
|---|---:|---:|
| 3 | 478.6 s | ~239 s |
| 4 | 704.5 s (one timeout + retry) | ~235 s |
| 5 | 183.3 s | ~92 s |

Mean ≈ **455 s/span** = **~228 s/call**, roughly **3.6× the assessment figure**,
and one later gap exceeded 60 minutes with no span completing at all.

> **Correction, 2026-08-19.** This section previously said **7×** and projected
> `Technologies/Docs/` at **~26 h**. Both were wrong: 455 s is *per span* and
> the 63 s baseline is *per call*, and a span is two calls. Comparing them
> directly double-counted. Like-for-like it is 228 s/call vs 63 s/call, or
> equivalently 455 s/span vs 126 s/span — **3.6× either way**. This is the same
> unit mismatch as the 7,237-vs-1,697 span error in §1 that caused this document
> to be written, which is worth stating plainly: *check the denominator before
> dividing two latency numbers.* The whole-vault conclusion is unchanged, since
> it was stated qualitatively.

Re-projected on extraction's own measured rate:

| Scope | Spans | At 455 s/span | Excluding the timeout span |
|---|---:|---:|---:|
| **Whole vault** | 1,686 | 213 h (~9 days) | 155 h |
| `Projects/` | 160 | 20.2 h | 14.7 h |
| `Technologies/Docs/` | 98 | **12.4 h** | **9.0 h** |
| `DSA/01_Patterns/` | 94 | 11.9 h | 8.6 h |
| `Courses/` | 37 | 4.7 h | 3.4 h |

The right-hand column drops span 4, whose 704.5 s included a timeout and a full
retry; the true rate is somewhere between the columns. The practical consequence
of the correction is that `Technologies/Docs/` is **an overnight run, not a
two-day one** — which changes whether it is worth starting at all.

Three data points from one document are not a characterisation either — they
are recorded here so the next estimate starts from extraction's own numbers
rather than borrowing another task's.

### The likely cause, and what it costs to check

Qwen3 is a **reasoning model**, and Ollama runs it with reasoning enabled
unless told otherwise. Forge was not telling it otherwise. Every extraction
call therefore generated a full chain of thought — at full token cost, in the
same wall clock — which the structured-output path then discards.

`FORGE_OLLAMA_THINK=0` sends Ollama's `think: false`. That is a change to what
the model *does*, not a tuning flag, so it is opt-in, and it appends
`+nothink` to the extractor's model id — think-on and think-off results can
never share a derivation cache entry, and must never be averaged in an
evaluation.

**Measured, 2026-08-19 — and the answer is no.** Full write-up in
`provider-availability.md` §8; the short version:

| | Think on | Think off |
|---|---:|---:|
| Reported mean/case | 112,078 ms | 21,867 ms |
| Typical case | ~45 s | ~18 s |
| Classification accuracy | 5/5 | **4/5** |

Latency dropped — **2.5×** typical-vs-typical, which is the number to plan with
(the 5.1× the two report lines imply is inflated by think-on's 345 s
timeout-and-retry outlier). But accuracy fell, and it fell on
`insufficient-partial-overlap`: with reasoning off the model turned partial
overlap into `POTENTIAL_CONFLICT` and raised a `CLAIM_CONFLICT` proposal. That
is a **false-positive conflict**, the exact failure mode
`provider-availability.md` §4 calls the largest open risk carried out of
Phase 4. Validity and grounding both held at 1.00, so this is a specific loss of
discrimination, not a general collapse.

**So reasoning-off is rejected for assessment, and remains unmeasured for
extraction.** The experiment could only be run against the assessment set:
`forge/evaluation/data/` holds `assessment-v1.yaml` and `retrieval-v1.yaml` and
nothing else, so **there is no extraction-quality eval** to grade the task this
section actually cares about. Extraction asks a different question and reasoning
may matter less there — but that is a guess, and running the vault with
`+nothink` on the strength of it means spending a full pass whose output nothing
can score, with a cache key (§3) that guarantees redoing it if the answer turns
out to be no.

Which leaves the original conclusion standing, unimproved: at ~455 s/span this
hardware does not extract the whole vault in reasonable time. The two ways
forward are **extract selectively** (§4's ordering — `Technologies/Docs/` first)
or **measure the cloud path**, which remains unmeasured. Building an
extraction-quality eval is the prerequisite for revisiting reasoning-off at all.

The per-document cap rarely binds: ingestion chunking produces roughly 5–12
usable spans per document, so `--max-spans 12` truncates almost nothing. Raising
it would buy very little coverage and cost proportionally more.

**Cloud, for comparison.** Span text totals ~2.19 M chars ≈ 547 k tokens; prompt
overhead is negligible (SYSTEM 47 tok, instructions 82–113 tok). Projected input
~3.0 M tokens, output ~2.7 M tokens — the output figure assumes ~200 tokens of
JSON per call and is **an assumption, not a measurement**. At $3/$15 per Mtok
that is roughly $50 for the entire vault. Cloud latency and cloud extraction
quality are both **unmeasured**; see `provider-availability.md`.

---

## 2b. The first complete run (2026-08-19) — and the bug it exposed

`Technologies/Docs`, ASUS, qwen3:8b, reasoning on.

```
19 source(s) in 20370.86s | 208 spans | 1386 concepts | 1170 claims | 2169 proposals
LLM calls: 416  cache: {'hits': 0, 'misses': 19, 'writes': 19}
```

**5.66 h, 49.0 s/call, 97.9 s/span.** Every projection above is per-call, and
this is the first per-call figure taken from a *complete* scope rather than
three spans of one document. It is **4.6× faster than the 455 s/span** the
three-span sample implied.

Both numbers are real. They differ because they measured different span sizes,
which is the finding:

### The run extracted over the wrong spans

`forge index` and `forge ingest` both write to the `spans` table, for different
jobs. Phase 1 produces heading-delimited spans for retrieval
(`chunk_strategy="heading"`); ingestion produces structurally grouped,
sentence-split spans for evidence (`"structural/0.2.0"`). They share a document.

The unchanged-source short-circuit checked only the content hash, so a vault
indexed *before* it was ingested — the order every runbook here recommends —
extracted over Phase 1's spans:

| | Spans | Calls |
|---|---:|---:|
| Clean store, ingestion chunker | **98** | **196** |
| After `forge index` (what actually ran) | **208** | **416** |

So the run cost **2.1× more model calls than necessary**, over boundaries the
extraction prompt was never written for. Reproduced locally and fixed: ingestion
now reads back only spans its own chunker produced. Phase 1's are kept rather
than deleted — retrieval is still using them. Regression tests in
`tests/integration/test_phase2_ingestion.py::TestChunkerProvenanceOnUnchangedSources`.

This is the same *class* of bug as the 2026-08-15 one in §3: an "unchanged"
short-circuit answering a narrower question than the caller needed. Twice now,
so it is worth stating as a rule: **"unchanged" describes the source, never the
derived state.**

### What that does to the latency numbers

Index spans are roughly half the size of ingestion spans, and per-call latency
fell **4.6×** for a ~2× reduction in span size. That is strongly superlinear,
but it rests on two samples at two sizes and one of them was three spans, so
treat it as a direction to investigate, not a law. The honest summary:

* **49 s/call over ~1,100-char spans** — measured, complete scope, n=416.
* **~228 s/call over ~2,400-char spans** — measured, n=6, one timeout included.
* A corrected `Technologies/Docs` run makes 196 calls over the larger spans. Its
  wall time is **not predictable from either figure** and is the next thing
  worth measuring.

The `[unchanged] … no work done` line printed against every source during this
run was also false — extraction was running. Fixed: an unchanged source that
still extracts now reports what it produced.

### The review bottleneck is now real, not projected

**2,169 proposals** from 19 documents. §5 estimated ~5 h of review for the whole
vault at one claim per span; one twentieth of the vault has produced 6 h of
reading at the same optimistic 10 s each. Scaled naively, the full vault is on
the order of **100+ hours of human review** — which is the argument for
extracting selectively, made concrete.

Before approving any of it, run the grounding audit (§3 of
`provider-availability.md` has the background):

```powershell
forge proposals audit-grounding
```

These proposals were extracted under the pre-2026-08-19 bag-of-words grounding
rule, which accepted quotes reassembled from the span's own vocabulary. The
audit re-checks every stored quote against the order-preserving rule at zero
model calls.

### First measured quote-fidelity rate (2026-08-19)

`forge proposals audit-grounding` over the 1,170 claims from that run:

```
1170 quote(s) checked, 30 ungrounded (2.56%).
```

**2.56%** is the first number Forge has on how often a stored "quote" is not
actually a quote. It is one model, one scope, and one chunking, so it is a
measurement rather than a rate — but it is no longer unmeasured.

**The failures are not hallucinations. They are tables.** 29 of 30 are cheat
sheet rows rewritten as prose:

| Source (a table row) | What the model stored as a *quote* |
|---|---|
| `\| az login \| Sign in \|` | "The command `az login` is used to login." |
| `\| docker ps \| List running containers \|` | "The command to list running Docker containers is `docker ps`." |
| `\| kubectl logs -f pod \| Live logs \|` | "The command `kubectl logs -f pod` is used to get live logs of a pod." |

Every fact in those sentences is correct and present in the span. What is absent
is the *sentence* — the model composed prose from tabular content and cited it
as verbatim. The old bag-of-words rule saw the words and accepted it; the
order-preserving rule sees that no such sentence exists and drops it. Both are
behaving correctly, and the drop is right: `EvidenceRelation.QUOTES` asserts the
span contains this text, which is a deterministic claim a model is not permitted
to make.

The 30th is different and worth the whole audit on its own:

```
'Increment | `INNE key`'
```

There is no `INNE` command in Redis; the source says `INCR`. That one is a
genuine fabrication in a fragment that looks structurally like a real table row,
and no amount of reading 2,169 proposals by hand would reliably have caught it.

**What this says about the prompt, not the model.** The failure clusters
entirely in `azure.md`, `docker.md`, `kubernetes.md` and `redis.md` — the
documents with large command tables. Asking for a verbatim quote from tabular
content is asking for something a table often does not contain: a sentence. That
is an extraction-prompt problem, and it is the concrete first case for the
extraction-quality eval that §2 says does not exist yet.

Reject them in one pass; the audit knows which they are:

```powershell
forge proposals audit-grounding --reject            # dry run, the default
forge proposals audit-grounding --reject --no-dry-run
```

Already-decided proposals are skipped — a human decision is not the audit's to
overturn.

### The clean number: think-on, ingestion spans (2026-08-20)

One document, a fresh store, reasoning on, no cache and no timeouts —
the first extraction measurement with no confound in it.

```
1 source(s) in 1311.16s | 4 spans | 17 concepts | 30 claims | 44 proposals
LLM calls: 8  cache: {'hits': 0, 'misses': 1, 'writes': 1}
```

**327 s/span, 164 s/call.** Per-span times were 179 / 502 / 482 / 146 s — a
3.4× spread within one document, so span size dominates and any single-document
mean carries wide error bars.

This **supersedes the 228 s/call** figure from the three-span sample, which
included a timeout and a retry. It is also 3.3× the 49 s/call of the first full
run — consistent with that run being reasoning-off over spans half this size,
and notably *less* than the ~5× that a naive stacking of both effects predicts.

| Scope | Spans | Think-on, ingestion spans |
|---|---:|---:|
| **`Technologies/Docs/`** | 98 | **8.9 h** |
| `DSA/01_Patterns/` | 94 | 8.6 h |
| `Projects/` | 160 | 14.6 h |
| `Courses/` | 37 | 3.4 h |
| Whole vault | 1,686 | 153 h (~6.4 days) |

So a correct `Technologies/Docs` run is **an overnight job, and about 1.6×
the wall time of the confounded one** — 8.9 h against 5.66 h. The extra cost
buys reasoning-on output over the boundaries the prompt was designed for.

Whole-vault extraction on this hardware remains out of reach at 6.4 days.
Selective extraction, or the still-unmeasured cloud path, are the only routes
to full coverage.

### Did the `0.3.0` prompt work? Partly — and the residue is not a prompt problem

Same document, reasoning on, new prompt. All 14 concept proposals and 20 of 30
claims read by hand.

| Failure mode | `0.2.0`, think-off | `0.3.0`, think-on |
|---|---:|---:|
| Concept junk (`RAM`, `Answer`, `maxmemory`, `VARCHAR(n)`) | ~60% of sample | **0** |
| Document-referential claims ("The text provides…") | ~8% | **0** |
| Near-duplicate claims | ~12% | **25%** |
| Table content mishandled | 2.56% ungrounded | 15%, new shape |

**The two rules that could work, did.** Every one of the 14 concepts is a real
RAG concept — `Reranking`, `Recall@k`, `Metadata filtering`, `hybrid (keyword +
vector) search`. Nothing in the `RAM`/`Fluency` class survived. No claim refers
to the document.

**The two that failed, failed structurally.**

*Deduplication cannot be a prompt rule.* The extractor sends **one span per
call**, so the model physically cannot see that another span already produced
"Evaluating retrieval quality involves calculating recall at k" when it emits
"Evaluating a RAG system requires a labeled evaluation set with recall@k
metrics". Three clusters in 20 claims, 5 redundant. The same applies to
concepts: `RAG` / `Retrieval Augmented Generation`, `Reranking` / `Reranker`,
`Hybrid search` / `hybrid (keyword + vector) search` — three alias pairs in 14.
Asking the prompt to fix this was a category error. **Cross-span dedup is
deterministic post-processing, and belongs in code.**

*Tables changed shape rather than resolving.* The `0.3.0` rule stopped the model
rewriting rows as prose, which removed the grounding failures. But it now emits
raw table cells as claim *statements*: "How many chunks retrieved; higher recall
but more dilution/cost". The quote is honest; the statement is a fragment, not
an assertion. Fixing this properly means the chunker or the prompt treating a
parameter table as a unit rather than a source of sentences.

**Usable-claim rate is ~60% either way, but the residue changed character** —
from junk that must be rejected to duplicates that can be merged mechanically.
That is the more tractable failure, and it does not require another model run:
like the grounding audit (§2b), dedup is deterministic and applies retroactively
to proposals already extracted. **It is not a reason to delay a run.**

---

## 2c. First trustworthy extraction-quality measurement (2026-08-31)

`openai/gpt-oss-120b` via Groq, `extract-prompts/0.3.0`, `extraction-v1` (6
cases), **all 6 completed** — so the run is scored, not marked UNTRUSTWORTHY.

| metric | value | notes |
|---|---|---|
| **junk rate** | **0.000** | 0 of the 25 forbidden strings emitted |
| recall | 0.472 | see the caveat below — this understates |
| grounding | 1.000 | over 23 claims, drops included in the denominator |
| calls | 12 | ~1.0-1.6 s each |

**Every number here needed four separate defects fixed before it existed.** A
decommissioned model id, a health check that only verified a credential was
present, a preset ceiling larger than the host's whole per-minute budget, and a
retry loop with no sleep. Three of the four produced *plausible output with
nothing raised* — the house failure mode.

### What the junk rate does and does not say

**It says the `0.3.0` prompt's exclusion rules work on the failures they were
written for.** The strings on that list are real observed output from the
2026-08-19 qwen3:8b run — `maxmemory`, `VARCHAR(n)`, `.dockerignore`, `Answer`,
`git commit`. None came back. That is the single most useful thing measured
about extraction so far, because those are what would have entered the graph
permanently.

**It does not say the output contains no junk.** The denominator is 25
specific strings, not the space of bad concepts. The run emitted `Dockerfile`,
`layers` and `cache reuse` as extras; `Dockerfile` is a file name, which the
prompt forbids in general but which is not on this list. **A junk rate of 0.000
means "none of the known failures recurred", not "nothing bad was produced."**

### Why recall=0.472 understates, and why it has not been "fixed"

Reading the per-case detail, most misses are the same concept under a different
surface form:

| case | expected (missed) | actually emitted |
|---|---|---|
| sliding-window | `sliding window` | `sliding window pattern` |
| git-history | `rebase` | `rebasing` |
| docker-images | `layer caching` | `layers`, `cache reuse` |

`score_case` matches through `normalize()`, which folds case and punctuation
but not morphology or modifier suffixes. `proposals/dedup.py` already has a
`_stem()` that would collapse `rebase`/`rebasing` — the eval does not use it.

**That change has deliberately not been made in the same breath as reading the
result.** Loosening the matcher can only move recall up, never down, and
deciding it is "the right matcher" immediately after seeing that it improves
the score is motivated reasoning — the same defect as picking a threshold to
suit an outcome. If alias matching is added it should be argued on its own
terms, applied to the forbidden list as well as the expected list, and the
before/after both recorded.

**Treat 0.472 as a floor on recall under exact normalized matching**, not as a
statement about how much the model found.

### What this does not establish

Six cases cannot produce a precision figure, and one run cannot produce a rate.
The comparison this enables is *relative* — prompt A against prompt B, model X
against model Y, on the same set — which is exactly what was missing when the
`0.3.0` rewrite and the `+nothink` experiment had to be recorded as
unjudgeable.

---

## 3. Why this is resumable

Extraction results are cached under a derivation key covering content hash,
processor version, model id, prompt version, and schema version. The cache is
written **after each source completes**. Consequences:

* **Re-running a completed scope costs zero calls.** Interrupt a run, restart
  it, and finished documents are served from the cache.
* **Deterministic ingestion first, extraction later, is a supported order** —
  and is the normal one, since extraction is opt-in. Until 2026-08-15 it was
  not: the unchanged-source short-circuit compared content hashes only, so
  `--extract` over an already-ingested vault skipped every source, made zero
  calls, and reported success. The same bug broke resumability, since a
  restarted run skipped sources it had ingested but never extracted. Fixed;
  regression test in
  `tests/integration/test_phase2_ingestion.py::test_extract_runs_on_a_source_ingested_without_extraction`.
* **Interrupting mid-document loses that document's work** — up to 12 spans ×
  2 calls ≈ 25 minutes locally. Document granularity, not span granularity.
* **Changing the model, prompt version, or schema version invalidates the
  cache by design.** That is correct — candidates from a different model are
  different derived objects — but it means a model swap re-spends the full cost.

Nothing is written to the Markdown vault. Extraction produces **proposals**;
they become canonical only through explicit approval and activation.

---

## 4. Runbook

Recommended order — highest concept density first:

```powershell
# Windows / ASUS. Provider and model come from ~/.config/forge/forge.env, so
# the only thing worth overriding per-run is the timeout.
#
# PICK THIS DEliberately — bigger is NOT safer. Ollama retries twice, so the
# worst case for one call is timeout x 3 before any work resumes. At 900 s that
# is 45 minutes of waiting per stalled call, and 2026-08-20 measured exactly
# that: one span burned 2,792 s, of which 2,700 s was three timeouts and 92 s
# was the attempt that succeeded. The stalls are hangs, not slow completions —
# the retry that worked was among the fastest calls of the run.
#
# Successful calls have measured 73-250 s on this hardware. 420 s gives ~1.7x
# headroom over the slowest observed success and caps a stalled call at 21
# minutes instead of 45.
$env:FORGE_LLM_TIMEOUT="420"

# Check what will actually be used before committing hours to it:
forge status

forge ingest "Technologies/Docs" --extract -v    # ~9-12 h
forge ingest "DSA/01_Patterns"   --extract -v    # ~9-12 h
forge ingest "Projects"          --extract -v    # ~15-20 h
```

`-v` is what makes a multi-hour run observable. Two progress signals are
emitted:

* `extraction_span_complete span=3 of=7 seconds=61.4` — per span. Without this
  a single document is ~25 minutes of silence, indistinguishable from a hang.
* `ingest_progress source=4 of=19 elapsed_minutes=48.2 estimated_remaining_minutes=181.0`
  — per document. The estimate is the mean per-source time **of this run**
  projected forward; it swings early and while cached sources fly past. It is a
  progress indicator, not a prediction.

Then review what came out — nothing is canonical yet:

```powershell
forge proposals list --status PENDING
forge proposals show <id>
forge proposals approve <id>
```

### Before the first run — done, 2026-08-19

Four concept-name collisions were undecided. Extracting with them open means
every downstream concept inherits an unresolved identity, so they were settled
first. Now recorded in `config/concept-identity.yaml` (committed, so it travels
with the vault rather than living on one machine):

```
Binary Search   -> pattern/Binary Search
Heap            -> pattern/Heap
Trie            -> pattern/Trie
weekly-review   -> templates/weekly-review
```

The three DSA names default to `DSA/01_Patterns/` because that is where the
vault's depth is and because prose in `DSA/` and `Projects/` is almost always
discussing the solving technique rather than the bare construct. `weekly-review`
defaults to the reusable template, with the Competitive-Programming file reading
as one filled-in instance of it.

These were naming decisions about the vault, not inferences — the engine
deliberately will not guess, and `forge identity decide "<name>"
<qualified-name>` is how you change one. `forge identity clear` reopens a
collision if a decision turns out wrong.

---

## 5. The cost that is not GPU time

Extraction produces proposals; a human approves them. If the vault yields even
one claim per extracted span, that is ~1,700 proposals. At an optimistic 10
seconds of review each, that is **~5 hours of reading** — and unlike GPU time it
cannot be parallelized, bought, or run overnight.

This is the real argument for extracting selectively rather than exhaustively:
the binding constraint is review attention, not inference throughput.

---

## Related

* [`provider-availability.md`](provider-availability.md) — where the 63 s/call
  figure comes from, what the 5/5 assessment result does and does not
  establish, and (§8) the reasoning-off experiment this document asked for.
* [`retrieval-baseline.md`](retrieval-baseline.md) — the deterministic
  retrieval path, which needs no extraction at all.
* [`../architecture/phase-2-implementation.md`](../architecture/phase-2-implementation.md)
  — the ingestion pipeline, chunking, and derivation keys.
