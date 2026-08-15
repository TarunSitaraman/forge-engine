# Extraction Cost — Measured, and a Runbook for Doing It

*What it actually costs to turn the vault into concepts and claims, which
subsets are worth spending that on, and how to run it without babysitting.*

**Headline: the whole vault is ~59 hours of local GPU time, not the ~236 hours
previously estimated. `Technologies/Docs/` — the highest-value subset — is
~3.4 hours, a single overnight run. Extraction is resumable at document
granularity, so it does not need to finish in one sitting.**

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

| Scope | Spans | Calls | Local (63 s/call) |
|---|---:|---:|---:|
| **Whole vault** | 1,686 | 3,372 | **59 h** |
| `Projects/` | 160 | 320 | 5.6 h |
| `Technologies/Docs/` | 98 | 196 | **3.4 h** |
| `DSA/01_Patterns/` | 94 | 188 | 3.3 h |
| `Courses/` | 37 | 74 | 1.3 h |

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

## 3. Why this is resumable

Extraction results are cached under a derivation key covering content hash,
processor version, model id, prompt version, and schema version. The cache is
written **after each source completes**. Consequences:

* **Re-running a completed scope costs zero calls.** Interrupt a run, restart
  it, and finished documents are skipped.
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
# Windows / ASUS. Raise the timeout first: the default 120 s is marginally
# too low for an 8B model on ~6 GB VRAM (one call timed out at the default).
$env:FORGE_LLM_PROVIDER="ollama"
$env:FORGE_MODEL_DEFAULT="qwen3:8b"
$env:FORGE_LLM_TIMEOUT="300"

python -m forge.cli.main ingest "Technologies/Docs" --extract -v    # ~3.4 h
python -m forge.cli.main ingest "DSA/01_Patterns"   --extract -v    # ~3.3 h
python -m forge.cli.main ingest "Projects"          --extract -v    # ~5.6 h
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
python -m forge.cli.main proposals list --status PENDING
python -m forge.cli.main proposals show <id>
python -m forge.cli.main proposals approve <id>
```

### Do this before the first run

Four concept-name collisions are undecided. Extracting first means every
downstream concept inherits an unresolved identity:

```
Binary Search   pattern/Binary Search      vs  algorithm/Binary Search
Heap            pattern/Heap               vs  data-structure/Heap
Trie            pattern/Trie               vs  data-structure/Trie
weekly-review   competitiveprogramming/…   vs  templates/weekly-review
```

Resolve with `forge identity decide "<name>" <qualified-name>`. These are
naming decisions about your own vault — the engine deliberately will not guess.

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
  figure comes from, and what the 5/5 assessment result does and does not
  establish.
* [`retrieval-baseline.md`](retrieval-baseline.md) — the deterministic
  retrieval path, which needs no extraction at all.
* [`../architecture/phase-2-implementation.md`](../architecture/phase-2-implementation.md)
  — the ingestion pipeline, chunking, and derivation keys.
