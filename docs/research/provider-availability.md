# Provider Availability — What Was and Was Not Measured

*Phase 4. A record of which inference providers could actually be exercised in
this environment, what that permits us to claim, and what it does not.*

**Update, 2026-08-14 — the local path has now been measured.** Qwen3 8B on an
RTX 4050 scored 5/5 on the assessment set with perfect structured-output
validity and perfect grounding. Details in §6. The cloud path remains
unmeasured. The original Phase 4 finding is preserved below because the
distinction it draws — measured pipeline vs unmeasured model — is the reason
the measurement was worth making, and because a document that quietly rewrites
its own history is worth less than one that shows its work.

**Original headline (Phase 4 development): no real model produced a single
assessment. The pipeline is fully measured; model quality is entirely
unmeasured.** Those are different claims and this document keeps them apart.

Companion to
[`local-model-capability-spike.md`](local-model-capability-spike.md), which
covers the same question for local models in Phase 1.

---

## 1. Why this document exists

The Phase 4 brief says, twice: *"If either provider is unavailable: say so. Do
not fabricate results."* The temptation is real, because a table of latency and
accuracy numbers is exactly what makes a phase report look finished. This page
exists so the absence is as visible as any measurement would have been.

---

## 2. Provider status in this environment

| Provider | Configured | Network reachable | Credential present | Inference run |
|---|---|---|---|---|
| Ollama (local) | yes | **no** — nothing listening on `localhost:11434` | n/a | **no** |
| Ollama (remote / ASUS) | yes | **no** — host is not on this network | n/a | **no** |
| Cloud (Anthropic) | yes | **yes** | **no** — `ANTHROPIC_API_KEY` unset | **no** |
| Mock / scripted | yes | n/a | n/a | yes (all tests, demo, eval) |

### Transcript

```
$ curl -sS -m 5 http://localhost:11434/api/tags
curl: (7) Failed to connect to localhost port 11434: Couldn't connect to server

$ python3 -c "import os; print('set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET')"
NOT SET

$ curl -sS -m 10 -o /dev/null -w "%{http_code}\n" https://api.anthropic.com/v1/messages
405                        # endpoint reachable; GET not allowed, as expected
```

---

## 3. What the cloud path *was* verified to do

This is where Phase 4 differs from Phase 3. In Phase 3 the network itself was
blocked, so nothing about the transport could be checked. Here the network path
to the cloud provider works, and only the credential is missing — which permits
a real, limited verification.

Driving the production `CloudProvider` against the **live** API with a
deliberately invalid key:

```
$ ANTHROPIC_API_KEY="sk-ant-invalid-probe" python3 -c "...CloudProvider.complete(...)"
health: (True, "cloud provider 'anthropic' configured with model 'claude-sonnet-5' ...")
ProviderUnavailable: cloud provider rejected the credential in ANTHROPIC_API_KEY (401)
```

Three things this genuinely establishes, none of which are simulated:

1. **The network path works.** The request left the process, crossed the proxy,
   and reached Anthropic's API.
2. **The request shape is well-formed enough to be authenticated.** The server
   answered `401 authentication_error`, not `400 invalid_request_error`. A
   malformed body, wrong path, or missing `anthropic-version` header would have
   produced the latter.
3. **The provider classifies the failure correctly.** A 401 becomes
   `ProviderUnavailable` and is *not* retried — verified against the live
   endpoint, not only against the stub transport in the unit tests.

What it does **not** establish: that a real completion succeeds, how long one
takes, whether the model returns schema-valid JSON, or whether its
classifications are any good.

**Correction, 2026-08-17 — inference 2 above was wrong, and the 401 concealed
it.** The request shape was *not* well-formed. The payload forwarded
`temperature: 0.0` (Forge asks for deterministic sampling everywhere), and
current Anthropic models reject non-default sampling parameters: `temperature`,
`top_p`, and `top_k` return `400 invalid_request_error` on Opus 4.7 and later,
and on the configured `claude-sonnet-5` any non-default value does the same. So
**every** cloud call would have failed — not for want of a credential, but on
the body. Authentication is checked before body validation, so a 401 tells you
nothing about the payload, and the original inference read more into it than the
probe could support.

The parameter has been removed from the Anthropic payload and the omission is
asserted in `tests/unit/test_providers.py`. Two things worth keeping from this:

- The probe was still worth running — it correctly established the network path
  and the failure classification. The error was in what was concluded about the
  *third* thing, not in the method.
- **A 401 probe cannot validate a request body.** Any future "the shape is
  fine" claim needs a call that gets far enough to be rejected *on the body*, or
  a real completion. This is the same rule the rest of this document already
  applies to model quality, applied one level lower down the stack.

The cloud path therefore remains **unmeasured**, and now for a second, better
understood reason: it has never completed a call.

---

## 4. What this means for the Phase 4 claims

| Claim | Status |
|---|---|
| Structured schemas validate model output | **Measured** (scripted + malformed-input tests) |
| Ungrounded citations are rejected | **Measured** |
| Classification maps to the right proposal | **Measured** |
| Assessments are cached and invalidated correctly | **Measured** |
| Workflow interrupts, checkpoints, and resumes | **Measured** (incl. process restart) |
| Provider unavailability is explicit | **Measured** |
| Cloud request shape is accepted by the real API | **Partially measured** (§3) |
| A real model returns schema-valid JSON in practice | **MEASURED** — 5/5, see §6 |
| Real assessment latency | **MEASURED, high variance** — typical 40-60 s, worst case >300 s, see §6 and §7 |
| Real classification accuracy | **MEASURED** — 5/5 on two independent runs, see §6 and §7 |
| False-positive conflict rate | **0 of 2 adversarial cases** with reasoning on; **1 of 2 with reasoning off**, see §8 |
| Extraction quality | **Unmeasured, and no eval exists** — `forge/evaluation/data/` covers assessment and retrieval only |

The false-positive row was the single largest open risk carried out of Phase 4.
It is now partially answered — see §6 — but two adversarial cases cannot
establish a rate, so it remains the thing most worth measuring next. §8 is why
that matters concretely: a configuration change that looked like a pure latency
win flipped one of those two cases, and a 2-case set is the only thing standing
between that and going unnoticed.

---

## 5. How to measure it

Both paths are one command each. The evaluation set and harness already exist
and are exercised in CI against the scripted provider, so nothing new needs to
be built — only run.

**Local model (the free, self-hosted path):**

```bash
ollama serve
ollama pull qwen3:8b
export FORGE_LLM_PROVIDER=ollama FORGE_MODEL_DEFAULT=qwen3:8b
python3 scripts/assessment_eval.py --provider ollama
```

**Remote Ollama (Forge on a laptop, model on a GPU box):**

```bash
export FORGE_LLM_PROVIDER=ollama FORGE_OLLAMA_URL=http://192.168.1.50:11434
python3 scripts/assessment_eval.py --provider ollama
```

**Cloud:**

```bash
export FORGE_LLM_PROVIDER=cloud ANTHROPIC_API_KEY=...
python3 scripts/assessment_eval.py --provider cloud
```

Each prints structured-output validity, grounding rate, classification
accuracy, proposal correctness, cache effectiveness, and latency per case.

**Do not compare the two runs as though they were interchangeable.** A local 8B
model and a hosted frontier model are different instruments; Forge records
provider and model identity on every assessment precisely so the two are never
silently mixed. Report them as two rows, never averaged.

---

## 6. Reproducing the environment probe

```bash
curl -sS -m 5 http://localhost:11434/api/tags               # local Ollama
curl -sS -m 10 -o /dev/null -w "%{http_code}\n" \
     https://api.anthropic.com/v1/messages                  # cloud reachability
python3 -c "import os; print(bool(os.environ.get('ANTHROPIC_API_KEY')))"
```

---

## Related

- [`local-model-capability-spike.md`](local-model-capability-spike.md) — Phase 1 local-model probe
- [`retrieval-baseline.md`](retrieval-baseline.md) — the same discipline applied to retrieval, where measurement *was* possible
- [`../architecture/phase-4-implementation.md`](../architecture/phase-4-implementation.md) — the pipeline these providers serve

---

## 6. First real-model results (2026-08-14)

**Hardware:** ASUS laptop, RTX 4050 (~6 GB VRAM), 16 GB RAM, Windows.
**Command:** `python scripts\assessment_eval.py --provider ollama`

```
provider: ollama / qwen3:8b

  ollama/qwen3:8b   valid=1.00 grounded=1.00 class=1.00 proposal=1.00 cache=1.00  63126ms/case

  [ok] supports-direct               expected=SUPPORTS              actual=SUPPORTS
  [ok] refines-adds-condition        expected=REFINES               actual=REFINES
  [ok] conflict-contrary-finding     expected=POTENTIAL_CONFLICT    actual=POTENTIAL_CONFLICT
  [ok] irrelevant-same-domain        expected=IRRELEVANT            actual=IRRELEVANT
  [ok] insufficient-partial-overlap  expected=INSUFFICIENT_EVIDENCE actual=INSUFFICIENT_EVIDENCE
```

### What this establishes

**Structured output survives a local 8B model.** 5/5 responses validated
against the strict schema on the first attempt — no repair retries. The Phase 1
capability spike specifically worried that local models would ignore JSON
schemas often enough to make strict validation impractical. On this model and
this task, it did not happen.

**Grounding held.** Every cited span id resolved to a span that was actually
shown. Zero hallucinated citations across five cases. The rejection path
therefore never fired — which is the *good* outcome, and distinguishable from
"the check is broken" because the unit tests exercise the rejection path
directly.

**Both adversarial cases classified correctly.** These are the two cases the
set was designed around:

- `irrelevant-same-domain` — same technology, different property (retrieval
  *latency* vs a claim about factual *accuracy*). A model pattern-matching on
  shared vocabulary flags this. Qwen3 did not.
- `insufficient-partial-overlap` — touches the topic without settling it. A
  model forced toward a substantive label reaches for SUPPORTS or
  POTENTIAL_CONFLICT. Qwen3 declined correctly.

**Zero false-positive conflicts.** The conservatism rules held on the two cases
built to break them.

### What this does not establish

**Five cases is five cases.** 5/5 is consistent with a model that is right 60%
of the time; the confidence interval is enormous. This is a smoke test that
passed, not a characterisation.

**A rate needs more than two negatives.** "0 false positives out of 2
adversarial cases" is not a false-positive rate, and should never be quoted as
one.

**Nothing here transfers to the cloud path**, which is still unmeasured.

### The operational finding: latency

**63 s/case mean**, and one call hit the 120 s timeout and retried:

```
[warning] ollama_retry  attempt=0  error='timed out'
```

Per-case wall time ranged from ~31 s to ~155 s (the retry). That has real
consequences:

- An 8B model on ~6 GB VRAM is at the edge of comfortable. Some of that time is
  likely layer swapping.
- The default 120 s timeout is *marginally* too low for this hardware. Raise it
  with `FORGE_LLM_TIMEOUT=300` before a long run.
- The derivation cache matters more than expected. A re-run costs 0 s and 0
  calls; at 63 s/case, re-assessing needlessly is the difference between a
  minute and an hour.
- Batching is worth revisiting. `DEFAULT_BATCH_SIZE = 6` claims per call was
  chosen to protect quality; at this latency the per-call overhead argues for
  measuring whether a larger batch degrades accuracy at all.

## 7. Second run (2026-08-19) — quality holds, one case blows the timeout

Re-run on the same machine after the CLI, cloud-provider, and config work, to
check whether any of it changed behaviour on the Ollama path. **Same command,
same model, same dataset.**

```
ollama/qwen3:8b   valid=1.00 grounded=1.00 class=1.00 proposal=1.00 cache=1.00  112078ms/case

  [ok ] supports-direct                expected=SUPPORTS               actual=SUPPORTS
  [ok ] refines-adds-condition         expected=REFINES                actual=REFINES
  [ok ] conflict-contrary-finding      expected=POTENTIAL_CONFLICT     actual=POTENTIAL_CONFLICT
  [ok ] irrelevant-same-domain         expected=IRRELEVANT             actual=IRRELEVANT
  [ok ] insufficient-partial-overlap   expected=INSUFFICIENT_EVIDENCE  actual=INSUFFICIENT_EVIDENCE
```

**5/5 again, every metric 1.00.** Two independent runs now agree, which is the
narrow thing this establishes: the engine changes between them did not alter
classification on this path. It is still ten case-runs of one model — not a
rate.

**The mean latency is misleading and should not be quoted on its own.** It reads
112,078 ms/case against the 63,126 ms of §6, which looks like a 78% regression.
It is not. Per-case wall times, derived from the log's completion timestamps:

| Case | Wall time |
|---|---:|
| `refines-adds-condition` | 41 s |
| `conflict-contrary-finding` | **345 s** |
| `irrelevant-same-domain` | 40 s |
| `insufficient-partial-overlap` | 55 s |

Three of the four measurable cases ran *faster* than the earlier 63 s mean. One
case consumed 300 s — exactly `FORGE_LLM_TIMEOUT` — timed out, retried, and
finished ~45 s later. That single outlier accounts for essentially the entire
increase in the mean.

Two things follow, and they point in opposite directions:

- **`FORGE_LLM_TIMEOUT=300` is no longer sufficient.** §6 raised it from 120 s
  because one call exceeded that. The same case has now exceeded 300 s. The
  worst case on this hardware is not bounded by anything measured so far, and
  each retry costs the full timeout before any work resumes. Raise it further
  for long runs rather than paying 300 s to learn nothing.
- **A five-case mean cannot absorb an outlier.** With n=5, one timeout moves the
  headline number by 78% while typical performance improves. Report the
  distribution, or at least the timeout count, alongside any mean from this
  set — a single number here describes neither run well.

Not established: why that case is the slow one every time, whether it is the
prompt length or the adversarial content, and whether the improvement in the
other three is real or noise. All are answerable by running the set more than
once, which has not been done.

### Reproducing

```powershell
$env:FORGE_LLM_PROVIDER="ollama"
$env:FORGE_MODEL_DEFAULT="qwen3:8b"
$env:FORGE_LLM_TIMEOUT="300"
python scripts\assessment_eval.py --provider ollama
```

---

## 8. Reasoning off (2026-08-19) — 5.1× faster, and it breaks the case that matters

`extraction-cost.md` §2 asked for exactly one experiment: run the same
assessment set with Qwen3's reasoning disabled, on the theory that the ~7×
gap between assessment latency and real extraction latency was chain-of-thought
being generated at full token cost and then discarded. Same machine, same
model, same dataset, one variable changed.

```powershell
$env:FORGE_OLLAMA_THINK="0"
python scripts\assessment_eval.py --provider ollama
```

```
ollama/qwen3:8b   valid=1.00 grounded=1.00 class=0.80 proposal=0.80 cache=1.00  21867ms/case

  [FAIL] insufficient-partial-overlap  expected=INSUFFICIENT_EVIDENCE  actual=POTENTIAL_CONFLICT
         expected proposal None but produced ProposalType.CLAIM_CONFLICT
```

Note the model id carries `+nothink`, so none of this shared a derivation-cache
entry with §6 or §7 and none of it can be averaged with them.

### The speedup is real, and smaller than the headline

Two honest numbers, because the means are not comparable to each other:

| Comparison | Think on | Think off | Ratio |
|---|---:|---:|---:|
| Reported mean/case | 112,078 ms | 21,867 ms | **5.1×** |
| Typical case (median-ish) | ~45 s | ~18 s | **2.5×** |

The 5.1× is inflated from the *think-on* side: §7's mean carries a 345 s
timeout-and-retry outlier. Per-case wall times with reasoning off were 22 / 18 /
15 / 20 s — no outlier at all, which is itself a finding: the case that blew
both the 120 s and the 300 s timeout was the adversarial one, and without a
reasoning phase it simply does not take long. **2.5× is the number to plan
with**; 5.1× is what the two report lines say and is the wrong one to quote.

### The accuracy cost lands on the worst possible case

Accuracy fell from 5/5 to 4/5, and the one that broke is
`insufficient-partial-overlap` — evidence that *partially* overlaps an existing
claim without contradicting it. With reasoning off the model called it
`POTENTIAL_CONFLICT` and emitted a `CLAIM_CONFLICT` proposal.

That is a **false-positive conflict**: precisely the failure mode §4 names as
"the single largest open risk carried out of Phase 4." Think-on classified this
same case correctly on both 2026-08-14 and 2026-08-19. One of the two
adversarial cases in the set now fails, and it fails by manufacturing a
conflict that is not there — the direction that costs human review attention and
erodes trust in every proposal the engine raises.

Structured-output validity and grounding both held at 1.00, so this is not the
model falling apart. It is a narrower and more specific loss: without a
reasoning phase it stops distinguishing "overlaps but does not contradict" from
"contradicts."

### Decision

**Rejected for assessment.** A 2.5× speedup does not buy a false-positive
conflict on a 5-case set where only two cases probe that behaviour at all.
`FORGE_OLLAMA_THINK` stays opt-in and off by default.

**Unmeasured for extraction, and currently unmeasurable.** The experiment was
motivated by extraction cost but was run on the *assessment* set, because
`forge/evaluation/data/` contains only `assessment-v1.yaml` and
`retrieval-v1.yaml` — there is no extraction-quality eval. Extraction asks a
different question (pull concepts and claims out of a span, with every quote
grounded) and reasoning may well matter less there. Nothing here licenses that
guess in either direction.

So the honest state is: reasoning-off is disqualified on the task we can
measure, and untested on the task we wanted it for. Building an
extraction-quality eval is the prerequisite for revisiting it — and until that
exists, running the vault with `+nothink` risks spending a full extraction pass
whose output nothing can grade, with a cache key that guarantees redoing it if
the answer turns out to be no.

### Reproducing

```powershell
$env:FORGE_LLM_PROVIDER="ollama"
$env:FORGE_MODEL_DEFAULT="qwen3:8b"
$env:FORGE_LLM_TIMEOUT="300"
$env:FORGE_OLLAMA_THINK="0"
python scripts\assessment_eval.py --provider ollama
```


---

## 9. `+nothink` ran an entire extraction unnoticed (2026-08-20)

§8 measured reasoning-off, rejected it, and recorded that its effect on
*extraction* was unmeasured. It was not unmeasured — it was governing extraction
the whole time, and nothing said so.

`FORGE_OLLAMA_THINK=0` was exported into a PowerShell session for the §8
experiment and stayed there. The next day's full `Technologies/Docs` run —
5.66 h, 416 calls, 2,169 proposals — ran with reasoning off. The tell only
surfaced by accident, in a cache-hit log line:

```
extraction_cache_hit  model_id=qwen3:8b+nothink  prompt_version=extract-prompts/0.3.0
```

**The caching layer behaved correctly throughout.** `identity_variant()` appends
`+nothink`, that is part of the derivation key, and think-on and think-off
results never shared an entry. The mechanism designed to keep the two modes
apart did its job. What was missing was any way to *see* which mode was active:
`forge status` reported the provider and its reachability but never the model
identity extraction would cache under.

Fixed: `forge status` now prints `model identity` including the variant, and
says plainly when reasoning is off. Three tests pin it.

### Two conclusions, and one of them is uncomfortable

**The cost numbers make more sense now.** The run measured 49 s/call over
~1,100-char spans. §8 measured reasoning-off as ~2.5× faster on typical cases,
which puts a think-on equivalent near 120 s/call — and the 228 s/call figure
from the earlier structural-span sample is roughly double that for roughly
double the span size. Three measurements that looked mutually inconsistent are
consistent once mode and span size are both accounted for.

**The quality assessment of that run is confounded.** A 25+25 sample put claims
at ~60-70% usable and concepts at ~35-40%, and that was attributed to the
extraction prompt. The prompt was genuinely underspecified — it never said what
qualifies as a concept — so the `0.3.0` rewrite stands on its own. But the
*magnitude* cannot be attributed cleanly: some share of that junk may be
reasoning-off, which §8 already showed degrades classification. **Do not quote
"~35-40% concept precision" as a property of the prompt.** It is a property of
one run under a mode nobody knew was on.

Separating the two needs a think-on run of the same scope, which is the next
measurement.
