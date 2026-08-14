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
| Real assessment latency | **MEASURED** — 63 s/case local, see §6 |
| Real classification accuracy | **MEASURED** — 5/5 on 5 cases, see §6 |
| False-positive conflict rate | **0 of 2 adversarial cases** — encouraging, not yet a rate |

The last row was the single largest open risk carried out of Phase 4. It is now
partially answered — see §6 — but two adversarial cases cannot establish a
rate, so it remains the thing most worth measuring next.

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

### Reproducing

```powershell
$env:FORGE_LLM_PROVIDER="ollama"
$env:FORGE_MODEL_DEFAULT="qwen3:8b"
$env:FORGE_LLM_TIMEOUT="300"
python scripts\assessment_eval.py --provider ollama
```
