# Provider Availability — What Was and Was Not Measured

*Phase 4. A record of which inference providers could actually be exercised in
this environment, what that permits us to claim, and what it does not.*

**Headline: no real model produced a single assessment during Phase 4
development. The pipeline is fully measured; model quality is entirely
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
| A real model returns schema-valid JSON in practice | **UNMEASURED** |
| Real assessment latency | **UNMEASURED** |
| Real classification accuracy | **UNMEASURED** |
| False-positive conflict rate | **UNMEASURED** — and this is the important one |

The last row deserves emphasis. Forge's conservatism rules — no `CONTRADICTS`,
prefer `INSUFFICIENT_EVIDENCE` when unsure, route conflicts to a human — exist
to keep false conflicts rare, because a false conflict costs more trust than a
missed one. **Whether those rules actually achieve that with a real model is
untested.** It is the single largest open risk carried into Phase 5.

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
