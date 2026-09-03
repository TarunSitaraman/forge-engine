# Evidence assessment, measured on a real cloud model

*Run 2026-09-03. `python3 scripts/assessment_eval.py --provider cloud`,
21 cases from `forge/evaluation/data/assessment-v1.yaml`, against
`openai/gpt-oss-120b` served by Groq. Every number here came out of that run.*

**Headline: the pipeline's safety properties hold perfectly. The model's
judgement does not, and it fails in one specific place — it cannot reliably
tell that evidence is insufficient.**

---

## 1. What was measured

| Metric | Result |
|---|---|
| Structured-output validity | **1.00** |
| Grounding (citations resolve to real stored spans) | **1.00** |
| Classification accuracy | **0.76** (16/21) |
| Proposal-type accuracy | 0.81 |
| Cache effectiveness | 1.00 |
| Latency | 8,038 ms/case |

Validity and grounding at 1.00 are the load-bearing results. Every response
parsed against the schema, and **not one citation was invented** — each
resolved to a span id actually in the store. The guard that rejects ungrounded
output never had to fire, which is the outcome it was built for.

Latency is inflated by Groq free-tier rate limiting: the run hit HTTP 429
repeatedly and the client's backoff absorbed it, sleeping 1–12 s between
attempts. Retries succeeded every time. The figure is throughput under a free
tier, not model speed.

## 2. The headline number is misleading

0.76 is an average over five classes that behave nothing alike:

| Expected class | Correct | Accuracy |
|---|---|---|
| SUPPORTS | 4/4 | **100%** |
| REFINES | 3/3 | **100%** |
| IRRELEVANT | 5/5 | **100%** |
| POTENTIAL_CONFLICT | 2/3 | 67% |
| **INSUFFICIENT_EVIDENCE** | **2/6** | **33%** |

**Four of the five failures are INSUFFICIENT_EVIDENCE cases.** On everything
else the model is at 100% or one case away from it. There is no general
accuracy problem here; there is one specific blind spot.

The pattern is that the model *wants to classify*. Given evidence that does
not settle the question, it reaches for whichever label is nearest rather than
declining. Each of the four failures picks a different neighbour:

| Case | Expected | Actual |
|---|---|---|
| `insufficient-partial-overlap` | INSUFFICIENT_EVIDENCE | POTENTIAL_CONFLICT |
| `insufficient-different-population` | INSUFFICIENT_EVIDENCE | POTENTIAL_CONFLICT |
| `insufficient-anecdote-without-comparison` | INSUFFICIENT_EVIDENCE | IRRELEVANT |
| `insufficient-mechanism-without-outcome` | INSUFFICIENT_EVIDENCE | SUPPORTS |

**The last one is the dangerous failure.** Evidence describing a mechanism
without establishing an outcome was read as support for the outcome, and the
pipeline duly produced a `CLAIM_EVIDENCE` proposal. That is the shape of
error that adds a wrong belief to a knowledge base rather than merely failing
to add a right one — and it is exactly what a provenance floor cannot catch,
because the citation is real and the quote is genuinely in the span. The
reasoning is what is wrong.

## 3. The false-positive conflict rate, at last

Phase 5's gate asks for this, and the roadmap was careful that the earlier
result was not one: *"zero false positives on two adversarial cases is
encouraging and is not a rate."*

Now it is a rate:

- **False-positive conflicts: 2 of 18 non-conflict cases — 11.1%.**
  Both are INSUFFICIENT_EVIDENCE cases misread as POTENTIAL_CONFLICT.
- **Conflict recall: 2 of 3 — 67%.** `conflict-contrary-finding` was missed,
  classified REFINES: a contrary finding was absorbed as a refinement rather
  than flagged as a disagreement.

Both errors matter, and they are not symmetric in cost. A false positive costs
a human a review of something that was never a conflict. A false negative —
the missed contrary finding — means a claim was quietly *refined* by evidence
that actually contradicts it. Under Phase 4's design a conflict routes to a
human and a refinement supersedes non-destructively, so the miss is the one
that changes stored knowledge without anyone looking.

**At 11.1% false positives and 67% recall, promoting `POTENTIAL_CONFLICT` to
an asserted `Contradiction` entity remains unjustified.** The roadmap's
decision to keep contradiction detection human-routed is now supported by a
measurement rather than by caution.

## 4. Why the 5-case smoke test could not have found this

The 2026-08-14 local run scored 5/5 with 0 false-positive conflicts. Nothing
about it was wrong; it was measuring a set too small to contain the failure
mode. It had one INSUFFICIENT_EVIDENCE case. This set has six, and four of
them fail.

The expansion to 21 cases is what turned "the model looks fine" into "the
model has a specific, reproducible blind spot in the one class where being
wrong is most expensive."

## 5. What this does not establish

- **One run, one model.** No variance estimate; a second run could differ,
  particularly on the borderline INSUFFICIENT_EVIDENCE cases.
- **21 cases is still small.** 33% on a class of six is 2 correct. The
  confidence interval on that is wide, and the honest reading is "clearly
  weak", not "exactly 33%".
- **No local-model row, and there will not be one soon.** The 2026-08-14
  Qwen3 8B result was produced on an RTX 4050 machine that is no longer in
  use; current work runs against Groq. The two runs use different sets (5 vs
  21 cases) as well as different models, so they are not comparable and are
  not presented as a pair. Should a local row ever be wanted, the standing
  rule holds: separate rows, never averaged — they are different
  instruments.

## 6. The prompt had three defects, and they map onto the four failures

Reading `ASSESSMENT_INSTRUCTION` after the run, the failures are not mysterious:

1. **INSUFFICIENT_EVIDENCE had the thinnest definition of any class** — one
   line, *"the new evidence touches the topic but does not say enough to
   judge"*, with no recognisable cues. Every other class got richer guidance.
   A class the model cannot recognise is a class it will not reach for.
2. **The tie-breaks were asymmetric and pointed the wrong way.** There was a
   SUPPORTS/REFINES rule biasing toward SUPPORTS, and *no* SUPPORTS /
   INSUFFICIENT_EVIDENCE rule at all — which is precisely the
   `mechanism-without-outcome → SUPPORTS` failure.
3. **Nothing distinguished IRRELEVANT from INSUFFICIENT_EVIDENCE.** Both
   produce no proposal, so no pressure existed to tell them apart — which is
   how an on-topic anecdote became IRRELEVANT.

Note the one rule that *did* exist — *"if you are unsure whether something
conflicts, choose INSUFFICIENT_EVIDENCE rather than POTENTIAL_CONFLICT"* — was
ignored twice. Restating a tie-break is not enough; the class needed positive
cues, not just a preference.

`assess-prompts/0.2.0` gives INSUFFICIENT_EVIDENCE five concrete cues (outcome
not reported, different population, partial overlap, single observation
without comparison, intended rather than measured), names which direction
"be conservative" runs in (asserting a relationship the text does not
establish is worse than declining, because a wrong SUPPORTS changes stored
knowledge), and states the IRRELEVANT boundary explicitly.

### The next number will be optimistic, and that has to be said

**These cues were written after seeing which cases failed.** That is fitting to
the evaluation set, and it means a post-fix score on these same 21 cases
measures "did the prompt absorb these five failure shapes", not "is the model
better at recognising insufficient evidence". The cues were deliberately
written as general epistemic categories rather than case-specific patches, but
that mitigates the problem; it does not remove it.

An honest confirmation needs held-out cases — new INSUFFICIENT_EVIDENCE cases
written without reference to the failures above. Until then the post-fix
number is a check that the change did something, not evidence of quality.

## 7. What follows

1. **Re-run the 21 cases on `assess-prompts/0.2.0`.** Read it as a sanity
   check, with the caveat above.
2. **Write held-out INSUFFICIENT_EVIDENCE cases** — the only way to know
   whether the fix generalises.
3. **Repeat runs for variance.** Every number here is one observation.
4. **Keep contradiction detection human-routed.** Measured, not assumed.
