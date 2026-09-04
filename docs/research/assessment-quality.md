# Evidence assessment, measured on a real cloud model

*Run 2026-09-03. `python3 scripts/assessment_eval.py --provider cloud`,
21 cases from `forge/evaluation/data/assessment-v1.yaml`, against
`openai/gpt-oss-120b` served by Groq. Every number here came out of that run.*

**Headline: the pipeline's safety properties hold perfectly. The model's
judgement does not, and it fails in one specific place — it cannot reliably
tell that evidence is insufficient.**

*Updated 2026-09-04. §7: a prompt fix took classification 0.76 → 0.86 and
conflict recall 2/3 → 3/3, but **the false-positive conflict rate held at
exactly 11.1%** — one false conflict was fixed and a different one created.
§8: on a held-out set the fix was not written against, **cases the cues
describe and cases they do not scored identically, 3/5 each** — the cues
conferred no measurable advantage, and the REFINES regression reproduced on a
fresh case. Validity and grounding stayed at 1.00 on both sets.*

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

## 7. Re-run on `assess-prompts/0.2.0` — the number moved, the rate did not

*Run 2026-09-04, same 21 cases, same model, same command.*

| | 0.1.0 | 0.2.0 |
|---|---:|---:|
| Classification | 0.76 | **0.86** |
| Proposal | 0.81 | **0.86** |
| Validity / grounding | 1.00 / 1.00 | 1.00 / 1.00 |
| **False-positive conflicts** | **2/18 (11.1%)** | **2/18 (11.1%)** |
| Conflict recall | 2/3 | **3/3** |

Per class:

| Class | 0.1.0 | 0.2.0 | |
|---|---:|---:|---|
| SUPPORTS | 4/4 | 4/4 | — |
| IRRELEVANT | 5/5 | 5/5 | — |
| POTENTIAL_CONFLICT | 2/3 | **3/3** | improved |
| INSUFFICIENT_EVIDENCE | 2/6 | **4/6** | improved |
| **REFINES** | **3/3** | **2/3** | **regressed** |

Three cases fixed, one broken, two unmoved:

- **fixed** — `conflict-contrary-finding`, `insufficient-partial-overlap`,
  `insufficient-anecdote-without-comparison`
- **broken** — `refines-narrows-scope`, which passed before and now returns
  POTENTIAL_CONFLICT
- **unmoved** — `insufficient-different-population`,
  `insufficient-mechanism-without-outcome`

### Four things the headline hides

**1. The false-positive conflict rate did not improve. It relocated.**
2 of 18 before, 2 of 18 after — identical. `insufficient-partial-overlap`
stopped being a false conflict and `refines-narrows-scope` started being one.
The metric that Phase 5 actually gates on is unchanged, and a reader looking
only at 0.76 → 0.86 would conclude otherwise.

**2. A case that passed now fails.** Prompt edits are not local. The cues
added for INSUFFICIENT_EVIDENCE include *"reports on a different population,
system, version, or setting"*, and narrowing a claim's scope is structurally
that — so a REFINES case now reads as a mismatch and escalates to conflict.
Strengthening one class degraded its neighbour, which is precisely what a
21-case set exists to catch and what a 5-case set could not have.

**3. The two most explicitly instructed cases did not move.** This is the
uncomfortable one. `insufficient-mechanism-without-outcome` has both a cue
(*"describes a mechanism, plan, or process without reporting the outcome the
claim is about"*) and a dedicated rule (*"If the evidence does not report the
outcome, measurement, or comparison the claim asserts, choose
INSUFFICIENT_EVIDENCE — not SUPPORTS"*). It still returns SUPPORTS.
`insufficient-different-population` is named almost verbatim in its cue and
still returns POTENTIAL_CONFLICT.

The three cases that *were* fixed had no such targeted instruction; they
improved from the general framing. **Writing a more specific instruction did
not produce a more reliable outcome — if anything the reverse.** That is
evidence against the reflex of adding another line to the prompt when a case
fails.

**4. The dangerous failure survives.** Evidence describing a mechanism is
still read as support for an outcome it never reported, still producing a
`CLAIM_EVIDENCE` proposal. The single error most likely to write a wrong
belief into the knowledge base is the one the fix did not touch.

### How much of the gain is real

The caveat from §6 stands and now has a size. Of the three fixed cases, all
three were among those the cues were written against — so the honest reading
is that the prompt absorbed part of a known failure set, not that the model
improved. Against that, two targeted cases resisted the fix and a fourth
broke, which suggests the absorption is shallower than +0.10 implies.

**The defensible claim is: conflict recall went 2/3 → 3/3, the false-positive
rate held at 11.1%, and INSUFFICIENT_EVIDENCE remains the weakest class at
4/6.** Everything else needs held-out cases.

## 8. The held-out set

`assessment-holdout-v1.yaml`, 18 cases, written 2026-09-04. Separate file, not
appended — appending would destroy both instruments at once, making the fitted
score uninterpretable and the held-out score fitted.

**The limitation this cannot remove:** its author had already read the
failures. That is a weaker guarantee than a set written blind, and no design
undoes it. What the design does is bound the leakage and make it visible:
cases derive from a taxonomy of *why* evidence fails to settle a claim rather
than from the observed failures; every case uses a technical domain absent
from the fitted set, which is entirely RAG and ML, so surface similarity
cannot carry a pattern match; and the strata separate what can be trusted from
what cannot.

| Stratum | n | Class | What it measures |
|---|---:|---|---|
| near-transfer | 5 | INSUFFICIENT_EVIDENCE | One per 0.2.0 cue, same shape, different content. Did the cue generalise, or was it memorised? |
| far-transfer | 5 | INSUFFICIENT_EVIDENCE | Five reasons **no cue names**. Is the class understood, or only the listed patterns? |
| regression-probe | 4 | REFINES ×2, SUPPORTS ×2 | Cases a conflict-happy prompt would over-escalate. Is the cure worse than the disease? |
| conflict | 2 | POTENTIAL_CONFLICT | Recall must not be traded away. |
| irrelevant | 2 | IRRELEVANT | The IRRELEVANT/INSUFFICIENT boundary, which 0.2.0 also touched. |

**Read a strong `near` score with suspicion and a strong `far` score as the
real signal.** Near-transfer shares its epistemic shape with a written cue, so
it can be passed by a prompt that generalised only slightly. Far-transfer
cannot: correlation offered for a causal claim, an aggregate that hides the
subgroup the claim names, a secondhand report of the claim rather than an
observation, a term the evidence defines differently, and a direction
confirmed where a magnitude was asserted — none appear in `assess-prompts`.

16 of the 18 have a correct answer other than POTENTIAL_CONFLICT, so the
false-positive conflict rate stays the headline number here as well.

`tests/unit/test_holdout_set.py` guards the design structurally: no id or
claim shared with the fitted set, the strata balance, and — the load-bearing
one — that no far-transfer case has become described by a prompt cue. If a
future cue names one of those five categories, that test fails and says the
case must move to near-transfer with a replacement written.

### Result, 2026-09-04: near and far transfer scored identically

`class=0.72`, `proposal=0.78`, validity and grounding **1.00**, 13/18.

| Stratum | Score | |
|---|---:|---|
| **near-transfer** | **3/5 (60%)** | cases a cue was written for |
| **far-transfer** | **3/5 (60%)** | cases no cue names |
| regression-probe | 3/4 (75%) | |
| conflict | 2/2 (100%) | |
| irrelevant | 2/2 (100%) | |

**The prediction going in was that near would score well and far would be the
real test. Near and far came out the same, to the case.**

That equality is the finding. If the cues worked by giving the model five
patterns to match, near-transfer — which shares its epistemic shape with a
written cue — should have beaten far-transfer, which shares nothing. It did
not, by any margin. **The cues conferred no measurable advantage on the cases
they were written to describe.**

Read alongside the fitted set that is fairly damning of the fix. On
`assessment-v1` after 0.2.0, INSUFFICIENT_EVIDENCE was 4/6 (67%). Here it is
6/10 (60%) — the same number within the noise of these sample sizes. **The
most economical explanation is that ~60% is this model's baseline on the class
and the prompt moved it very little**, the +0.10 on the fitted set being two
cases' worth of absorption plus a conflict flip rather than a capability
change.

### The regression is confirmed, not noise

`ht-probe-refines-tightens-bound` returned POTENTIAL_CONFLICT. That case was
written as a deliberate mirror of `refines-narrows-scope` — the case 0.2.0
broke on the fitted set — in an unrelated domain. **It reproduced.** Two
independent cases, same shape, same wrong answer: 0.2.0 escalates a stated
boundary condition to disagreement. The §7 note that one regression might be
noise is settled; it is a behaviour.

### The dangerous failure is the characteristic one

Three of the five failures asserted a relationship the text does not
establish, each producing a `CLAIM_EVIDENCE` proposal:

| Case | Actual | What the passage actually said |
|---|---|---|
| `ht-near-single-incident` | SUPPORTS | One host, one evening, no baseline |
| `ht-far-correlation-for-causal-claim` | SUPPORTS | An association — **and the passage states adoption was voluntary and self-selected** |
| `ht-far-term-defined-differently` | SUPPORTS | **The passage defines the key term differently from the claim** |

On the fitted set this shape was one failure in five. Here it is three in
five. It is not an edge case; it is what this model does when evidence is
on-topic and inconclusive. The last two are the striking ones — in both, the
passage explicitly contains the sentence that should have blocked the
inference, and the model asserted support anyway.

### What did hold

Validity and grounding stayed at **1.00 on a set the pipeline had never
seen** — no malformed output, no invented citation, across 18 fresh cases in
domains absent from the fitted set. The safety properties generalise; the
judgement does not. That contrast is the clearest single statement this
document can make about where the engineering is sound and where it is not.

False-positive conflicts were **1/16 (6.2%)** and conflict recall 2/2. Both
are small-n and should not be read as improvements over §7 — different cases,
not a fairer test of the same ones.


## 9. What follows

1. **Stop tuning the prompt for this class.** Near and far scored the same,
   which is what it looks like when instruction is not the binding
   constraint. A third pass of cue-writing has no evidence behind it.
2. **Try the structural fix instead.** A second, single-question pass over any
   assessment returning SUPPORTS or REFINES — *"does this passage report the
   outcome, measurement, or comparison the claim asserts?"* — demoting to
   INSUFFICIENT_EVIDENCE on a no. It targets the three-in-five failure
   directly, costs one extra call on the assertive path only, and is
   measurable on both sets. This is the next thing worth building.
3. **Consider reverting the boundary-condition language in 0.2.0.** It is the
   confirmed cause of two REFINES failures and bought little measurable
   elsewhere.
4. **Repeat runs for variance.** Still one observation per prompt version per
   set.
5. **Keep contradiction detection human-routed.** Nothing measured has
   changed that.
