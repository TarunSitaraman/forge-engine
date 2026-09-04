# Evidence assessment, measured on a real cloud model

*Run 2026-09-03. `python3 scripts/assessment_eval.py --provider cloud`,
21 cases from `forge/evaluation/data/assessment-v1.yaml`, against
`openai/gpt-oss-120b` served by Groq. Every number here came out of that run.*

**Headline: the pipeline's safety properties hold perfectly. The model's
judgement does not, and it fails in one specific place — it cannot reliably
tell that evidence is insufficient.**

*Updated 2026-09-04 (§7): a prompt fix took classification 0.76 → 0.86 and
conflict recall 2/3 → 3/3, but **the false-positive conflict rate held at
exactly 11.1%** — one false conflict was fixed and a different one created.
A previously-passing REFINES case regressed, and the two cases given the most
explicit instructions did not move at all.*

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

## 8. What follows

1. **Write held-out cases.** Now the priority rather than one of several:
   three of the fixed cases were written against, two targeted ones resisted,
   and nothing here distinguishes absorption from improvement. Held-out
   INSUFFICIENT_EVIDENCE and REFINES cases would settle it.
2. **Investigate `mechanism-without-outcome` specifically.** It has the most
   explicit instruction in the prompt and ignores it. If instruction cannot
   fix it, the options are a structural one — a second pass that asks only
   "does this text report the outcome the claim asserts?" — or accepting it
   and relying on human review of `CLAIM_EVIDENCE` proposals.
3. **Watch REFINES for further erosion.** One regression may be noise across a
   single run; two would be a pattern.
4. **Repeat runs for variance.** Still one observation per prompt version, and
   the whole 0.76 → 0.86 delta is 2 cases.
5. **Keep contradiction detection human-routed.** The false-positive rate did
   not move.
