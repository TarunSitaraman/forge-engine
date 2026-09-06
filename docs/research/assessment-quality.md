# Evidence assessment, measured on a real cloud model

*Run 2026-09-03. `python3 scripts/assessment_eval.py --provider cloud`,
21 cases from `forge/evaluation/data/assessment-v1.yaml`, against
`openai/gpt-oss-120b` served by Groq. Every number here came out of that run.*

**Headline: the pipeline's safety properties hold perfectly. The model's
judgement does not, and it fails in one specific place. It cannot reliably
tell that evidence is insufficient.**

*Updated 2026-09-05, and the update is a retraction. §10: re-running the
fitted set against the same model, the same prompt and the same command scored
**15/21 where §7 had measured 18/21**, and the three cases that differed were
exactly the three §7 credited a prompt revision with fixing. **The 0.76 → 0.86
gain is withdrawn, along with every per-class movement resting on it.** A
single run on this set carries a spread of at least three cases, which is wider
than most of the deltas this document was arguing over.*

*Updated again 2026-09-06, and this is the finding the document was looking
for. A second model family, `qwen/qwen3.8-27b`, scored **16/21 twice with zero
cases answered inconsistently**, and **every one of its five failures is a case
gpt-oss-120b also fails, with the same wrong label**. Two independently trained
families, one a fifth the size, converging on the same answer for every hard
case is evidence about the cases, not the models: re-reading them, one is
mislabelled outright and three are genuinely contestable. Three rounds of prompt
work failed because they were trying to instruct a model into labels the cases
do not clearly support. See §11, including why relabelling to match model
consensus would be exactly the motivated reasoning this project has caught
itself in before.*

*Repeated 2026-09-05 with `--repeat 3`: four complete runs now stand at 18, 15,
16 and 16 of 21. **Three cases flip between runs and three are wrong in every
run**, and the three that flip are exactly the three the prompt revision was
credited with fixing. Every published movement in this document lives inside
those three.*

*What survives the retraction: validity and grounding at **1.00 on every run of
both sets**; INSUFFICIENT_EVIDENCE as the consistently weakest class; and one
case, `insufficient-mechanism-without-outcome`, that has returned SUPPORTS in
every run recorded here despite a cue, a dedicated rule, and a structural
check written against it. §8's finding that cues conferred no advantage on the
cases they describe (3/5 each, near and far) also survives. It is an argument
from no difference, which noise cannot manufacture.*

*Earlier updates, now read in that light. §7: a prompt fix appeared to take
classification 0.76 → 0.86 while the false-positive conflict rate held at
exactly 11.1%. §8: a held-out set the fix was not written against. §9: a
structural second-pass check scored 13/18 with and 13/18 without. **Three
approaches failed to move this class, and the first one now looks as though it
never worked at all.***

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
parsed against the schema, and **not one citation was invented**: each
resolved to a span id actually in the store. The guard that rejects ungrounded
output never had to fire, which is the outcome it was built for.

Latency is inflated by Groq free-tier rate limiting: the run hit HTTP 429
repeatedly and the client's backoff absorbed it, sleeping 1-12 s between
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
to add a right one, and it is exactly what a provenance floor cannot catch,
because the citation is real and the quote is genuinely in the span. The
reasoning is what is wrong.

## 3. The false-positive conflict rate, at last

Phase 5's gate asks for this, and the roadmap was careful that the earlier
result was not one: *"zero false positives on two adversarial cases is
encouraging and is not a rate."*

Now it is a rate:

- **False-positive conflicts: 2 of 18 non-conflict cases, 11.1%.**
  Both are INSUFFICIENT_EVIDENCE cases misread as POTENTIAL_CONFLICT.
- **Conflict recall: 2 of 3, 67%.** `conflict-contrary-finding` was missed,
  classified REFINES: a contrary finding was absorbed as a refinement rather
  than flagged as a disagreement.

Both errors matter, and they are not symmetric in cost. A false positive costs
a human a review of something that was never a conflict. A false negative,
the missed contrary finding, means a claim was quietly *refined* by evidence
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
  rule holds: separate rows, never averaged. They are different
  instruments.

## 6. The prompt had three defects, and they map onto the four failures

Reading `ASSESSMENT_INSTRUCTION` after the run, the failures are not mysterious:

1. **INSUFFICIENT_EVIDENCE had the thinnest definition of any class.** One
   line, *"the new evidence touches the topic but does not say enough to
   judge"*, with no recognisable cues. Every other class got richer guidance.
   A class the model cannot recognise is a class it will not reach for.
2. **The tie-breaks were asymmetric and pointed the wrong way.** There was a
   SUPPORTS/REFINES rule biasing toward SUPPORTS, and *no* SUPPORTS /
   INSUFFICIENT_EVIDENCE rule at all: which is precisely the
   `mechanism-without-outcome → SUPPORTS` failure.
3. **Nothing distinguished IRRELEVANT from INSUFFICIENT_EVIDENCE.** Both
   produce no proposal, so no pressure existed to tell them apart: which is
   how an on-topic anecdote became IRRELEVANT.

Note the one rule that *did* exist, *"if you are unsure whether something
conflicts, choose INSUFFICIENT_EVIDENCE rather than POTENTIAL_CONFLICT"*: was
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

An honest confirmation needs held-out cases, new INSUFFICIENT_EVIDENCE cases
written without reference to the failures above. Until then the post-fix
number is a check that the change did something, not evidence of quality.

## 7. Re-run on `assess-prompts/0.2.0`: the number moved, the rate did not

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
| SUPPORTS | 4/4 | 4/4 | - |
| IRRELEVANT | 5/5 | 5/5 | - |
| POTENTIAL_CONFLICT | 2/3 | **3/3** | improved |
| INSUFFICIENT_EVIDENCE | 2/6 | **4/6** | improved |
| **REFINES** | **3/3** | **2/3** | **regressed** |

Three cases fixed, one broken, two unmoved:

- **fixed**, `conflict-contrary-finding`, `insufficient-partial-overlap`,
  `insufficient-anecdote-without-comparison`
- **broken**: `refines-narrows-scope`, which passed before and now returns
  POTENTIAL_CONFLICT
- **unmoved**, `insufficient-different-population`,
  `insufficient-mechanism-without-outcome`

### Four things the headline hides

**1. The false-positive conflict rate did not improve. It relocated.**
2 of 18 before, 2 of 18 after: identical. `insufficient-partial-overlap`
stopped being a false conflict and `refines-narrows-scope` started being one.
The metric that Phase 5 actually gates on is unchanged, and a reader looking
only at 0.76 → 0.86 would conclude otherwise.

**2. A case that passed now fails.** Prompt edits are not local. The cues
added for INSUFFICIENT_EVIDENCE include *"reports on a different population,
system, version, or setting"*, and narrowing a claim's scope is structurally
that, so a REFINES case now reads as a mismatch and escalates to conflict.
Strengthening one class degraded its neighbour, which is precisely what a
21-case set exists to catch and what a 5-case set could not have.

**3. The two most explicitly instructed cases did not move.** This is the
uncomfortable one. `insufficient-mechanism-without-outcome` has both a cue
(*"describes a mechanism, plan, or process without reporting the outcome the
claim is about"*) and a dedicated rule (*"If the evidence does not report the
outcome, measurement, or comparison the claim asserts, choose
INSUFFICIENT_EVIDENCE: not SUPPORTS"*). It still returns SUPPORTS.
`insufficient-different-population` is named almost verbatim in its cue and
still returns POTENTIAL_CONFLICT.

The three cases that *were* fixed had no such targeted instruction; they
improved from the general framing. **Writing a more specific instruction did
not produce a more reliable outcome, if anything the reverse.** That is
evidence against the reflex of adding another line to the prompt when a case
fails.

**4. The dangerous failure survives.** Evidence describing a mechanism is
still read as support for an outcome it never reported, still producing a
`CLAIM_EVIDENCE` proposal. The single error most likely to write a wrong
belief into the knowledge base is the one the fix did not touch.

### How much of the gain is real

The caveat from §6 stands and now has a size. Of the three fixed cases, all
three were among those the cues were written against, so the honest reading
is that the prompt absorbed part of a known failure set, not that the model
improved. Against that, two targeted cases resisted the fix and a fourth
broke, which suggests the absorption is shallower than +0.10 implies.

**The defensible claim is: conflict recall went 2/3 → 3/3, the false-positive
rate held at 11.1%, and INSUFFICIENT_EVIDENCE remains the weakest class at
4/6.** Everything else needs held-out cases.

## 8. The held-out set

`assessment-holdout-v1.yaml`, 18 cases, written 2026-09-04. Separate file, not
appended: appending would destroy both instruments at once, making the fitted
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
confirmed where a magnitude was asserted: none appear in `assess-prompts`.

16 of the 18 have a correct answer other than POTENTIAL_CONFLICT, so the
false-positive conflict rate stays the headline number here as well.

`tests/unit/test_holdout_set.py` guards the design structurally: no id or
claim shared with the fitted set, the strata balance, and: the load-bearing
one. That no far-transfer case has become described by a prompt cue. If a
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
patterns to match, near-transfer: which shares its epistemic shape with a
written cue: should have beaten far-transfer, which shares nothing. It did
not, by any margin. **The cues conferred no measurable advantage on the cases
they were written to describe.**

Read alongside the fitted set that is fairly damning of the fix. On
`assessment-v1` after 0.2.0, INSUFFICIENT_EVIDENCE was 4/6 (67%). Here it is
6/10 (60%), the same number within the noise of these sample sizes. **The
most economical explanation is that ~60% is this model's baseline on the class
and the prompt moved it very little**, the +0.10 on the fitted set being two
cases' worth of absorption plus a conflict flip rather than a capability
change.

### The regression is confirmed, not noise

`ht-probe-refines-tightens-bound` returned POTENTIAL_CONFLICT. That case was
written as a deliberate mirror of `refines-narrows-scope`, the case 0.2.0
broke on the fitted set, in an unrelated domain. **It reproduced.** Two
independent cases, same shape, same wrong answer: 0.2.0 escalates a stated
boundary condition to disagreement. The §7 note that one regression might be
noise is settled; it is a behaviour.

### The dangerous failure is the characteristic one

Three of the five failures asserted a relationship the text does not
establish, each producing a `CLAIM_EVIDENCE` proposal:

| Case | Actual | What the passage actually said |
|---|---|---|
| `ht-near-single-incident` | SUPPORTS | One host, one evening, no baseline |
| `ht-far-correlation-for-causal-claim` | SUPPORTS | An association, **and the passage states adoption was voluntary and self-selected** |
| `ht-far-term-defined-differently` | SUPPORTS | **The passage defines the key term differently from the claim** |

On the fitted set this shape was one failure in five. Here it is three in
five. It is not an edge case; it is what this model does when evidence is
on-topic and inconclusive. The last two are the striking ones, in both, the
passage explicitly contains the sentence that should have blocked the
inference, and the model asserted support anyway.

### What did hold

Validity and grounding stayed at **1.00 on a set the pipeline had never
seen**, no malformed output, no invented citation, across 18 fresh cases in
domains absent from the fitted set. The safety properties generalise; the
judgement does not. That contrast is the clearest single statement this
document can make about where the engineering is sound and where it is not.

False-positive conflicts were **1/16 (6.2%)** and conflict recall 2/2. Both
are small-n and should not be read as improvements over §7, different cases,
not a fairer test of the same ones.


## 9. The structural check also failed, 2026-09-04

Built `assess-prompts` out of the loop entirely: a second pass over any
SUPPORTS or REFINES asking one question, *does this passage state the
outcome, measurement, or comparison the claim asserts?*: demoting on a no,
with the yes required to quote a sentence found in the evidence.

Same held-out set, same model, with and against itself:

| | without | with |
|---|---:|---:|
| Classification | 13/18 (0.722) | **13/18 (0.722)** |
| Latency | 10,385 ms/case | 11,939 ms/case |

**Identical.** It fixed `ht-far-term-defined-differently` and broke
`ht-probe-refines-adds-precondition`, a correct REFINES it demoted.

Its own behaviour, which is the more useful number:

- 6 assertions examined, 2 demoted
- **1 demotion correct, 1 wrong: 50% precision**
- **1 of the 3 target failures caught: 33% recall**
- `ht-near-single-incident` and `ht-far-correlation-for-causal-claim` were
  examined and **upheld**. Asked the narrow question directly, about a single
  incident with no baseline and about an explicitly self-selected
  correlation, the model still answered that the passage reports the outcome.

### Three approaches, none of which moved the class

| Approach | Result |
|---|---|
| Cues describing the failure (`0.2.0`) | No advantage on cases they describe over cases they do not |
| An explicit rule naming the case | The two most explicitly instructed cases never moved |
| A narrow structural question | 33% recall, 50% precision, net zero |

That is now enough evidence to stop: **the constraint is the model's
judgement, not how it is asked.** `openai/gpt-oss-120b` cannot reliably tell
that on-topic evidence fails to establish a claim, and no phrasing tested
changes it.

### Why it is kept, and defaulted off

Off by default. A coin flip on whether a demotion is right does not earn a
call per assertion.

But the trade it makes is the cheaper one, and that is a real argument for
turning it on deliberately. Dangerous failures, an assertion built from
evidence that does not establish the claim, went **3 → 2**. It removed a
`CLAIM_EVIDENCE` proposal derived from a passage that redefined the claim's
key term, and paid for it by declining a legitimate REFINES. Wrong assertions
write false beliefs into the graph; wrong declines only fail to write true
ones.

That asymmetry is not free either. A demoted REFINES produces no proposal at
all, so a legitimate update is dropped silently rather than routed to anyone.
At 50% precision the check buys a small reduction in the worse error with a
matching increase in the quieter one, and a reader should be able to see both
halves of that trade rather than a single accuracy figure that shows neither.

`--corroborate` turns it on; `corroborate=True` in the constructor does the
same.

## 10. The 0.2.0 gain did not reproduce, 2026-09-05

*Same 21 cases, same model, same command, corroboration off, on
`assess-prompts/0.3.0`: whose diff against 0.2.0 is the version string and an
appended block of corroboration constants. `SYSTEM` and the assessment
instruction are byte-identical, so this is a re-run of §7, not a new condition.*

**15/21 (0.71). §7 measured 18/21 (0.86) the day before.**

| Run | Score |
|---|---:|
| 0.1.0, 2026-09-03 | 16/21 (0.76) |
| 0.2.0, 2026-09-04 | **18/21 (0.86)** |
| 0.2.0 re-run, 2026-09-05 | **15/21 (0.71)** |

Which cases failed is the part that matters:

| Case | §7, 0.2.0 | Re-run |
|---|---|---|
| `conflict-contrary-finding` | fixed by 0.2.0 | **failed** (REFINES) |
| `insufficient-partial-overlap` | fixed by 0.2.0 | **failed** (REFINES) |
| `insufficient-anecdote-without-comparison` | fixed by 0.2.0 | **failed** (IRRELEVANT) |
| `refines-narrows-scope` | broken by 0.2.0 | failed (POTENTIAL_CONFLICT) |
| `insufficient-different-population` | unmoved | failed (POTENTIAL_CONFLICT) |
| `insufficient-mechanism-without-outcome` | unmoved | failed (SUPPORTS) |

**The three cases credited to the prompt fix are exactly the three that came
back.** The three §7 called resistant or broken failed identically both times.
That is not a prompt that half-worked; it is a prompt that changed nothing,
measured twice through three cases of sampling noise.

§7 already hedged that the gain was absorption rather than improvement. The
hedge was too generous. **There was no gain.** The defensible reading now is
that `assess-prompts/0.2.0` is indistinguishable from `0.1.0` on this set, and
that the +0.10 was a single sample of a statistic with a spread of at least
three cases on 21.

### What this invalidates

Everything in this document argued from a one- or two-case delta, which is most
of it:

- §7's `0.76 → 0.86`, **withdrawn**. Within noise.
- §7's per-class movements (POTENTIAL_CONFLICT 2/3 → 3/3, INSUFFICIENT_EVIDENCE
  2/6 → 4/6): **withdrawn**. Both rest on the three unstable cases.
- §9's "the structural check scored net zero, fixing one case and breaking
  another": the *net zero* stands as an observation, but "fixed
  `ht-far-term-defined-differently`" and "broke
  `ht-probe-refines-adds-precondition`" are single-run flips and cannot be
  attributed to the check.
- §8's near-transfer 3/5 vs far-transfer 3/5 equality, **survives**, because
  it is an argument from *no difference*, which noise cannot manufacture. A
  spread this wide makes a real advantage harder to see, not easier, and none
  was there.

### What survives

The properties measured as absolutes rather than deltas:

- **Validity and grounding at 1.00**, on both sets, every run. No malformed
  output, no invented span id, ever.
- **INSUFFICIENT_EVIDENCE is the weakest class**, at 2/6 and 4/6 on the fitted
  set and 6/10 held out. Its rank is stable even though its value is not.
- **The characteristic failure is asserting a relationship the passage does not
  report.** It appeared in every run, and `insufficient-mechanism-without-outcome`
  has returned SUPPORTS on every single run recorded here: cued, instructed,
  and structurally checked.

That last one is the honest version of this document's thesis. Not "the model
scores 0.86", which it does not reliably, but: *the safety properties hold
absolutely, one class fails consistently, and the deltas in between were mostly
noise.*

### Method change

`--repeat N` runs the whole set N times, each with its own store so no
repetition is served the previous one's cache, and reports the score spread
plus every case answered inconsistently. **No further single-run number belongs
in this document.** A model comparison needs it most: a Kimi-versus-gpt-oss
delta of two cases would have been written up here as a result and would have
meant nothing.

```
python3 scripts/assessment_eval.py --provider cloud --model MODEL --repeat 3 --json > MODEL.json
```

### Four runs now, and the spread is three cases

*Repeated 2026-09-05 with `--repeat 3`. Run 3 lost DNS at case 13, so nine of
its cases never reached the model and it is excluded. Two complete runs from
that batch, plus the two single runs above.*

| Run | Score |
|---|---:|
| 2026-09-04 | 18/21 |
| 2026-09-05, 06:50 | 15/21 |
| 2026-09-05, 07:08 run 1 | 16/21 |
| 2026-09-05, 07:11 run 2 | 16/21 |

**Same model, same prompt, same set, same command: 15 to 18.** The spread is
three cases wide, which is wider than any delta this document argued from
before §10.

Per case, across the four:

| Case | Correct | Verdict |
|---|---|---|
| `conflict-contrary-finding` | 1 of 4 | flips |
| `insufficient-partial-overlap` | 1 of 4 | flips |
| `refines-narrows-scope` | 1 of 4 | flips |
| `insufficient-anecdote-without-comparison` | 0 of 4 | **always wrong** |
| `insufficient-different-population` | 0 of 4 | **always wrong** |
| `insufficient-mechanism-without-outcome` | 0 of 4 | **always wrong** |

Every other case was correct every time.

**Three cases flip and three never work.** The three that flip are the three
§7 credited a prompt revision with fixing. The three that never work were
already the ones §7 called resistant, plus `anecdote-without-comparison`, which
0.2.0 was also credited with fixing and which has since failed four times out
of four.

So the set has a stable core of 15 correct, three cases that answer at roughly
chance, and three fixed failures. Every published movement in this document
lives entirely inside the flipping three.

**The stable failures are the finding.** All three are the same shape: evidence
that is on-topic and inconclusive, read as support.
`insufficient-mechanism-without-outcome` has a cue, a dedicated rule and a
structural check written against it, and has now returned SUPPORTS in every run ever recorded here.

### The outage nearly entered the table as data

Run 3's nine unreachable cases were recorded as ordinary misses. Left alone
they would have appeared in the stability table as nine cases that changed
their answer, and dropped that run's score by nine, in the tool built
specifically to stop a delta being misread.

Fixed: a case the provider never answered is now flagged `unavailable`, every
rate is computed over the cases that reached the model, the headline prints
`[9/21 UNMEASURED]`, `--repeat` warns which runs are incomplete, and
`assessment_compare.py` refuses an incomplete report outright rather than
diffing it. **This is the third metric in this project to have scored a
non-answer as an answer**, after the grounding score that could only print
1.000 and the extraction eval that scored timed-out runs as clean. The pattern
is the same every time: the failure path produces a number instead of a gap.

### A run with no network reported 0.00, 2026-09-06

A Kimi comparison was attempted on a laptop that had lost DNS. Both runs
finished in six seconds, all 42 cases empty, and the summary read:

```
score range 0-0 of 21 across 2 runs; 0 case(s) answered inconsistently.
```

A total outage, rendered as a score and a consistency claim. Two separate
things had to hold for that:

1. `CloudProvider._probe_models` caught the DNS failure under a blanket
   `except Exception` and reported "model list unreachable, unverified" as
   **reachable**. The rule it was protecting is real: a gateway that serves no
   `/models` must not be refused. But a transport error is not a missing
   endpoint, it is a missing host, and every later call fails identically.
2. Nothing stopped the run once the provider had clearly gone. The remaining
   cases each failed in under a millisecond and printed a MISS.

Fixed: a transport error (DNS, refused, timeout) now fails the health check
before any case runs, and a run abandons itself after three consecutive
provider failures, prints that it is not a measurement, and exits 2. The
pre-existing test asserting the old behaviour was updated rather than worked
around, with the reason written into it.

**That is the fourth non-answer scored as an answer in this project**, after
the grounding score that could only print 1.000, the extraction eval that
scored timed-out runs as clean, and the outage recorded as nine misses the day
before. The recurring shape is worth naming: **when the failure path returns a
value of the same type as the success path, it gets averaged into the result.**
The fix is always the same, make the absence of an answer a distinct state
rather than a bad one.

## 11. A second model family fails the same cases, the same way, 2026-09-06

*`qwen/qwen3.8-27b`, `--repeat 2`, same 21 cases, same prompt, same command. The
only other general-purpose model this Groq key offers besides the gpt-oss pair.*

**16/21 twice. Zero cases answered inconsistently.** Validity and grounding
1.00.

The headline sits inside gpt-oss-120b's 15-18 band, so on accuracy the two are
indistinguishable. That is not the finding. The finding is *which* cases failed:

| Case | gpt-oss-120b | qwen3.8-27b | Expected |
|---|---|---|---|
| `conflict-contrary-finding` | REFINES | REFINES | POTENTIAL_CONFLICT |
| `refines-narrows-scope` | POTENTIAL_CONFLICT | POTENTIAL_CONFLICT | REFINES |
| `insufficient-anecdote-without-comparison` | IRRELEVANT | IRRELEVANT | INSUFFICIENT_EVIDENCE |
| `insufficient-different-population` | POTENTIAL_CONFLICT | POTENTIAL_CONFLICT | INSUFFICIENT_EVIDENCE |
| `insufficient-mechanism-without-outcome` | SUPPORTS | SUPPORTS | INSUFFICIENT_EVIDENCE |

**Five for five, the same wrong label.** Two independently trained families, one
of them a fifth the size, converge on the same answer for every case the larger
one gets wrong. Qwen also got `insufficient-partial-overlap` right both times,
which gpt-oss manages 1 time in 4.

### This moves the constraint from the model to the cases

The question §10 left open was whether INSUFFICIENT_EVIDENCE is `gpt-oss-120b`'s
limit or the task's. It is neither, on this evidence. **The disagreement is
between the models and the labels, and the models agree with each other.**

Reading the five back, the labels are weaker than this document has been
assuming:

1. **`conflict-contrary-finding` looks mislabelled.** The claim is *"RAG **can**
   improve factual accuracy"*. Evidence that it can also hurt does not
   contradict a possibility claim, it conditions it. REFINES is the better
   answer and both models gave it. A counterexample cannot conflict with "can".
2. **`insufficient-anecdote-without-comparison` exposes an undefined boundary.**
   Both models said IRRELEVANT for evidence that is plainly on-topic but
   establishes nothing. That is §6's unfixed defect surfacing again: IRRELEVANT
   and INSUFFICIENT_EVIDENCE both produce no proposal, so nothing in the schema
   or the prompt ever forced them apart.
3. **`insufficient-different-population` and `refines-narrows-scope` are
   genuinely contestable.** A contradicting result from a population the claim
   does not cover: non-transferable, or a conflict worth a human's attention?
   A claim of "no data loss" against a default that loses a second: over-broad,
   or wrong? Both models chose POTENTIAL_CONFLICT for both, which is the answer
   that **routes to a human**: operationally the safer error.
4. **`insufficient-mechanism-without-outcome` is the one where the label holds.**
   A documented cache-read price is not a measurement of a bill. But even here
   the claim is ambiguous between "the price is lower" (supported) and "the cost
   fell" (unmeasured).

**So three rounds of prompt engineering failed because they were trying to
instruct a model into labels the cases do not clearly support.** That is a more
useful explanation than "the model cannot judge insufficiency", and it is the
one the evidence now favours.

### The trap this creates

The obvious next move is to relabel the cases the models agree on. **Do not do
that on this evidence.** Two models agreeing is not two independent
observations: they share training data, architecture lineage, and probably
much of the same reasoning about these very categories. Model consensus is
weak evidence about ground truth and strong evidence about *shared prior*.

Relabelling to match would also raise the score from 16 to 20 out of 21, which
is exactly the shape of motivated reasoning this project has already caught
itself in once, when an extractor scoring 1.0 recall was correctly judged bad.
**A gold label is only worth revising against an argument, never against a
score.** The argument for `conflict-contrary-finding` is above and stands on the
word "can". The others need a human deciding what the category means, not a
tally of what two models said.

### Two properties where Qwen is simply better

- **It is deterministic here.** 57 real answers across three runs, zero
  disagreement, and the five failures are the same five every time. gpt-oss-120b flipped three cases across four runs. For an
  assessor whose output is audited and cached under a derivation key, a stable
  answer is worth more than a point of accuracy: §10 exists entirely because
  the larger model's variance swamped every delta measured before it. *Repeated
  on a third run, which confirmed it once a scoring bug was removed: see below.*
- **It is roughly twice as fast.** Before rate limiting took over, Qwen
  answered in 0.7-1.1 s against gpt-oss's 1.2-2.2 s. Nearly all the reported
  9-12 s per case is Groq's 429 backoff, not inference.

At a fifth the parameters, matching accuracy, deterministic, and faster, **the
27B model is the better choice for this component** on everything measured
here. That conclusion is worth more than the model comparison was expected to
produce.

### The third run confirmed it, after a bug had to be removed first

*`--repeat 3`, 2026-09-06, on a heavily rate-limited key.* The tool reported
`score range 13-16 of 21 across 3 runs; 6 case(s) answered inconsistently`,
which reads as the determinism claim collapsing. It was not. **Six cases had
exhausted their retries against a 429 and never reached the model at all**,
and the previous day's fix did not cover them.

`AssessmentOutcome.RETRYABLE_FAILURE` is a distinct value from
`SEMANTIC_ANALYSIS_UNAVAILABLE`, and only the second was listed as a
non-answer. So retry exhaustion was scored as a wrong answer with `actual=None`,
and `no result` entered the stability table as if it were an opinion. Every one
of the six "inconsistent" cases reads `X x2, no result x1`. **Not one of them
gave two different real answers.**

Counting only cases that reached the model:

| Run | Measured | Correct | Real misses |
|---|---:|---:|---|
| 1 | 18/21 | 14 (0.78) | contrary, narrows, anecdote, diffpop |
| 2 | 18/21 | 13 (0.72) | those four, plus mechanism |
| 3 | 21/21 | 16 (0.76) | those five |

`mechanism-without-outcome` is absent from run 1 only because it was one of the
dropped calls. **Across 57 real answers in three runs, zero cases gave two
different answers**, and the five failures are the same five every time they
were asked. The determinism finding survives and is now on three runs rather
than two.

**This is the fifth non-answer scored as an answer in this project, and the
first one inside the code written to stop the fourth.** The list is now: a
grounding score that could only print 1.000; an extraction eval that scored
timed-out runs as clean; an outage recorded as nine misses; a dead network
reported as 0.00; and retry exhaustion counted as a changed opinion. The
pattern named in §10 held again, one layer up: **the failure path returned a
value of the same type as the success path**, and this time the type was
"a case result with `actual=None`" rather than a float.

Fixed: both infrastructure outcomes are now non-answers, and
`ASSESSMENT_REJECTED` deliberately is not, because there the model did answer
and the answer was invalid. Under this much throttling, raise
`FORGE_LLM_MAX_RETRIES` before running: three attempts is not enough when
nearly every call is rate-limited on the first try.

### What this key cannot answer

Groq serves 14 models on it, and after removing two speech-to-text, two
text-to-speech, two prompt-injection classifiers of 22M and 86M parameters, a
safety classifier and a 7B Arabic model, four general-purpose models remain:
`gpt-oss-120b`, `gpt-oss-20b`, and the two Qwen 27Bs. **There is nothing larger
than gpt-oss-120b available here.** The "try a frontier model" question stays
open for want of a provider, and should be recorded as unresolved rather than
answered by the largest model that happened to be reachable.

## 12. What follows

**Re-run anything before believing it.** §10 is the precondition for every
item below: a single run on this set carries a spread of at least three cases.

**And review the cases before blaming the model.** §11 supersedes much of what
follows: the five hardest cases are ones two unrelated model families answer
identically and against the label. The next work is a human deciding what
IRRELEVANT and INSUFFICIENT_EVIDENCE mean when they differ, and whether a
possibility claim ("RAG *can* improve accuracy") can be conflicted at all by a
counterexample. That is category design, not prompt engineering.

1. **Do not write a fourth fix for this class.** Three failed, and §10 says
   the first one arguably never worked at all. The next honest move is a
   different model, not a different prompt or wrapper.
2. **Try one stronger model on both sets** before concluding the task is
   infeasible. If a larger model handles INSUFFICIENT_EVIDENCE cleanly, the
   finding is about this model; if it does not, the finding is about the task,
   and that is worth knowing either way.

   `--model` overrides the configured model for one run, so the comparison
   needs no config edit and leaves `forge.env` alone:

   ```
   python3 scripts/assessment_eval.py --provider cloud \
       --model MODEL --json > fitted-MODEL.json

   python3 scripts/assessment_eval.py --provider cloud \
       --model MODEL --json \
       --dataset engine/forge/evaluation/data/assessment-holdout-v1.yaml \
       > holdout-MODEL.json

   python3 scripts/assessment_compare.py holdout-gpt-oss-120b.json holdout-MODEL.json
   ```

   The comparison step is not optional bookkeeping. **A model swap can score
   identically and disagree on a third of the set**. That is exactly what the
   corroboration check did, 13/18 both ways while fixing one case and breaking
   another, and a headline delta showed none of it. `assessment_compare.py`
   prints the per-class table and names every case that moved in each
   direction. It refuses to diff two different datasets, and refuses a scripted
   run, whose accuracy is 1.0 by construction.

   Both sets, because the fitted set has seen three rounds of prompt work aimed
   at it and the held-out set has seen none. A model that scores well on the
   fitted set alone has told us nothing the prompt did not already encode.

   The comparison to beat, `openai/gpt-oss-120b`: **0.86 fitted (18/21), 0.72
   held-out (13/18)**. But the headline is not the number to read. Read
   INSUFFICIENT_EVIDENCE, which sat at **4/6 fitted and 6/10 held-out** and did
   not move on any variant tried, and within it the three held-out failures in
   the table above, where the passage contains the sentence that should have
   blocked the inference and the model asserted SUPPORTS anyway. Those three are
   the test. A model that scores 0.80 overall while still asserting support from
   a self-selected sample has not fixed anything this document is about.
3. **Design around it rather than through it.** Phase 4 already routes every
   proposal to a human. The measured position is that `CLAIM_EVIDENCE`
   proposals in particular cannot be trusted unreviewed, which is an argument
   for keeping that gate, not for removing it once accuracy "improves".
4. ~~**Repeat runs for variance.**~~ Done, and it invalidated most of the
   deltas above. See §10. Left struck through rather than deleted because the
   warning was written before the measurement and turned out to be the most
   important line in the document.
5. **Keep contradiction detection human-routed.** Unchanged by anything
   measured here.
