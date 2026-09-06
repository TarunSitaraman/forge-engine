# Extraction measured against the vault's own structure

*Written 2026-09-06. The harness is built and verified offline; the model run
is outstanding and this document will gain a results section when it happens.
Nothing below is a claim about any model's extraction quality.*

## 1. Why a second extraction eval

`forge extraction-eval` scores a labelled set of six hand-written cases. It is
a real instrument and it closed a real gap: before it existed, a prompt rewrite
was judged by reading fifty proposals by hand. But it has a structural limit
that no amount of care removes. The expected concepts in it were chosen by the
same person who wrote the prompt, on text that person also chose.

`forge bootstrap` produces a second reference set that has none of that
property. It derives **545 concepts from the vault's directory listing with
zero model calls**: a human decided `Binary Search` deserved one canonical page
and created it, years before any of this existed. That decision is exactly the
judgement extraction is trying to reproduce, and it was recorded for a
completely different reason.

## 2. What is measured, and what is refused

Three numbers, and recall is not one of them.

**Self-recovery.** For one page, does extraction over that page's own text
recover that page's own concept? This is the only recall-shaped question the
vault can answer honestly.

**Junk rate.** Share of emitted names appearing on the forbidden list the
labelled set already carries: 25 strings including `RAM`, `Answer`,
`maxmemory` and `VARCHAR(n)`, none of them hypothetical, every one observed
coming out of this corpus in the August 2026 samples. The script imports that
list from the labelled set rather than retyping it, so the two evals' junk
rates mean the same thing and can be quoted side by side. Retyping would let
them drift while both columns still read "junk", which is the worst version of
this.

**Off-vocabulary rate.** Share of emitted names matching no page in the vault.
Reported, and deliberately **not** called junk. A genuinely new concept the
vault has no page for lands here, and that is extraction working. It is a shape
measurement, useful next to junk rate, and quotable as neither quality nor its
absence.

**Global recall is refused.** "How many of the 545 did it find" is meaningless
per page: a page about B-trees is not supposed to mention 544 other concepts,
so a miss would not be a miss. And the failure this corpus actually had is
over-extraction, so the metric that ranks a maximally greedy extractor best is
the wrong headline. A test asserts that specific property: an extractor
emitting the whole vocabulary scores 1.000 self-recovery, and the report shows
its junk anyway.

## 3. Two decisions the vault forced

**The page title counts as the page's own name.** `Technologies/Docs/rag.md` is
titled "RAG (Retrieval-Augmented Generation)". An extractor emitting `RAG` has
recovered that page's concept by any reading, and scoring the filename stem
alone would record a miss that is not a fact about extraction. Both names are
accepted, and a hit records which one matched.

**541 names over 545 concept pages, and that is not a bug.** A decided
collision produces two namespaced concepts sharing one bare name. This vault
has four: `Heap`, `Binary Search`, `Trie` and `weekly-review`. The namespace is
disambiguation a human recorded in `concept-identity.yaml`; the extractor is
never asked to produce it and must not be scored on it. Emitting `Heap`
therefore counts as recovering either Heap page.

## 4. The finding the offline mode produced, and the fix

The scripted provider exists so the harness can be verified with no key. It
answers with the page's own name, so self-recovery is 1.000 by construction and
says nothing about any model. That much follows the assessment eval's
precedent.

What is different here: a grounding check that always passes in the only mode
runnable offline is worse than no check, because it reassures. So the scripted
answer pairs every real quote with an **adversarial probe**: the same line with
its token order reversed, built entirely from the span's own vocabulary, so
only the order-preserving half of `_grounded` can reject it. Every probe
emitted must come back dropped.

They did not. The first run reported grounding 0.528 where the probes alone
predicted 0.500.

### 4.1 The cause

`_grounded` tries a substring match first, then falls back to a longest-common-
subsequence **ratio** over words. The ratio is scale-free. A short quote matched
against a long span is therefore cheap to satisfy: the quote's words need only
appear in ascending order *somewhere*, and successive repetitions of a line
supply as many ascending positions as the quote has words.

The concrete case, from `DSA/04_Problems/Backtracking - Combination Sum.md`.
Its validation block holds five near-identical assert lines. Reversing

    assert sorted(result) == sorted([[2,2,3],[7]])

gives the tokens `sorted 2 2 3 7 sorted result assert`, and those five lines
match all eight in order:

| span | overlap | grounded |
| ---- | ------- | -------- |
| the real 5-assert block | 1.000 | yes, wrongly |
| a 2-assert version of the same block | 0.875 | no, correctly |

So the check was not broken for code. It was defeated by how many ascending
positions the span offers.

### 4.2 Why it went unseen

Every negative case previously written for `_grounded` was prose, and natural
language does not repeat its tokens this way. The tests were right about the
mechanism they targeted: the 2026-08-19 rewrite from bag-of-words to ordered
overlap fixed a real defect, where a quote inverting a span's meaning scored
1.0. What nobody tested was a span whose own text is repetitive, which in this
corpus means every DSA problem page's validation block.

### 4.3 The fix

The subsequence match is now constrained to a **window** roughly the length of
the quote (`_local_overlap`), which encodes what a quote actually is: a
contiguous passage. The window slides in quarter-window steps, so no source
region up to three quarters of a window can be split across a boundary.

The constant deserves a note, because the first version of its comment claimed
a margin that had not been measured and was wrong. Measured on the motivating
case, the reversed quote is rejected at **every factor from 1 to 11** and only
accepted at 12, where the window is the whole 100-token span again. The setting
is nowhere near a boundary and a later reader should not treat it as delicate.

What the factor does buy is tolerance for a quote whose words are all present
and in order but spread apart by words the model did not quote:

| window factor | source region tolerated | reversed quote |
| ------------- | ----------------------- | -------------- |
| 2 | 2.5x the quote's length | rejected |
| **3 (chosen)** | **3.5x** | **rejected** |
| 4 | 4.5x | rejected |
| 12 | 12x | **accepted, the defect returns** |

3 takes the extra elision headroom at no cost to the defect. A floor of 16
tokens keeps a three-word quote from being held to a nine-token window; on a
span measured here that floor is the difference between 0.667 (rejected) and
1.000.

### 4.4 Re-measured, and a harness bug found on the way

Scored over **all 545 pages** in scripted mode, which is 1,536 adversarial
probes:

| | probes emitted | dropped | survived | on pages |
| ---- | ---- | ---- | ---- | ---- |
| before the fix | 1,536 | 1,526 | **10** | 9 |
| after the fix | 1,536 | 1,536 | **0** | 0 |

Probes and pages are counted separately on purpose: one page carried two
survivors, so the first version of this table said 9 survivors against 1,526
dropped, three numbers that cannot all be true.

The eval exits non-zero on a survivor, so this is a pass/fail the offline suite
can hold, not a rate to interpret. Reverting `_local_overlap` to
`_ordered_overlap` reproduces the 10, which is what makes the check sensitive
rather than merely green.

The grounding rate itself moves only 0.519 to 0.515, which is why it is the
wrong number to read: probes are a small share of claims, and the rate sits
above 0.500 anyway because unprobed spans contribute a kept claim and no
probe.

**11 of the first pass's "survivors" were the harness, not the check**, and
that is worth recording because it nearly became a false finding. The probe was
originally built by reversing a line's *words*. That barely permutes its
*tokens* when one word holds most of them:

    result = groupAnagrams(["eat","tea","tan","ate","nat","bat"])

is four whitespace-delimited words, and reversing them leaves seven of eight
tokens in their original order. Lines like `self.parent[x] =
self.parent[self.parent[x]]` are near-palindromic in token space, and a grid of
ones and zeros reverses to nearly itself. None of those is a reordering, so
accepting them is correct behaviour. Probes are now built by reversing
**tokens**, and a line is skipped unless at least 60% of its tokens are
distinct; 98 spans have no such line and are left unprobed rather than counted
against the check.

Selection deliberately **never calls `_grounded`**. Trying scrambles until one
the check rejects would make every run pass by construction. A test asserts the
absence of that reference by parsing the function.

### 4.5 What this does and does not change

Nothing about the assessment numbers. That path grounds on **span ids**, not
quoted text, so Phase 5's classification and false-positive-conflict figures
are untouched. What moves is the extractor's claim-drop path and the two
extraction evals, and only for quotes matched against repetitive spans.

The invariant is what it was supposed to be all along: a quote is checked
against the span it came from, and being assembled from that span's own
vocabulary no longer suffices.

## 5. Running it

    # offline, verifies the harness, no key needed
    python3 scripts/concept_extraction_eval.py --limit 12

    # the real measurement
    python3 scripts/concept_extraction_eval.py --provider cloud --limit 40

One page costs up to `--max-spans` concept calls plus the same number of claim
calls. The eval default is 3 rather than the production 12, because a
rate-limited hosted key will not survive 545 pages at 12; runs at different
values are not comparable. The sample is seeded, so two runs of the same size
score the same pages.

`--limit 0` scores all 545, which is roughly 3,300 calls.
