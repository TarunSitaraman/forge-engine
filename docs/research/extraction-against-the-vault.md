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

## 4. The finding the offline mode produced

The scripted provider exists so the harness can be verified with no key. It
answers with the page's own name, so self-recovery is 1.000 by construction and
says nothing about any model. That much follows the assessment eval's
precedent.

What is different here: a grounding check that always passes in the only mode
runnable offline is worse than no check, because it reassures. So the scripted
answer pairs every verbatim quote with **the same sentence reversed**, built
entirely from the span's own vocabulary, so only the order-preserving half of
`_grounded` can reject it. Grounding should therefore read exactly **0.500**,
and a run reading 1.000 means the drop path stopped working.

It read **0.528**.

### 4.1 The cause

`_grounded` tries a substring match first, then falls back to a longest-common-
subsequence **ratio** over words. The ratio is scale-free. A short quote matched
against a long span with repeated tokens is therefore cheap to satisfy: the
quote's words need only appear in ascending order *somewhere*, and successive
repetitions of a line supply as many ascending positions as the quote has
words.

The concrete case, from `DSA/04_Problems/Backtracking - Combination Sum.md`.
Its validation block holds five near-identical assert lines. Reversing

    assert sorted(result) == sorted([[2,2,3],[7]])

gives the tokens `sorted 2 2 3 7 sorted result assert`, and those five lines
match all eight in order:

| span | overlap | grounded |
| ---- | ------- | -------- |
| the real 5-assert block | 1.000 | yes, wrongly |
| a 2-assert version of the same block | 0.875 | no, correctly |

So the check is not broken for code. It is defeated by how many ascending
positions the span offers.

### 4.2 Why it went unseen

Every negative case previously written for `_grounded` was prose, and natural
language does not repeat its tokens this way. The tests were right about the
mechanism they targeted: the 2026-08-19 rewrite from bag-of-words to ordered
overlap fixed a real defect, where a quote inverting a span's meaning scored
1.0. What nobody tested was a span whose own text is repetitive, which in this
corpus means every DSA problem page's validation block.

### 4.3 Pinned, not fixed

Both behaviours are now asserted in
`tests/unit/test_phase2_units.py`: the 5-assert span accepts the reversal, the
2-assert span rejects it. The first test says in its message that a failure
means the fallback was tightened, and that the fix is to update the test and
re-measure rather than to revert.

It is not fixed here. Narrowing the fallback to a window, or putting a floor on
the absolute number of matched tokens, changes what **every shipped extraction
accepts**, and would move the assessment and extraction numbers Phase 5 is
currently gated on. That is a deliberate separate change with its own
measurement, not a side effect of writing an eval script.

The bound on the damage is worth stating plainly, because it is a rule this
project treats as load-bearing. "Nothing is stored without evidence" still
holds in the sense that a quote is always checked against the span it came
from. What this weakens is the strength of that check on repetitive spans: a
fabricated quote assembled from such a span can pass. Claims from prose spans,
which is most of the corpus, are unaffected.

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
