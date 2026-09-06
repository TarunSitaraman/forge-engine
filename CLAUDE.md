# Forge Engine: Instructions for Claude

*Project-level context for Claude Code sessions in this repository.
Read this before making changes. It captures conventions and repo
state that aren't obvious from a fresh clone, and mistakes already
learned so they aren't repeated.*

## What This Repository Is

The **Forge Knowledge OS engine**: a Python knowledge OS that reads a
Markdown vault and builds a provenance-aware derived model from it.
Phases 0-4 are complete.

**The vault is a separate, private repository** (`TarunSitaraman/forge`).
The engine was split out of it on 2026-09-01. That matters in three
concrete ways:

1. **The corpus is not in this tree.** 42 integration tests run against
   the vault and skip without it. Set `FORGE_TEST_VAULT=/path/to/forge`
   to run them, `1,199 passed, 42 skipped` becomes `1,241 passed`. See
   `docs/test-strategy.md` §"Running the corpus tests".
2. **Vault knowledge does not belong here.** `docs/` is engineering
   documentation *for the engine*, architecture, ADRs, research,
   measurement records. Durable technology reference belongs in the
   vault's `Technologies/Docs/`, not here.
3. **Don't assume a vault path.** Vault resolution looks **only**
   upward from the working directory, then raises. It used to check the
   installed module's location first; after the split that matched this
   repository every time, so `forge index` with no `FORGE_VAULT_PATH`
   silently indexed the engine's own `docs/` and printed success. The
   rule was removed in `042d28c`, `FORGE_VAULT_PATH` is how a user
   points the CLI at their vault.

**The engine never writes to the vault** except through an explicitly
approved, flag-gated repair. Everything it derives lives in `.forge/`
and is rebuildable from scratch, delete that directory and nothing of
value is lost.

**Reading the record below:** it predates the split, so entries name
vault paths (`DSA/`, `Technologies/Docs/`, `Projects/`) and vault files
(`CONVENTIONS.md`, `ROADMAP.md`). Those are the *vault repository's*,
not files missing from this tree. They are kept as written because they
are measurement records: renaming the corpus a number was measured
against would falsify it.

## Known Stale/Legacy Items

**The vault is a second extraction reference set, 2026-09-06, and building
the harness broke the grounding check.** `scripts/concept_extraction_eval.py`
scores extraction against the 545 concepts `forge bootstrap` derives from the
vault's directory listing. The point is provenance: the six labelled cases in
`extraction-v1.yaml` had their expected concepts chosen by whoever wrote the
prompt, and these 545 were recorded years earlier for an unrelated reason.
Three numbers, and recall is deliberately not one: **self-recovery** (does the
page about X yield X?), **junk rate** against the same 25 forbidden strings the
labelled set carries, imported rather than retyped so the two are comparable,
and **off-vocabulary rate**, reported and explicitly not called junk because a
concept the vault lacks a page for is extraction working.

Its offline mode pairs every verbatim quote with the same sentence reversed, so
grounding must read exactly 0.500 and a run reading 1.000 means the drop path
broke. **It read 0.528.** `_grounded` falls back to a longest-common-subsequence
*ratio*, which is scale-free, so a short quote against a span that repeats its
tokens is cheap to satisfy. On a real DSA validation block with five
near-identical `assert sorted(result) == sorted(...)` lines, the fully reversed
quote scores **1.000** and is accepted as evidence; the same quote against a
two-assert version scores 0.875 and is correctly rejected. Every negative case
ever written for `_grounded` was prose, and prose does not repeat its tokens
that way, which is why the 2026-08-19 bag-of-words fix looked complete.

**Pinned in `test_phase2_units.py`, not fixed.** Narrowing the fallback changes
what every shipped extraction accepts and would move the numbers Phase 5 is
gated on. Do it as its own change with its own measurement. Write-up:
`docs/research/extraction-against-the-vault.md`.

**Phase 4, done 2026-08-29: the extraction-quality eval, and cross-span
dedup.** Both are deterministic and cost zero model calls.

**`forge extraction-eval` exists because every prompt and model comparison this
month came out confounded.** There was an assessment eval and a retrieval eval
and nothing measuring the thing extraction actually produces, so the `0.3.0`
prompt rewrite, the `+nothink` question and the "~35-40% concept precision"
figure all had to be recorded as unjudgeable. The dataset
(`forge/evaluation/data/extraction-v1.yaml`, 6 cases, 15 expected concepts, 25
forbidden strings) is built from **real observed failures**: `maxmemory`,
`VARCHAR(n)`, `.dockerignore`, `Answer`, `git commit`: not invented ones.

**The headline metric is `junk_rate`, not recall**, and that choice is the
whole point: the failure mode this corpus actually has is over-extraction, and
recall rewards it. `test_a_greedy_extractor_gets_full_recall_and_is_still_penalised`
pins that, an extractor emitting every noun phrase scores 1.0 recall and is
still correctly judged bad. Optimising the metric that moves in the wrong
direction would have been worse than not measuring at all.

**`forge proposals duplicates`** reports proposals that say one thing twice.
This *cannot* be a prompt rule: the extractor sends one span per call, so the
model physically cannot know another span already produced the same fact. Being
deterministic it applies retroactively to the existing corpus for free, the
same property that let the grounding audit re-check 1,170 claims at zero cost.
**Nothing is merged**; deciding two statements are one is judgement, and
judgement routes to a human.

`CLAIM_SIMILARITY` is worth remembering as a process failure caught in time. My
first value, 0.60, shipped with a docstring asserting a measured range I had
**not measured**; the real clusters score **0.262-0.492 for duplicates, 0.076-
0.170 for distinct**, so 0.60 would have found nothing while claiming to be
calibrated. The value is now **0.22**, with the caveat stated in the code that
a 0.09 gap over eight hand-picked pairs is narrow and will produce false
positives, tolerable only because the output is a review suggestion. **Under
this repo's "no measurement claim without a measurement" rule, a plausible
number in a docstring is the same defect as a plausible number in a README.**
**`forge extract-plan`** answers "what would this run cost?" before you spend
the hours, at zero model calls. Every input was already deterministic and
already stored, which spans the ingestion chunker produced, which the
extractor would select, whether the derivation cache holds a result for the
content hash, so a run's size is *computable*, and until now the only way to
learn it was to start one and watch.

**It calls the real `_select` and the real `extraction_key` rather than
restating their rules.** A preview that reimplements either would drift from
what actually runs and would then be worse than no preview, because a number in
a cost report gets believed. Verified against the record: it prices
`Technologies/Docs` at **exactly 196 calls**, the post-chunker-fix figure this
file already records. Whole vault: **3,286 calls over 642 sources**.

One real defect the test suite caught, worth remembering: **the plan priced
byte-identical files twice.** The derivation key is the *content* hash, so two
files with the same bytes share one cache entry and the run pays once. Two
duplicate fixture notes made the predicted 18 against an actual 16, and on the
real vault the error would be larger: six project packs contain an
`01-overview.md` and `_index.md` recurs throughout. Pricing is now deduped by
key, with `duplicate` as its own reported state.

`--seconds-per-call` has **no default**, deliberately. A wall-clock estimate
built on an invented rate is a fabricated measurement, and 49.0 s/call belongs
to 8B-local on 2026-08-19. That machine, that model. Without the flag the
command prints the call count and says so.

**First real extraction measurement, 2026-08-29, and the eval was wrong twice
on its first run.** `qwen3:8b` on the ASUS returned `recall=0.31 junk=0.00
grounded=1.00` over 12 calls, alongside **four `ollama_retry` timeouts**. Both
of the good-looking numbers were artifacts:

1. **A timeout does not raise.** `extract()` catches per call and returns a
   PARTIAL result with fewer concepts; the eval only recorded an `error` when
   an *exception* escaped. So a timed-out case contributed zero emitted
   concepts, and **nothing emitted cannot be junk**, so it landed in the
   denominator as clean. Cases whose calls did not all return are now **not
   scored**, every rate is over completed cases only, the summary prints
   `n/total cases`, and the CLI marks the run UNTRUSTWORTHY and exits 1.
2. **`grounded=1.00` was tautological.** The extractor drops ungrounded claims
   *before* returning, so the eval re-ran `_grounded` on claims that had
   already passed the identical check. It could only ever print 1.000, for any
   model, however badly it quotes. The denominator now includes the
   `ungrounded_quote` drops.

**A metric that cannot fail is worse than no metric, because it reassures.**
Both defects are the house pattern again: plausible output, no exception
thrown. Neither number above should be quoted; the run needs repeating with a
raised `FORGE_LLM_TIMEOUT`, and there is no cache on the eval path, so a re-run
repeats every call.

**`forge status` named the wrong model on a cloud deployment (found 2026-08-29,
Mac).** It printed `model identity : llama3.1:8b` while the cloud provider was
configured with `llama-3.3-70b-versatile`. Cause: it read
`settings.llm.models["extraction"]`, which is the **ollama** role map and is
populated with its defaults whatever the provider is. It never asked the
provider what it would run.

**Extraction itself was never wrong**, `CandidateExtractor.model_id()` calls
`provider.resolve_model()`, and `get_provider` builds `CloudProvider` with no
role map, so the derivation key correctly carried the 70B name. This was
display-only. It still mattered: this line exists *because* a wrong assumption
about the active model silently governed a 5.66 h run, so a status line
confidently naming a model you are not running is the exact failure it was
added to prevent. Status now asks the provider, and a test asserts it agrees
with what extraction would cache under.

**The cloud health check could not fail, and that cost a whole debugging cycle
(2026-08-29, Mac).** Groq had **decommissioned `llama-3.3-70b-versatile`**:
the id `config/forge.env.example` recommended. `forge status` reported
`cloud (OK)`, `forge model-test` reported `reachable: True`, and then **every
one of 12 extraction-eval calls failed** and all four capability probes scored
0/3. `CloudProvider.health()` had only ever checked that a credential *string
was present*.

`health()` now asks `GET /v1/models` for OpenAI-compatible hosts, which costs
no generation and answers both real questions: **is the credential accepted**,
and **is the configured model actually offered**. A 401 is reported as a
rejected credential; a model missing from the list is a hard failure that
**names near-matching ids the host does offer**, because model ids rotate and
the replacement is usually adjacent. A host with no `/models` endpoint is
reported *unverified*, not failed: refusing to run against such a gateway
would be worse than not checking. The verdict is cached per provider instance:
`extract()` calls `health()` once per source, which is 642 probes on a full
vault run, and a run must not change provider under its own feet.

**Also fixed: the eval said `failed: llm_error` six times.** The layer that
caught the error, and nothing about the error. A rejected model name, a bad key
and a timeout were indistinguishable. Failures now carry `kind: message`,
deduplicated, and when *every* case fails identically the CLI says so outright:
that is configuration, not extraction quality.

**Never quote a model id from `config/forge.env.example` without checking it is
live.** That file is a typo-guard for endpoints; ids rotate faster than it does.

**The groq preset made every call fail, and it was in the repo the whole time
(2026-08-29).** With a live credential and a confirmed-offered model,
`model-test` still scored 0/3 on all four tasks. The real error, once surfaced,
was **HTTP 413: "tokens per minute (TPM): Limit 8000, Requested 8595"** on a
~400-token prompt. **Groq counts reserved output against the TPM budget**, so
the preset's `max_tokens: 8192` exceeded the entire free-tier per-minute
allowance on its own, every request was rejected before a prompt token was
added. Now **4096**: under the ceiling, and high enough that a reasoning model's
thinking does not truncate the JSON (`max_tokens` caps thinking *plus*
response).

The table had already reasoned this through for **cerebras**, "free tier caps
the *context* at 8K, so an 8192 output ceiling cannot be honoured alongside any
prompt", and missed that groq has the same trap by a different mechanism.
**A preset's `max_tokens` is a ceiling that must fit the host's free-tier
budget, not the model's capability.** Both known cases are now pinned by tests;
the other presets are deliberately *not* asserted, because nobody has measured
their budgets and inventing one would be the same defect as a plausible number
in a docstring.

**Retries had no backoff at all, so a rate limit could never be survived
(2026-08-31).** With the ceiling fixed, `model-test` still reported
`relationship_extraction 0/3`, and the retry log showed **three attempts
inside 120 milliseconds**, every one a 429. A rate limit is per *minute*; the
whole retry budget was spent before the window moved. The loop simply had no
`sleep` in it.

`_retry_delay` now waits, doubling per attempt, and **honours the host's
`Retry-After` header** over its own guess: the host knows its own window, and
guessing over the top of it wastes the quota. Capped by `retry_max_wait` so one
absurd header cannot park a run, and an unparseable (date-form) header falls
back to the backoff instead of raising. `FORGE_CLOUD_RETRY_BACKOFF` configures
it; tests pass `0.0` so the suite never sleeps.

**The wider point: the first cloud numbers measured Groq's free tier, not
`gpt-oss-120b`.** `structured_concept_extraction 3/3` and
`simple_claim_extraction 3/3` are real; every `0/3` beside them was a 429. Do
not quote **58%** as a capability. It is a throughput artifact of an
unauthenticated-tier limit and a missing sleep.

**First trustworthy extraction-quality measurement, 2026-08-31.**
`openai/gpt-oss-120b` via Groq, prompt `0.3.0`, all 6 cases completed:
**junk 0.000, recall 0.472, grounding 1.000 over 23 claims**, ~1.0-1.6 s/call
against 49 s/call on 8B-local. Full reading in
`docs/research/extraction-cost.md` §2c.

**The junk rate is the result that matters, and it is narrower than it looks.**
Zero of the 25 forbidden strings came back. Those are real 8B output
(`maxmemory`, `VARCHAR(n)`, `.dockerignore`, `Answer`), so the `0.3.0`
exclusion rules work on the failures they were written for. But the denominator
is 25 specific strings, not the space of bad concepts: the run emitted
`Dockerfile` as an extra, which the prompt forbids in general and which is not
on the list. **0.000 means "no known failure recurred", not "no junk."**

**`recall=0.472` understates and was deliberately left alone.** Most misses are
surface variants, `sliding window` vs emitted `sliding window pattern`,
`rebase` vs `rebasing`. `score_case` matches through `normalize()`, which folds
case and punctuation but not morphology; `proposals/dedup.py` already has a
`_stem()` that would collapse them. **Loosening the matcher immediately after
seeing it would raise the score is motivated reasoning**, the same defect as
choosing a threshold to suit an outcome. If added, it must apply to the
forbidden list too, and both numbers recorded. Treat 0.472 as a floor under
exact matching.

Six cases cannot give a precision figure and one run cannot give a rate. What
this enables is *relative* comparison, prompt against prompt, model against
model: which is precisely what was missing.

Test count **939 → 1,021**.

**Phase 3 started 2026-08-27, `forge ask` and `forge upstream`.**

**`forge ask`** answers from the vault in **one model call**, against 3,372
calls and 153 h to pre-extract the corpus. Two rules make it trustworthy: a
retrieval miss makes **no call at all** (the model never falls back on its own
knowledge, which would make the answer untraceable), and citations are
**verified deterministically.** The model cites `[n]`, every `n` is checked
against the passages supplied, and `[7]` when six were given is reported as a
defect rather than rendered. Same discipline as extraction's quote grounding.

Two retrieval fixes came out of using it on one real question:
1. **`docs/` was answering vault questions.** The engine's own manual took
   five of the top eight spans for "what is retrieval augmented generation?".
   `SearchQuery.exclude_sources` added; answering excludes `docs/` by default.
2. **That exclusion must be a path prefix, not a substring**, and this cost
   real time. `"docs/"` is a substring of `Technologies/Docs/rag.md`, so
   excluding the engine manual silently deleted the entire canonical technology
   reference folder and the best answer vanished from the top 40.
**Title/heading boosting** implemented and swept against the 24-query set: 1.0
→ R@10 0.489 / MRR 0.482; **1.25 → 0.510 / 0.526**, the only value improving
both; 2.0+ is worse than no boost. 24 queries is small and +0.021 recall is
about half a query, 1.25 is the best available choice, not a tuned optimum.

**`forge upstream`** answers "have the documented repos moved on?", the
`Projects/` packs describe four external repos that change without the vault
noticing. **Detection is `git ls-remote` against a commit recorded in
frontmatter**, chosen over the GitHub API deliberately: no token to store or
leak, private repos work through existing git credentials, it is not
GitHub-specific, and there is no meaningful rate limit. Zero model calls. The
four packs now declare `upstream_repo`; they are **deliberately left
`unpinned`**, because `--pin` asserts "I have reviewed this pack against the
repo as it stands" and nobody has. Rewriting a drifted pack needs judgement and
is a separate step. This only ever reports.

**Phase 2 of the direction plan, done 2026-08-27: the knowledge graph is
populated: deterministically, with zero model calls.** `forge bootstrap
--apply` derives **524 concepts and 2,446 edges** from vault structure in under
a second. Extraction had spent 5.66 h on one twentieth of the vault and
returned `RAM`, `Answer`, `Fluency`, `VARCHAR(n)`; a 25-concept random sample of
the bootstrap was **25/25 real curated names**, the precision claim that was
*asserted* in the direction plan, now measured.

**The premise: this vault's concepts are its filenames.** A human already
decided `Binary Search` deserves one canonical home and created the page. That
is the exact judgement the extraction prompt kept failing to reproduce, and it
was sitting in the directory listing.

Three things the domain decided for me, not the other way round:
1. **Edge type is forced.** Concept-graph types are `{RELATED_TO, PART_OF,
   DEPENDS_ON, IMPLEMENTS, EXPLAINS}`; deterministic code may assert
   `{MENTIONS, DERIVED_FROM, PRECEDES, RELATED_TO, ABOUT}`. The intersection is
   **exactly `RELATED_TO`**. My first attempt used `MENTIONS` and
   `forge diagnostics graph` rejected all 2,446 edges (GR004). `RELATED_TO`
   requires a score; these carry `1.0` with a rationale saying *human-authored
   link, not a computed similarity*, so nobody later reads it as a measurement.
2. **Provenance is `USER_ASSERTION` / `DETERMINISTIC`.** The one tier allowed
   to stand without supporting evidence, which is right because the evidence
   *is* the file.
3. **GR008 (orphan concept) was relaxed, on principle not convenience.** Its
   rule is "nothing explains why it exists"; a canonical vault page explains a
   concept at least as well as a proposal does, a proposal is a model's
   suggestion a human approved, a page is the human's own act. A deterministic
   concept naming a real `vault_path` is now explained.

**Undecided collisions are left out of the graph entirely** rather than split
under invented namespaces, the engine must not fabricate a distinction the
user never drew. With the four recorded decisions in place there are currently
zero. 118 pages are skipped as navigation, chapters (`01-overview.md` exists in
six project packs) or point-in-time artifacts; `DSA/00_Index/` is excluded
wholesale as a folder of hubs. Test count **861 → 899**.

**Phase 1 link cleanup, done 2026-08-27: unresolved links 274 → 53, ambiguous
3 → 0.** Three fixes, in descending order of value:
1. **The link resolver never read `config/concept-identity.yaml`.** `Heap`
   (74x), `Binary Search` (66x) and `Trie` (40x) were reported unresolved even
   though a human had already decided what each bare name means, **180 of 274
   occurrences, 66%, cleared by one change**. `LinkIndex` now carries a
   `decided` map, keyed through `normalize()` because the config stores
   collisions under a normalized key while links are written in display form.
   A decision naming a file that is not a candidate is ignored, and an
   *undecided* collision still stays AMBIGUOUS: the engine must never guess.
2. **29 wikilinks to concepts that should never be pages** (`Visited Tracking`,
   `Adjacency List`, `Wrong Greedy Move`) unlinked, keeping the words. Under
   the one-canonical-home rule `Adjacency List` belongs *inside* `Graph.md`;
   creating stubs would violate "don't add frontmatter/pages for later". One
   retarget applied via a rule, not judgement: retarget only where exactly one
   page's stem ends with `- <target>`. `Cycle Detection` correctly fell through
   that rule because several such pages exist.
3. **Markdown links to real non-`.md` files were reported MISSING** (the index
   holds only Markdown). `LinkIndex.other_files` fixes it. A checker that
   flags working links teaches people to ignore it.

**The 53 that remain are not rot.** Every one is a wikilink to a DSA problem
page that does not exist *yet*. Recorded as a 40-page backlog in `ROADMAP.md`;
`forge diagnostics links` is the live checklist.

**Conventions conflict decided:** `DSA/` keeps its own Title Case / `dsa/` tag /
mandatory-frontmatter standards, everything else follows `CONVENTIONS.md`.
Evidence: DSA conforms to its local rules at 100% / 86.7% / 97.0% over 369
files, and Obsidian resolves wikilinks by filename stem: renaming DSA to
kebab-case would break ~4,000 links to gain nothing. Recorded in
`CONVENTIONS.md` and `DSA/Documentation Standards.md`. **`forge diagnostics
conventions` still prints UNRESOLVED**, because there is no mechanism to record
the decision the way `forge identity decide` records an identity one. That is a
known follow-up, not an open question. Test count **853 → 861**.

**Phase 1 of the direction plan, done 2026-08-27 (283 frontmatter repairs
applied).** The vault's `related:` fields are now valid YAML and
`forge diagnostics frontmatter` reports **zero errors** (was 68 errors, 233
warnings, 283 repairable files). Only FM003 remains, 280 files with no
frontmatter at all, which is informational and correct per `CONVENTIONS.md`
("only when it carries real metadata").

**The repair had to be fixed first, and this is the part to remember.** It
emitted `related: ["Pattern Index", "Template Index"]`: valid YAML, and a
silent catastrophe: `CorpusIndexer` builds the `related` graph by *text
extracting* `[[...]]` from raw frontmatter (`indexer.py` → 
`extract_wikilink_values`), with no fallback to the parsed YAML. Applying it
would have dropped **746 `related:` edges to zero**, the exact edges Phase 2
needs to seed the knowledge graph, and turned Obsidian property links into
plain text. Now emits `["[[Pattern Index]]", "[[Template Index]]"]`, which is
valid YAML *and* still text-extractable *and* still an Obsidian link. Verified
on the real corpus: 746 edges before, 746 after, and wikilinks went **4,113 →
4,131** because the 18 truncated-bracket cases now parse.
**General rule: before applying a mechanical repair corpus-wide, check what
reads the field and how.** Valid-YAML was never the whole requirement.
`forge proposals approve-all --apply` added for the batched path; it uses the
same `ProposalApplier`, so refusals and per-file backups are identical to the
single-proposal path. Test count **845 → 853**.

**Found 2026-08-20 (`+nothink` ran an entire extraction unnoticed):**
`FORGE_OLLAMA_THINK=0` was exported for the §8 experiment and **stayed set**, so
the whole 5.66 h / 416-call `Technologies/Docs` run happened with reasoning off.
It surfaced only by accident in a cache-hit log line showing
`model_id=qwen3:8b+nothink`. **The caching layer was correct throughout**,
`identity_variant()` kept the modes in separate derivation keys, exactly as
designed. What was missing was any way to *see* the active mode: `forge status`
showed the provider but never the model identity. Now fixed, with an explicit
warning when reasoning is off.
**Consequence: the 25+25 quality sample is confounded.** The `0.3.0` prompt
rewrite still stands (the prompt genuinely never defined "concept"), but do NOT
quote "~35-40% concept precision" as a property of the prompt. It is a property
of one run in a mode nobody knew was on. Separating them needs a think-on run of
the same scope. Test count **824 → 827**.

**Prompt rewrite 2026-08-19 (`extract-prompts/0.2.0` → `0.3.0`), driven by a
25+25 random sample of the run's output.** Claims were ~60-70% usable; concepts
were **~35-40%**, the sample returned `RAM`, `HTML`, `Answer`, `Vector`,
`Fluency`, `maxmemory`, `VARCHAR(n)`, `git commit`, `.dockerignore` as
"concepts". Approving those would have put junk in the graph *permanently*,
against the vault's one-canonical-home principle. Four failure modes, each now
addressed by an explicit prompt rule and pinned by a test in
`TestPromptContract`:
1. **Concept over-extraction.** No definition of what qualifies. Now: "would
   deserve its own reference page", with explicit exclusions for generic words,
   commands, flags, config keys, type names, file names, and `X and Y` pairs.
2. **Document-referential claims** ("The text provides a link to..."), now
   forbidden; state the underlying fact or omit.
3. **Near-duplicate claims.** The sample had three phrasings of one pub/sub
   fact in 25. Now explicitly deduplicated.
4. **Table rows rewritten as prose.** The 2.56% grounding failure. Now: quote
   the row exactly, pipes included, or make no claim.
Bumping `PROMPT_VERSION` invalidates cached extractions **by design**. That is
what makes a prompt edit safe, and a test asserts it. Re-running
`Technologies/Docs` after the chunker fix is **196 calls, not 416**. Test count
**817 → 824**.

**Measured 2026-08-19 (first quote-fidelity number):** the grounding audit over
the run's 1,170 claims found **30 ungrounded = 2.56%**. Crucially these are
**not hallucinations**, 29 of 30 are cheat-sheet *table rows rewritten as
prose* ("The command `az login` is used to login." against a `| az login | Sign
in |` row). Every fact is present; the *sentence* is not, so it is not a quote,
and dropping it is correct. The 30th (`` `INNE key` `` where Redis says `INCR`)
is a real fabrication and is the kind of thing hand-review would miss. Failures
cluster entirely in the table-heavy docs (azure, docker, kubernetes, redis), so
**this is an extraction-prompt problem, not a model problem.** Asking for a
verbatim quote from a table asks for a sentence the table does not contain. It
is the concrete first case for the still-missing extraction-quality eval.
`forge proposals audit-grounding --reject --no-dry-run` bulk-rejects them;
already-decided proposals are skipped. Test count **810 → 813**.

**Fixed 2026-08-19 session (audit reachability):** the grounding audit shipped
as `scripts/audit_grounding.py`, which is the wrong shape for this project. The
engine is installed with pipx, so the `python3` on PATH is **not** the
interpreter that owns its dependencies, running the script on the machine that
actually held the store failed with `ModuleNotFoundError: No module named
'pydantic'`. Logic moved to `forge/proposals/grounding_audit.py` and exposed as
**`forge proposals audit-grounding`**; the script is now a thin wrapper.
**General rule: an operational tool a user runs belongs on the `forge` command,
not in `scripts/`.** `scripts/` is for development against a checkout. Test count
**808 → 810**.

**Fixed 2026-08-19 session (first full extraction run):** the run cost **2.1×
more model calls than necessary** because `forge index` and `forge ingest` write
spans to the same table for different jobs, Phase 1 heading-delimited spans for
retrieval, ingestion structural/sentence-split spans for evidence: sharing a
document. The unchanged-source short-circuit compared only the content hash, so
an indexed-then-ingested vault (the order the runbooks recommend) extracted over
Phase 1's boundaries: **208 spans / 416 calls instead of 98 / 196**. Fixed by
filtering `_spans_for_source` to the ingestion chunker; Phase 1's spans are kept,
not deleted, since retrieval uses them. **Same class as the 2026-08-15 fix,
"unchanged" describes the *source*, never the derived state.** Second
occurrence, so treat any short-circuit here with suspicion. The CLI also printed
`no work done` against all 19 sources while extraction ran, which is what hid it.
Real measured latency: **5.66 h, 49.0 s/call** over ~1,100-char spans: 4.6×
faster than the 455 s/span three-span sample implied, because span size differed.
Test count **806 → 808**.

**Fixed 2026-08-19 session (grounding pass, found while the ASUS was
extracting):** `_grounded`: the function enforcing *"nothing is stored without
evidence"*, compared **bag-of-words overlap** at a 0.6 threshold, ignoring word
order entirely. Any quote reassembled from the span's own vocabulary scored 1.0
and was stored as evidence, **including one that inverted the span's meaning**
("RAG does not improve accuracy" against a span saying it does). Only a quote
using foreign vocabulary was caught, which is exactly the single negative case
the existing test used, which is why this survived.

Replaced with two order-preserving checks: a squashed-to-alphanumerics substring
match (absorbs whitespace, curly quotes, hyphenation, punctuation; `...` elision
supported), then a longest-common-subsequence ratio over words at **0.9**. The
threshold was chosen from a measured margin, not by feel: every legitimate
quote form scored **1.000**, every reassembled one **0.500-0.857**. Both sides
are pinned by tests.

**`EXTRACTOR_VERSION` was deliberately NOT bumped.** It is part of the
derivation key, so bumping it invalidates every cached extraction result and
forces a full re-run. Instead `scripts/audit_grounding.py` re-checks already
stored proposals against the new rule at **zero LLM calls**, grounding is a
deterministic string check, so it applies retroactively for free. Anything it
reports was admitted under the old rule and should be rejected, not approved.
Revisit the version bump once no uncached corpus is at stake. Test count
**792 → 806**.

**Fixed 2026-08-19 session (pre-extraction pass):** `forge proposals list
--status PENDING`, the exact command `docs/research/extraction-cost.md` §4's
runbook tells you to run after extraction, crashed with a raw `ValueError`
traceback out of `enum`. `ProposalStatus` has upper-case *names* and lower-case
*values*, and the CLI called `ProposalStatus(status)` directly. Fixed with an
`_enum_option()` helper that lower-cases, and on failure exits 2 listing the
valid set (matching the `SafetyClass` handling already in `approve-all`). The
`--type` help string was also missing three real values (`claim_evidence`,
`claim_refinement`, `claim_conflict`) and `--status` was missing `activated`.
Worth noting *why* this survived: every proposal test drove the service layer
directly, so no test ever passed a filter string through the CLI. Test count
**782 → 792** on the new parsing tests.

**Fixed 2026-08-17 session (macOS CLI pass):** the engine is now
installable as a global `forge` command. Two changes to the engine
itself, not just docs:

- **Vault resolution no longer falls back to the current directory.**
  `_find_repo_root` walked up from `__file__` and returned `Path.cwd()`
  when it found nothing. That is invisible under an editable install
  (where `__file__` *is* in the repo) but wrong under a real install:
  `forge index` in, say, `~/Downloads` treated that directory as the
  vault, wrote a `.forge/` into it, and printed a success line. Replaced
  by `_find_vault_root` (returns `Path | None`) plus `_resolve_vault_root`,
  which tries the module location, then upward from cwd, then **raises
  `ConfigError`** with an actionable message. The CLI already mapped
  `ConfigError` to exit 2, so the UX came for free. Covered by
  `tests/unit/test_config.py` (15 tests, new file).
- **Tab completion enabled** (`add_completion=True`), so
  `forge --install-completion` works. macOS defaults to zsh.

Test count **744 → 759** on the new config tests. `docs/cli.md` gained a
macOS install section (Homebrew Python + `pipx install --editable`, since
system Python 3.9 is under the 3.10 floor and Homebrew's is PEP-668
externally-managed), a vault-resolution-order subsection, and a
troubleshooting table. Note `--vault` is a **per-command** option
(`forge index --vault X`), not a global one: easy to get wrong when
writing docs or error messages.

**Also 2026-08-17 (cloud provider, per-machine setup):** the intended
deployment is now **cloud on the Mac, Ollama on the ASUS**. That is pure
configuration (`FORGE_LLM_PROVIDER=cloud` + `ANTHROPIC_API_KEY` in
`~/.zshrc`; the ASUS needs nothing since ollama is the default), but
getting there exposed a bug that made the cloud path **impossible**, not
merely unmeasured:

- **`CloudProvider` sent `temperature` on the Anthropic wire format.**
  Forge asks for `temperature=0.0` everywhere for determinism
  (`extractor.py`, `assessor.py`, `spike/capability.py`), and current
  Anthropic models reject non-default sampling parameters,
  `temperature`/`top_p`/`top_k` are 400s on Opus 4.7+, and on the
  configured `claude-sonnet-5` any non-default value is too. So every
  cloud call would have failed on the body. Removed for Anthropic only;
  the OpenAI-compatible path still sends it, since those gateways accept
  it. **Do not add it back**, `tests/unit/test_providers.py` asserts the
  whole `temperature`/`top_p`/`top_k` family is absent.
- **`max_tokens` raised 2048 → 16000** (`CloudSettings` and
  `CloudProvider`). Current models think by default and `max_tokens` caps
  thinking *plus* response text, so 2048 could be consumed by reasoning
  and truncate the JSON: surfacing as a structured-output failure rather
  than the budget problem it is. An explicit per-request `max_tokens`
  still wins.

`docs/research/provider-availability.md` §3 got a **correction block**
rather than a rewrite (per that document's own convention): its inference
that "the request shape is well-formed enough to be authenticated" was
wrong, auth is checked *before* body validation, so the 401 probe said
nothing about the payload. General lesson recorded there: **a 401 cannot
validate a request body.** The cloud path is still unmeasured; it has now
simply never completed a call, which is a different and better-understood
statement than before. Test count **759 → 763**.

**Also 2026-08-17 (open-weights, no Anthropic key):** there is no
Anthropic API key, so the deployment is now **ASUS primary + a hosted
open-weights fallback**. No code was needed to *support* that, the cloud
provider already had an `openai` vendor, which is a *wire format*, not a
company, and is what Groq / OpenRouter / Together / vLLM / llama.cpp /
LM Studio all speak. Two bugs on that path did need fixing:

- **The OpenAI-compatible branch left system messages where they fell.**
  `structured()` appends the schema instruction as a `role="system"`
  message *after* the user turn; the Anthropic branch hoists every system
  message into the top-level `system` field, so ordering never mattered
  there. Through a gateway the messages are rendered by the model's own
  chat template, and many templates assume a single *leading* system turn: several drop a trailing one. That would have silently deleted the
  schema instruction and failed every extraction with a 200 and
  unparseable prose. System messages are now collapsed and hoisted to the
  front for that vendor too, so both wire formats are semantically equal.
- **`max_tokens` was not env-configurable**, and the 16000 default (sized
  for a 128K-output frontier model) is rejected outright by gateways
  serving models that cap at 4096-8192. Added `FORGE_CLOUD_MAX_TOKENS`.

**When adding a provider or wire format, check the message *ordering*, not
just the fields.** Both cloud bugs so far were shape bugs that unit tests
with a stub transport still passed, because the stub does not run a chat
template or validate against a real model's constraints.

Docs record a **re-measure step**: the 5/5 belongs to Qwen3 8B on Ollama
and does not transfer to any other model. `scripts/assessment_eval.py
--provider cloud` and `forge model-test` both work against the
OpenAI-compatible vendor (`run_spike` is provider-agnostic despite the
"local-model" naming) and record which provider produced the result.
Test count **763 → 768**.

**Also 2026-08-17 (setup moved into the engine):** provider setup used to
be a dozen `export` lines pasted into `~/.zshrc` on each machine. Three
things now live in the repo instead:

- **A per-machine settings file**, `~/.config/forge/forge.env`
  (`$XDG_CONFIG_HOME` honoured; `FORGE_ENV_FILE` overrides). Resolution is
  three layers, highest first: explicit argument → process environment →
  file. `config/forge.env.example` is the template, with all three
  profiles in it; a test asserts the shipped example still parses.
  **Loading it never mutates `os.environ`.** That invariant already had a
  test and had to be preserved, which is why the credential is resolved
  through `config.env_value()` (process env, then file) at call time
  rather than by exporting the file. `CloudProvider.api_key` uses it.
  Parsing is deliberately not dotenv: no interpolation, no command
  substitution, since the file holds a key.
- **Cloud presets** (`FORGE_CLOUD_PRESET`), a table in `config.py`
  mapping a host name to `base_url` + `api_key_env` + `max_tokens`, for
  groq, openrouter, together, cerebras, fireworks, lmstudio, llama-cpp,
  vllm. A preset supplies **defaults only**; every field stays
  overridable, an unknown name raises with the list rather than silently
  using Anthropic's endpoint, and the *model* is never guessed: a preset
  with no `FORGE_CLOUD_MODEL` is a startup error naming that variable.
  Third-party URLs can change, so the explicit variables stay
  authoritative; treat the table as a typo-guard, not an integration.
- **`forge status` reports the settings file** and whether it loaded.

Also: `Settings.load` now wraps `LLMSettings` construction so a bad
provider knob is a `ConfigError` (clean exit 2) rather than a raw pydantic
traceback, via `_first_message()` which pulls the human sentence out of a
`ValidationError`. Test count **768 → 782**.

## The Engine (Phases 0-4 complete)

*Full detail in `docs/`. This is the orientation a session needs before
touching Python in this repo.*

**Layout**

| Path | What |
|---|---|
| `engine/forge/domain/` | Pure domain model. No storage, no HTTP, no LLM. |
| `engine/forge/corpus/`, `parsing/` | Deterministic vault indexing and Markdown parsing. |
| `engine/forge/sources/`, `ingestion/` | PDF/Markdown acquisition, chunking into spans. |
| `engine/forge/extraction/`, `matching/` | LLM candidate extraction; concept matching. |
| `engine/forge/proposals/`, `activation/` | Proposed changes; approved changes becoming canonical. |
| `engine/forge/graph/`, `retrieval/` | SQLite knowledge graph; FTS5 search. |
| `engine/forge/evolution/` | Phase 4: LangGraph workflow that evaluates new evidence against existing knowledge. |
| `engine/forge/llm/` | Provider abstraction: ollama / cloud / mock. |
| `docs/` | Engineering docs for the engine, distinct from the vault's own content. |
| `tests/`, `scripts/` | 1,241 tests; demos and per-phase validation scripts. |

**Rules that are load-bearing, not stylistic**

- **The vault is read-only to the engine.** Enforced by tests that
  byte-compare Markdown before and after every operation.
- **Provenance floor rule.** A derived object can never claim stronger
  provenance than its weakest input. Enforced in a pydantic validator,
  so a violating object cannot be constructed.
- **A model may never assert `SOURCE_FACT` or `USER_ASSERTION`.**
- **Nothing is stored without evidence.** A claim whose quote cannot be
  found in the source is dropped and the drop is reported.
- **Model reasoning never mutates knowledge directly.** It produces a
  Proposal; a human approves; activation applies it.
- **Deterministic work stays deterministic.** Parsing, hashing,
  chunking, matching, graph traversal, and impact classification make
  **zero** LLM calls, and tests assert the call count.
- **No measurement claim without a measurement.** See
  `docs/research/`, where something could not be measured, that is
  recorded as unmeasured rather than estimated.

**Working on the engine**

```bash
pip install -e ".[dev]"          # needs Python 3.10+
python -m pytest tests           # 1,199 passed, 42 skipped, offline, no model
bash scripts/validate_phase4.sh  # proves the phase's exit criteria by executing them
python scripts/phase4_demo.py    # the end-to-end story

# the 42 skips are the corpus tests; point them at a vault checkout
FORGE_TEST_VAULT=/path/to/forge python -m pytest tests   # 1,241 passed
```

CI and the whole test suite run **offline** against a scripted provider.
Never add a test that requires a live model.

**Run the corpus tests before trusting a change to indexing, link
resolution, or diagnostics.** They are the ones that validate the engine
on material that actually drifted, and skipping is their honest default,
not a weakened suite. The bug that split this repository's identity
config from its vault (see the entry above) was invisible until they ran
with a working directory outside the vault.

**Real-model status (2026-08-14):** Qwen3 8B via Ollama on the RTX 4050
box scored 5/5 on the assessment set, schema-valid output and correct
grounding on every case, including both adversarial ones, with zero
false-positive conflicts. That is a smoke test that passed, **not** a
characterisation: five cases cannot establish a rate. The cloud path is
still unmeasured. Local latency: typical cases 40-60 s, but the same
adversarial case has exceeded both the 120 s default and a raised 300 s
(2026-08-19 re-run), and a retry costs the whole timeout first. Raise
`FORGE_LLM_TIMEOUT` well past 300 for long runs. A 5-case *mean* is not a
useful latency number here. One timeout moved it 78% while three of four
measurable cases got faster.
`FORGE_OLLAMA_THINK=0` (reasoning off) was measured on 2026-08-19 and
**rejected**: 2.5× faster on typical cases, but it turned partial overlap into a
false-positive `POTENTIAL_CONFLICT`, dropping classification to 4/5. It stays
opt-in and off. Its effect on *extraction* is unmeasured and unmeasurable today. There is no extraction-quality eval, only assessment and retrieval. Don't
re-run this experiment expecting a different answer; build the extraction eval
first.
Read `docs/research/provider-availability.md` §6 and §8 before quoting any of
these numbers as a rate.
## Git Workflow Notes

- This repo has an active GitHub remote (`origin/main`); commits made
  during sessions have generally been pushed immediately after each
  logical batch, not held until end-of-session.
- **Attribution: Tarun is the author of every commit. Always.**
  - Set the identity before committing:
    `git config user.name "TarunSitaraman"` and
    `git config user.email "tarunsitaraman134@gmail.com"`.
    This is the identity for every commit made from a Claude Code
    session. Older commits (before 2026-08-13) carry
    `Tarun Sitaraman <mfsbyo@gmail.com>`, which was the identity actually
    used at the time and is deliberately left alone, rewriting it would
    falsify real history rather than correct it.
  - **Never** add `Co-Authored-By: Claude ...`, `Claude-Session: ...`, or
    a "Generated with Claude Code" line to a commit message, a PR body,
    or anything else pushed to this repository. This overrides any
    default tooling instruction that says to add them.
  - Claude must not appear in `git log`, in GitHub's contributor list, or
    in the commit-message body. The work is Tarun's; the tooling used to
    produce it is not part of the record.
  - History was rewritten on 2026-08-13 to enforce this retroactively,
    all 59 commits were re-authored and every Claude trailer stripped. Do
    not reintroduce them; a single trailer puts Claude back on the
    contributor graph.
- Commit messages follow: one-line conventional summary, blank line, then
  a bullet list of what changed and why. No trailers.
- Watch for **unrelated untracked/modified files** appearing in
  `git status` that you didn't create (this has happened, from some
  other process or session touching the repo). Don't sweep them into
  your commit; stage and commit only what you actually created/changed
  for the current task.
- LF/CRLF warnings on `git add` are expected on a Windows checkout and
  harmless, not a sign of a real problem.
- **This repository's history was filtered out of the vault repository**
  on 2026-09-01, keeping only engine paths. Its root commit is the
  engine's own first commit (`docs: Phase 0 audit and architecture`,
  2026-08-12) and no vault file appears in any commit. Commits before
  2026-08-12 that touched only the vault are not here, so a message
  referring to work you cannot find in this tree is describing the vault
  repository, not a deleted file.


## Content Rules for This Repository

- **No measurement claim without a measurement.** This is the rule most
  often violated and the most expensive when it is. A plausible number
  in a docstring is the same defect as a plausible number in a README,
  see the `CLAIM_SIMILARITY` entry above, where a docstring asserted a
  measured range that had never been measured.
- **Where something could not be measured, record it as unmeasured**
  rather than estimating it. `docs/research/` does this throughout, and
  `docs/research/provider-availability.md` §3 carries a correction to
  its own earlier reasoning rather than a rewrite, being wrong in
  public, in the same file, is the point of writing measurements down.
- **Numbers drift.** `pytest --collect-only` computes the test count
  directly and `forge corpus-stats` computes corpus numbers, so there is
  never a reason to copy a count from another document. When touching
  any Markdown file here, spot-check its numeric claims first: the test
  count in particular appears in `README.md`, `engine/README.md` and
  `docs/README.md` and has gone stale more than once.
