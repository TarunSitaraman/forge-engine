# ADR-001: Evolve Forge into a Knowledge Operating System

- **Status:** **Accepted**, D1 and D2 approved 2026-08-12; Phase 1 implemented against this ADR. D3-D5 remain open.
- **Date:** 2026-08-12
- **Deciders:** repository owner (pending)
- **Context commit:** `bb88c35`
- **Supersedes:** nothing
- **Related:** [current-state audit](../architecture/forge-current-state.md), [target architecture](../architecture/target-architecture.md), [canonical model](../knowledge-model/canonical-model.md), [roadmap](../roadmap.md)

---

## 1. Context

Forge is a 620-file, ~48,700-line curated Markdown knowledge base with
**no application code.** No manifests, no tests, no CI (audit §1). Its
"AI-maintained" property is real but entirely human-triggered: Claude
Code sessions editing files by hand, steered by written standards and a
79-prompt library.

The proposal is to build an engine that maintains an evolving,
evidence-traceable knowledge model, with Obsidian, a web UI, a CLI, and
MCP as interchangeable interfaces.

### What makes this decision non-obvious

The corpus already encodes the right invariants. One canonical home per
concept, typed relationships, mandatory metadata, a 12-point validation
checklist. And the audit measured that a disciplined author, following
their own written rules, still accumulated:

- **145 unresolved wikilinks** (~7% of the link graph),
- **42% of files with no machine-readable metadata**,
- **283 malformed `related:` fields**: 68 of which fail YAML parsing
  outright,
- **published counts stale in three separate files**.

The rules were not wrong. They were unenforceable by hand at 620 files.
That is the actual argument for this project, and it is stronger than
"add AI to notes": **Forge already specified the invariants it cannot
mechanically maintain.**

---

## 2. Decision

Build the Forge Knowledge OS as a new engine **alongside** the existing
corpus, under these binding commitments:

### 2.1 Markdown remains the source of truth

The vault is authoritative for human-authored knowledge. Relational,
vector, and graph stores are **derived and rebuildable**. Deleting all
derived stores and re-running ingestion must reproduce them.

*Exception:* original source bytes (PDFs, fetched pages) are retained
content-addressed in a blob store. They cannot be regenerated.

### 2.2 The existing corpus is preserved absolutely

No file is deleted, moved, or mass-rewritten. All 621 paths remain
stable. The corpus becomes ingestion source #1 and the primary
evaluation dataset.

### 2.3 Provenance is structural

Five tiers (`SOURCE_FACT`, `EXTRACTED_CLAIM`, `MODEL_INFERENCE`,
`SYNTHESIS`, `USER_ASSERTION`), enforced by the **provenance floor
rule**: a derived object's tier can never be stronger than the weakest
tier among its inputs. Enforced at write time, deterministically.

### 2.4 Evolution is non-destructive

Supersession, never overwrite. Both states retained, with the evidence
that caused the change, in an append-only `Revision` log that exists
from the first write.

### 2.5 Determinism by default

Hashing, parsing, chunking, traversal, dedup, and lineage are ordinary
software. The LLM is called only for genuinely semantic tasks, behind a
provider abstraction, with `MockProvider` as the CI default.

### 2.6 Local-first, free-first

Ollama is the default and the CI target. No paid provider on any core
path. The full test suite runs offline with no model.

### 2.7 LangGraph, scoped

Four workflows only, ingestion/evolution, contradiction resolution,
re-synthesis, corpus backfill. Nodes named for operations, never
personas. Retrieval stays a plain composed pipeline.

---

## 3. Options considered

| Option | Verdict |
|---|---|
| **A. Do nothing**, keep the manual workflow | Rejected. Measured drift shows the approach has already exceeded what discipline can hold, and it cannot ingest PDFs, papers, or repos at all |
| **B. Obsidian plugins / Dataview** | Rejected. Cannot express provenance tiers, confidence, or contradiction; violates the existing minimal-plugin and Markdown-only policies; the root roadmap already flags Dataview as unadopted |
| **C. Off-the-shelf RAG over the vault** | Rejected. Delivers retrieval, not evolution. Fails Principles 3, 10, 11, 12, and is precisely the "generic RAG chatbot" boundary |
| **D. Rewrite the vault into a database-native app** | Rejected. Destroys the Git-first, Obsidian-compatible, Markdown-only guarantees; violates the brief's preservation requirement |
| **E. Engine alongside the corpus, Markdown authoritative** | **Chosen** |

Option E is the only one that preserves every existing guarantee while
adding capabilities none of the others can.

---

## 4. Consequences

### Positive

- The vault keeps working exactly as today with the engine switched off.
- The user's knowledge survives Forge's deletion, plain Markdown, in Git.
- Schema iteration is cheap: change the model, rebuild the index.
- The corpus provides ground truth for evaluation (~500 named concepts,
  ~4,100 human-authored relationships): an unusually strong starting
  position.
- Invariants the corpus states become mechanically enforceable.

### Negative / accepted costs

- **Significant new complexity** in a repository that currently has none.
  Accepted: it is the cost of the capability.
- **Dual representation.** Markdown and derived stores can diverge; the
  rebuild path is the reconciliation mechanism.
- **Local model quality is unproven** for contradiction detection, the
  project's largest technical unknown (technology decisions §4.4).
  Mitigated by spiking it in Phase 1.
- **Root-directory pollution.** `docs/` and any `engine/` appear in the
  Obsidian vault, see §6.1.
- **Derived state that Markdown cannot express is at risk on rebuild.**
  Contradictions, confidence, and revision history are exactly that.
  This is the tension that makes §6.2 unavoidable.

### Neutral

- Two roadmaps now exist (vault content, engine). Deliberate; they are
  separate tracks and merging them would conflate unrelated work.

---

## 5. Compliance with the stated principles

| # | Principle | How this decision satisfies it |
|---|---|---|
| 1 | Research ≠ document collection | Concepts/claims/relationships are first-class; documents are evidence |
| 2 | Knowledge ≠ stored notes | Ingestion's completion criterion is a model diff |
| 3 | New info can change the model | Change analysis + supersession (§2.4) |
| 4 | Traceable to evidence | `Span` as the provenance atom |
| 5 | Minimize manual organization | Automated concept resolution and relationship discovery |
| 6 | LLM is a component | Provider abstraction; `MockProvider` in CI |
| 7 | Deterministic work stays deterministic | §2.5; four of seven core components need no LLM |
| 8 | LLMs used selectively | Hash short-circuit, span caching, candidate narrowing |
| 9 | Local-first, free-first | §2.6 |
| 10 | Claims trace to evidence | Provenance floor rule; unevidenced non-user claims fail validation |
| 11 | No silent overwrite | §2.4; approval gates on merges and high-severity contradictions |
| 12 | Preserve uncertainty and history | `Contradiction` (incl. `accepted_tension`), confidence with recorded origin, `Revision` log |

---

## 6. Decisions

D1 and D2 were **approved on 2026-08-12** and are recorded below as resolved.
D3-D5 remain open.

### 6.1 D1: Repository layout, **APPROVED: option (a), monorepo**

> The repository root remains the Obsidian vault. Existing Markdown paths are
> immutable unless an explicit migration is approved. The engine lives
> alongside the corpus in `engine/`, `tests/`, `scripts/`, `docs/`, `.forge/`.

Implemented as approved. All 621 original paths are unchanged;
`test_indexing_does_not_modify_the_vault` enforces it against `git status`.

`docker/` was **not** created: Phase 1 requires no containers (SQLite + stdlib
only), and an empty directory documenting a future need is the kind of
placeholder that goes stale. It arrives at Phase 4 with the first service.

*Original options, retained for the record:*

### 6.1.1 D1 options as considered *(blocked all implementation)*

The repo root **is** the Obsidian vault. Adding `engine/` and `docs/`
means Obsidian indexes engineering docs as vault notes. They appear in
quick-switcher and graph view.

| Option | Trade-off |
|---|---|
| **(a) Monorepo + Obsidian exclusion filters**, *recommended* | Least disruptive; all 621 paths stable. Requires vault config, which is gitignored, so it must be documented in `.obsidian-config/` |
| (b) Move vault under `vault/` | Cleanest separation; rewrites every path and breaks external links. Violates preservation point P1 |
| (c) Separate engine repository | Cleanest of all; splits the corpus from the code that maintains it and complicates local-first setup |

### 6.2 D2: Write-back policy, **APPROVED: segregated write-back**

> The corpus remains the canonical human-readable source of truth. The engine
> may read it freely and must not silently modify it. AI-generated changes
> exist first as proposed/derived state. A future human-approval workflow may
> apply approved changes. **No automatic in-place enrichment in Phase 1.**

Implemented as approved:

- No engine code path writes to a `.md` file. The only directory written is
  `.forge/`.
- Frontmatter repairs are generated, applied *in memory*, re-parsed to confirm
  validity, and reported as `verified: true` proposals. 283 files have verified
  proposals; **zero have been applied.**
- Ambiguous links produce candidates, never a chosen target.
- `test_cli_never_writes_to_the_vault` byte-compares every Markdown file before
  and after running all read commands.

*Original options, retained for the record:*

### 6.2.1 D2 options as considered

Does the engine ever write into the vault?

| Option | Trade-off |
|---|---|
| **(a) Read-only**, generated knowledge lives only in derived stores | Zero risk to existing content; but everything the engine learns is invisible in Obsidian, and lost on rebuild |
| **(b) Segregated write-back**, a dedicated directory, provenance-stamped frontmatter, generated files never mixed with hand-authored ones | Knowledge is visible and durable in Markdown; risk is contained by segregation |
| **(c) In-place enrichment**, engine adds frontmatter/links to existing files | **Highest value, highest risk.** Directly threatens Principles 10 and 11 |

*Recommendation: **(b)**, with (c) only for additive, namespaced
frontmatter fields (`forge_*`) that can never clobber hand-authored
values, and only after Phase 7.*

This is deliberately left open because it determines whether
`Concept.vault_path` is read-only or bidirectional, and because it is
the decision most likely to be regretted if made hastily.

### 6.3 D3: Convention reconciliation

`CONVENTIONS.md` and `DSA/Documentation Standards.md` contradict each
other on filenames, tags, and frontmatter (audit §6.5), and both are in
active use. *Recommendation: namespace-aware vocabularies, costs no
content churn.*

### 6.4 D4: Corpus provenance tier

Is the existing vault an ordinary ingestion source, privileged
`USER_ASSERTION` ground truth, or both by folder? *Recommendation: both
by folder, `Technologies/Docs/` and `Projects/` as user assertions,
`Resources/` as pointers to external sources to ingest.*

### 6.5 D5: Graph store

Relational adjacency vs Neo4j. Lowest-confidence recommendation in the
technology decisions (§5.3). *Recommendation: defer to a measurement at
the Phase-4 gate, adopt Neo4j only if queries routinely exceed 3 hops
or need path-finding.*

Phase 1 deferred it as recommended: storage sits behind protocols in
`forge/storage/base.py` with a SQLite implementation. No graph or vector
database was introduced.

### 6.6 D6: Truncated-wikilink frontmatter *(new, raised by Phase 1)*

Implementation found a **third** malformed-frontmatter shape the Phase 0 audit
did not characterize: 18 files whose final wikilink is truncated to one closing
bracket (`related: [[A]], [[B]`). Diagnosed as `FM008` with verified repairs,
none applied.

No decision is required to proceed. It is handled, but it is recorded here
because it revises the audit's "two defect shapes" finding, and because the
same authoring slip may exist in files that happen to still parse.

---

## 7. What this ADR does not decide

- Claim granularity (canonical model Q1), needs tuning against the
  corpus.
- Concept identity thresholds (Q2).
- Confidence arithmetic (Q4), recommended: none in Phase 1; store and
  display evidence counts rather than fabricate a posterior.
- Contradiction sensitivity (Q6), needs a labeled evaluation set first.
- Any interface design beyond the layering rule.

---

## 8. Revisit criteria

Reopen this ADR if:

- the Phase-1 spike shows local models cannot perform contradiction
  detection acceptably at any threshold (would force reconsidering
  Principle 9's scope, or narrowing the evolution engine's ambitions);
- the rebuild-from-Markdown guarantee proves impossible to maintain once
  contradictions and confidence exist (would force D2 toward option (b)
  or (c) as a *requirement*, not a choice);
- the user decides the vault should become read-only to humans, which
  would invert the source-of-truth decision in §2.1.
