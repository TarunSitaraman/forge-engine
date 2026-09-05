# Forge: Current State Audit

*Factual audit of the Forge repository as it exists at commit `bb88c35`, performed 2026-08-12, before any Knowledge-OS implementation work. Every number here was measured against the filesystem, not copied from an existing doc.*

---

## 1. Executive Summary

**Forge today is a hand-authored Markdown corpus, not a software system.**

There is no application code in this repository. No Python, no
TypeScript, no `package.json`, no `pyproject.toml`, no Dockerfile, no
CI configuration, no tests. The repository contains 620 Markdown files
and one `.gitignore`, 621 files total, ~48,700 lines.

The "AI-maintained" property that Forge's philosophy claims is real but
**entirely human-triggered and human-supervised**: it is implemented by
Claude Code sessions editing files by hand, steered by `CLAUDE.md`, a
79-prompt library, and a set of written standards. There is no runtime,
no scheduler, no ingestion pipeline, no index, and no automated
enforcement of any rule the documentation states.

This has one very important consequence for the Knowledge-OS work:

> Forge is not a system to be refactored. It is a **corpus plus an
> informally-specified set of invariants**. The Knowledge OS is a
> greenfield engine build whose first and most valuable input is this
> corpus, and whose correctness target is already written down in
> `DSA/Forge Engineering Constitution.md`, `CONVENTIONS.md`, and
> `DSA/Validation Checklist.md`.

The engine's job is to *mechanically enforce and scale* the invariants
that 620 files currently maintain by discipline alone, and the audit
below shows that discipline has already measurably drifted.

---

## 2. Repository Inventory (measured)

### 2.1 Size and shape

| Metric | Value |
|---|---|
| Total files | 621 (620 `.md` + `.gitignore`) |
| Total Markdown lines | 48,737 |
| Application code files | **0** |
| Build/CI/dependency manifests | **0** |
| Tests | **0** |
| Largest single file | `Projects/smartresq-project-handoff.md` (728 lines) |
| Commits on `main` | 20+ (all `docs:`, no code has ever been committed) |

### 2.2 Content distribution

| Top-level folder | Files | Lines | Role |
|---|---:|---:|---|
| `DSA/` | 369 | 26,996 | Flagship. Algorithms/interview knowledge base |
| `Technologies/` | 140 | 10,409 | Prompt library, playbooks, templates, tech docs |
| `Projects/` | 49 | 6,526 | Five project knowledge packs |
| *(root)* | 13 | 2,144 | Governance docs + session summaries |
| `Courses/` | 21 | 1,420 | Two course trackers |
| `Career/` | 11 | 688 | Career tooling |
| `Resources/` | 12 | 415 | Curated external links |
| `Systems/` | 1 | 42 | **Legacy stub** (see §6.1) |
| `Reference/`, `Inbox/`, `Archive/`, `.obsidian-config/` | 1 each | 15-34 | Index-only, effectively empty |

### 2.3 Verified sub-counts

Corrected against the filesystem; several published counts were stale
(see §6.4).

| Component | Actual |
|---|---:|
| DSA patterns | 32 |
| DSA algorithms | 30 |
| DSA data structures | 18 |
| DSA detailed problem pages | **85** (README claimed "70+") |
| DSA representative-problem indices | 32 |
| DSA interview guides | 32 |
| DSA cheat sheets | 39 |
| DSA mistake pages | 12 |
| Python templates | 30 |
| Prompt library prompts | 79 across 28 categories |
| Playbooks | 18 |
| Document templates | **25** (`ROADMAP.md` claimed 21) |
| Technology reference docs | 11 |
| Mermaid diagrams | 37 files |
| Python code blocks | 240 |

---

## 3. Current Architecture

There is no software architecture. What exists is an **information
architecture plus a manual operating procedure**. Both are documented
well enough to be treated as a specification.

### 3.1 The actual "system"

```
                    HUMAN (author/operator)
                            |
                            | opens a session, pastes a prompt
                            v
              CLAUDE CODE  (the entire "runtime")
                            |
        steered by: CLAUDE.md, CONVENTIONS.md, WORKFLOW.md,
        Forge Engineering Constitution, Documentation Standards,
        Validation Checklist, 79 Prompt-Library prompts
                            |
                            v
                    MARKDOWN FILES  (state)
                            |
                +-----------+-----------+
                |                       |
           GIT (history,           OBSIDIAN
           audit trail)         (read/navigate)
```

Every arrow is manual. Nothing runs on a schedule, on commit, or on
file change.

### 3.2 Storage model

- **Format:** plain Markdown, UTF-8, one concept per file.
- **Source of truth:** the Git working tree. `WORKFLOW.md` is explicit
  that "the repository is the source of truth... not backlinks, plugin
  metadata, or a database."
- **Indexing:** hand-maintained `_index.md` files per folder, plus
  hand-maintained index pages under `DSA/00_Index/`. These are the only
  "index" that exists and they drift whenever content is added without
  updating them.
- **Metadata:** YAML frontmatter, applied inconsistently (§3.3).
- **Retrieval today:** Obsidian quick-switcher, Obsidian backlinks,
  `grep`, and GitHub search. There is no semantic search of any kind.

### 3.3 Metadata layer: measured coverage

Frontmatter is present on **362 of 620 files (58%)**, and is almost
entirely confined to one folder:

| Folder | Files with frontmatter |
|---|---:|
| `DSA/` | 358 |
| root session-summary files | 4 |
| **Everything else** (`Technologies/`, `Projects/`, `Courses/`, `Career/`, `Resources/`) | **0** |

Keys in use: `type` (358), `status` (358), `tags` (358), `canonical`
(358), `related` (283), `difficulty` (77), plus stragglers (`title`,
`date`, `session`, `author`).

So: the 258 files outside `DSA/`, including all 11 authoritative
technology docs, all 49 project-pack files, and all 79 prompts: carry
**no machine-readable metadata at all**. Their type, status, and
relationships exist only in prose and folder position.

### 3.4 Link graph

Measured after stripping fenced code blocks and inline code (this
matters, see §6.3):

| Metric | Value |
|---|---:|
| Wikilinks (`[[...]]`) | 4,131 |
| Distinct wikilink targets | 516 |
| **Unresolved targets** | **145 distinct, 289 occurrences** |
| Relative Markdown links (`](x.md)`) | 385 |
| Broken relative links | 0 |

A real link graph exists and is dense (~6.7 links per file). Relative
Markdown links are clean. Wikilinks are **~7% broken**, and nothing
detects that. Obsidian's "unresolved links" panel is the only check, and
`WORKFLOW.md` relies on a human running it weekly.

> **Method note.** A first pass reported two broken relative links in
> `README.md`. That was a false positive: the links are URL-encoded
> (`DSA/00_Index/DSA%20Home.md`), which is correct for GitHub, and the
> checker compared the encoded string against the filesystem without
> decoding it. **A link checker must URL-decode before resolving**,
> recorded alongside §6.3 as a second parser requirement discovered by
> getting it wrong here first.

### 3.5 AI functionality that exists today

Three artifacts encode AI behavior, all as *prose instructions for a
human-invoked model*: none are executable:

1. **`DSA/AI Ingestion Workflow.md`.** The closest thing to a
   specification of the target pipeline. It already names the exact
   stages the Knowledge OS needs: parse → classify → create/update
   *exactly one* page → link without duplicating → validate. It also
   already specifies conflict handling ("prefer existing canonical
   pages... keep the older canonical page and merge links into it") and
   duplicate detection ("search by title, platform slug, pattern, and
   distinctive constraints before creating a new page"). **This is a
   hand-written draft of the ingestion graph and should be treated as
   requirements input, not deleted.**
2. **`DSA/05_Templates/HackerRank Ingestion Prompt.md`.** A concrete
   single-source ingestion prompt.
3. **`Technologies/Prompt-Library/`**, 79 prompts across 28
   categories. These are operating procedures for a human+LLM pair,
   not system prompts for an application. Several
   (`Research/literature-review-synthesis.md`,
   `Research/paper-critical-reading.md`,
   `RAG/chunking-strategy-design.md`,
   `AI/hallucination-mitigation-review.md`) encode reasoning the engine
   will eventually need and are worth mining when writing node prompts.

**There is no LLM integration, no API client, no model configuration,
no embeddings, and no provider abstraction anywhere in the repository.**

### 3.6 Obsidian integration

- `.obsidian-config/README.md` is documentation *about* how to
  configure Obsidian, a reference copy of intended settings. The live
  `.obsidian/` directory is gitignored.
- No Obsidian plugin exists. No community plugins are used, by policy.
- Integration surface is therefore: wikilinks, folder layout, YAML
  frontmatter, and Mermaid: all vanilla Markdown features.
- Policy constraint worth carrying forward:
  `.obsidian-config/README.md` forbids "anything that stores state
  Markdown can't represent." **Any Knowledge-OS write-back must remain
  meaningful when read as plain Markdown**, or it breaks a stated
  principle.

### 3.7 Existing integrations

None. No external services, no APIs, no webhooks, no CI. `origin` on
GitHub is the only external dependency.

---

## 4. What the Corpus Already Gets Right

These are the load-bearing assets. They are why this is an *evolution*
and not a rewrite.

1. **A written, coherent specification of the invariants.** The
   Constitution, `CONVENTIONS.md`, `Documentation Standards.md`,
   `Repository Linking Architecture.md`, and `Validation Checklist.md`
   collectively define: one canonical home per concept, typed
   relationships, mandatory metadata, and a 12-point quality gate. The
   Knowledge OS does not need to invent its quality model. It needs to
   *execute* this one.
2. **A dense, typed link graph.** 4,131 links, and
   `Documentation Standards.md` already says links should express
   relationships ("uses, depends on, contrasts with, common mistake,
   representative problem, template implementation"). That is a
   relationship vocabulary, already drafted.
3. **A repeatable content structure.** `Technologies/Docs/` files all
   follow Overview → Mental Model → Architecture → Core Concepts →
   Common Workflows → Common Mistakes → Best Practices → Cheatsheet →
   Interview Questions → Further Reading. Knowledge packs follow
   `_index.md` + numbered docs. **Predictable headings make
   deterministic, structure-aware chunking possible**: a significant
   head start over ingesting arbitrary Markdown.
4. **Git as an existing provenance and history substrate.** Every
   change is already attributed and timestamped. Principle 12 (preserve
   history) is partially satisfied for free.
5. **A high-quality, domain-dense corpus.** ~48.7k lines of curated
   engineering knowledge is a genuinely good evaluation set for
   retrieval and concept extraction, far better than synthetic
   fixtures.
6. **An explicit anti-bloat culture.** README's "What Forge is not" and
   ROADMAP's "explicitly out of scope" sections are the same discipline
   the product-positioning doc now applies to the engine.

---

## 5. Current Limitations (relative to the Knowledge-OS goal)

| # | Limitation | Consequence |
|---|---|---|
| L1 | No semantic retrieval | Can't answer "what do I believe about X"; only exact-string grep |
| L2 | No provenance model | A claim in `rag.md` cannot be traced to any source. Nothing distinguishes source fact from model inference, the exact failure Principle 10 forbids |
| L3 | No change analysis | New information is merged by a human deciding, in the moment, what to overwrite. Superseded understanding is recoverable only by reading Git diffs |
| L4 | No contradiction handling | Two docs can disagree indefinitely; nothing detects it |
| L5 | No non-Markdown ingestion | PDFs, papers, repos, and web pages cannot enter Forge except by a human reading and rewriting them |
| L6 | Enforcement is entirely manual | Every rule in the Constitution is advisory. Measured drift: 145 broken links, 42% of files missing metadata, all 283 `related:` fields malformed |
| L7 | Indexes drift | `_index.md` files and README counts go stale silently (§6.4) |
| L8 | No confidence or uncertainty representation | Everything reads as equally certain. Principle 12 unsatisfied |
| L9 | Scaling ceiling | The pattern that produced 620 files does not survive 10,000 heterogeneous sources, a human is in the loop for every write |

---

## 6. Technical Debt (concrete, measured)

### 6.1 `Systems/` is an orphaned legacy folder: **highest-signal debt**

`Systems/` contains exactly one file, `_index.md`, which describes eight
subfolders (`Prompt-Library/`, `Playbooks/`, `Templates/`,
`Competitive-Programming/`, `Career/`, `Project-System/`, `Docs/`,
`Resources/`) that **no longer live there**. They were moved to
`Technologies/`, `Courses/`, `Career/`, and `Resources/`. Every relative
link in that file is broken.

This is a fossil of a prior reorganization. `CLAUDE.md` records that
broken `../../Systems/Docs/...` links were already repaired *elsewhere*
in a previous session, but the source of those links. This stub: was
never removed or rewritten.

**Recommendation:** delete `Systems/` or rewrite `_index.md` as an
explicit tombstone pointing at the new locations. Not done in this pass. It is content surgery outside the audit's scope, and the audit's job
is to report it.

### 6.2 All 283 `related:` frontmatter fields are malformed

This is the single largest machine-readability defect, and it was
invisible until parsed.

The intended syntax was a list of wikilinks. What was written is not
valid for either purpose:

```yaml
# 68 files — HARD YAML PARSE FAILURE (ParserError)
related: [[Pattern Index]], [[Template Index]]

# 215 files — parses, but WRONG: yields nested lists, not links
related: [[[DFS]], [[BFS]], [[Graph Traversal]]]
#  -> [[['DFS']], [['BFS']], [['Graph Traversal']]]
```

Of 362 frontmatter blocks, **68 fail `yaml.safe_load` outright**: a
strict ingester would reject those files entirely. The other 215 parse
into nested string lists that are not usable as relationships without
custom repair.

Meanwhile `Documentation Standards.md` explicitly promises "Dataview
Compatibility, keep metadata fields simple scalars or lists." That
promise is broken in every file that makes it.

**Recommendation:** a deterministic one-shot migration script
(Phase 1) normalizing to a valid, quoted form:

```yaml
related: ["Pattern Index", "Template Index"]
```

This is mechanical, lossless, and reversible. It should be the first
code committed to the repository, because *nothing downstream can trust
frontmatter until it lands.*

### 6.3 Wikilink parsing requires code-fence stripping

240 Python code blocks contain matrix/list literals such as
`[[1,2],[3,4]]`. A naive `\[\[...\]\]` regex over this corpus produces
hundreds of false-positive "links" (the first pass of this audit did
exactly that). **Any parser must strip fenced and inline code before
extracting links.** Recorded here so the mistake isn't repeated in
implementation.

### 6.4 Documentation drift in published counts

| Claim | Location | Actual |
|---|---|---|
| "70+" detailed problems | `README.md` | 85 |
| "21 templates" | `ROADMAP.md` | 25 |
| "10 authoritative ... manuals" | `ROADMAP.md` | 11 |

`CLAUDE.md` already flags this as a recurring pattern. It recurs
because nothing checks it. This is the clearest possible argument for
a derived, generated index: **any number a human maintains by hand in
this repo eventually goes stale.**

### 6.5 Two competing convention systems

`CONVENTIONS.md` (repo-wide) and `DSA/Documentation Standards.md`
(DSA-local) contradict each other:

| Rule | `CONVENTIONS.md` | `DSA/Documentation Standards.md` |
|---|---|---|
| Filenames | `kebab-case.md` | Title Case, e.g. `Binary Search.md` |
| Tags | namespaced `#status/`, `#type/`, `#stack/`, max 3 | `dsa/pattern`, `dsa/algorithm`, ... |
| Frontmatter | minimal, "only when it carries real metadata" | mandatory on every page |

Measured filename reality: 329 Title-Case-with-spaces, 214 kebab-case,
30 SHOUT_CASE, 47 other. Both conventions are in active use.

This is defensible as a deliberate DSA-local dialect, but it is
**nowhere stated to be deliberate**. The knowledge model must therefore
treat `type`/`tags` vocabularies as *per-namespace*, not global, or
normalize them explicitly. Flagged as an open decision (§8).

### 6.5.1 Filenames with spaces

329 files contain spaces. Harmless in Obsidian; a persistent source of
bugs in shell tooling (this audit hit it immediately). Any scripts must
be space-safe by construction, no unquoted `for f in $(find ...)`.

### 6.6 Root-directory clutter

Six root files are historical session artifacts, not knowledge:
`DSA_IMPLEMENTATION_PLAN.md`, `FORGE_COMPLETION_STATUS.md`,
`FORGE_SESSION_2_SUMMARY.md`, `FORGE_SESSION_3_FINAL_SUMMARY.md`,
`IMPROVEMENTS_SUMMARY.md`, `GITHUB_SETUP_CHECKLIST.md` (~1,400 lines).
They describe work already done. They belong in `Archive/`, which is
currently empty. Low priority; noted for completeness.

### 6.7 Empty structural folders

`Inbox/`, `Reference/`, and `Archive/` contain only `_index.md`. The
capture→file workflow described in `START_HERE.md` and `WORKFLOW.md` is
documented but unused in practice. Worth knowing before designing an
ingestion inbox: **the manual inbox pattern has already been tried here
and did not stick.** That is an argument for automated routing rather
than a human triage queue.

---

## 7. Preservation Points (must survive)

Ordered by how damaging it would be to break them.

| # | Preserve | Why |
|---|---|---|
| P1 | **Every existing `.md` file, at its current path** | Non-negotiable per the brief. 4,131 internal links and all external GitHub links depend on paths |
| P2 | **Markdown as the human source of truth** | `WORKFLOW.md`'s core promise. The graph/vector stores must be *derived and rebuildable*, never authoritative, see ADR-001 |
| P3 | **Obsidian-vanilla compatibility** | No plugin-only constructs. Content must stay readable in any editor |
| P4 | **Git as the history substrate** | Already provides attribution/timestamps; don't duplicate it in a database |
| P5 | **The canonical-home rule** | The corpus's defining invariant and the engine's core dedup requirement |
| P6 | **`CONVENTIONS.md` + Constitution + Validation Checklist** | The de-facto quality spec. Encode as executable checks; do not discard |
| P7 | **The `AI Ingestion Workflow` stage list** | A hand-derived draft of the ingestion graph; converges with the LangGraph design |
| P8 | **Structural heading conventions** | Enables deterministic structure-aware chunking |
| P9 | **The Prompt-Library's reasoning** | Mine for node prompts rather than writing new ones from scratch |

---

## 8. Migration Points (must change), with risk

| # | Change | Risk | Mitigation |
|---|---|---|---|
| M1 | Repair all 283 `related:` fields | Low | Deterministic script + full diff review; Git is the undo |
| M2 | Extend frontmatter to the 258 unmetadata'd files | **Medium** | Additive only; never rewrite prose. Generated fields must be namespaced (e.g. `forge_*`) so hand-authored values are never clobbered |
| M3 | Repair 145 unresolved wikilinks | Medium | Only auto-fix exact-normalization matches; everything else is a report for a human, since a broken link may mean "page not written yet" |
| M4 | Introduce derived stores (vector/graph/relational) | Low | Fully rebuildable from Markdown; `.gitignore`d; deleting them loses nothing |
| M5 | Introduce engine source code into the repo | **High** | Adding `engine/` and `docs/` pollutes the Obsidian vault root. **Open decision, see below** |
| M6 | Write generated knowledge back into the vault | **Highest** | Directly risks Principles 10/11. Must be segregated, provenance-stamped, and human-approved. **Open decision** |
| M7 | Retire or tombstone `Systems/` | Low | One stub file, links already broken |
| M8 | Reconcile the two convention systems | Medium | Prefer per-namespace vocabularies over a forced global rewrite |

### Open decisions requiring human approval

**D1, Repository layout (blocks all implementation).** The repo root
*is* the Obsidian vault. Adding `engine/`, `tests/`, and `docs/` at the
root means Obsidian indexes engineering docs as vault notes and they
appear in graph view and quick-switcher. Options:

- **(a) Monorepo + Obsidian exclusion filters.** Least disruptive to
  existing paths; requires vault config that is gitignored, so it must
  be documented in `.obsidian-config/`. *Recommended.*
- **(b) Move the vault under `vault/`.** Cleanest separation, but
  rewrites all 621 paths and breaks every external link. Violates P1's
  spirit.
- **(c) Separate engine repository.** Cleanest of all, but splits the
  corpus from the code that maintains it and complicates local-first
  setup.

*This document introduces `docs/` under option (a) as the minimum
needed to deliver the required documentation. That choice is reversible
and does not commit the engine's location.*

**D2, Write-back policy.** Does the engine ever write into the vault,
or does generated knowledge live only in derived stores and surface
through read-only interfaces? This is the highest-stakes decision in
the project and is deliberately left open. See ADR-001.

**D3, Convention reconciliation.** Normalize DSA to repo-wide
conventions, formally bless the DSA dialect, or make the model
namespace-aware? Recommended: namespace-aware, because it costs no
content churn.

**D4, Scope of the corpus as a knowledge source.** Is the existing
vault (a) an ingestion source treated like any other, (b) privileged
"user assertion" ground truth, or (c) both, per folder? Affects
provenance tiering directly. Recommended: (c), with `Technologies/Docs/`
and `Projects/` treated as user assertions and `Resources/` treated as
pointers to external sources.

---

## 9. What This Audit Does *Not* Establish

Stated explicitly rather than guessed:

- **Whether the corpus's technical claims are accurate.** This audit
  measured structure, not correctness. The four externally-researched
  project packs flag their own confirmed-vs-unconfirmed findings; that
  self-reporting was not independently verified here.
- **Actual usage patterns.** Nothing in the repository records which
  files are read most, or whether the daily loop in `START_HERE.md` is
  followed. The empty `Inbox/` suggests it is not, but that is
  inference.
- **Hardware available for local inference.** Model-size choices in
  `technology-decisions.md` are therefore given as tiers, not a single
  pick.
- **Whether the user wants the vault to remain human-writable** after
  the engine exists. Assumed yes (P2), but it is an assumption.

---

## Related

- [Target architecture](./target-architecture.md)
- [Technology decisions](./technology-decisions.md)
- [Canonical knowledge model](../knowledge-model/canonical-model.md)
- [ADR-001: Forge as a Knowledge OS](../decisions/001-forge-knowledge-os.md)
- [Roadmap](../roadmap.md)
