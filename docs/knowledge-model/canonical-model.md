# Forge — Canonical Knowledge Model

*Proposed entity/relationship model for the Forge Knowledge OS. This is a proposal for review, not a frozen schema — §9 lists what still needs a human decision.*

**Status:** proposed · **Supersedes:** nothing · **Blocks:** all Phase-1 implementation

---

## 1. Design rules

Six rules that decide every modeling question below. When in doubt,
apply these rather than adding an entity.

| # | Rule | Consequence |
|---|---|---|
| R1 | **Provenance is structural, not a field** | Anything assertable carries an unforgeable link to what produced it |
| R2 | **Evidence is a relationship, not a thing** | `Evidence` is a reified edge from a claim to a source span |
| R3 | **Reify an edge only when it needs its own provenance, confidence, or history** | Claim↔Claim edges are reified; `MENTIONS` is not |
| R4 | **Prefer one entity with a `kind` discriminator over many near-identical entities** | `Technology`, `Project`, `Person` are `Concept` kinds, not tables |
| R5 | **Nothing is destroyed; state changes are events** | `SUPERSEDES`, never `UPDATE` |
| R6 | **Every entity records how it was derived: deterministic or model** | Directly enforces Principle 7 and makes drift measurable |

The brief's 19-entity list is the requirements input. Applying R3 and R4
reduces it to **9 entities for Phase 1**, with the rest either derived,
reified as edges, or deferred with their semantics reserved. Nothing in
the brief's list is dropped — §7 maps each one.

---

## 2. Entity core (Phase 1)

### 2.1 `Source`

The origin of information. Stable identity, independent of parsing.

| Field | Type | Notes |
|---|---|---|
| `id` | ULID | |
| `kind` | enum | `pdf`, `markdown`, `repo`, `web`, `code`, `dataset`, `manual` |
| `locator` | string | path, URL, or repo ref — canonicalized |
| `content_hash` | sha256 | **deterministic**; the dedup and change-detection key |
| `title`, `authors`, `published_at` | — | extracted deterministically where the format allows |
| `ingested_at`, `last_seen_at` | timestamp | |
| `trust_tier` | enum | see §4.3 |

> Re-ingesting an unchanged `content_hash` must be a **no-op that costs
> zero LLM calls**. This single rule is what makes incremental
> re-ingestion of a 620-file vault viable.

### 2.2 `Document`

A parsed rendering of a `Source` at a point in time. Separate from
`Source` because a source can be re-parsed (better extractor, new
version) without losing its identity or its prior parse.

`id`, `source_id`, `version` (int), `parser`, `parser_version`,
`structure` (heading tree), `parsed_at`.

### 2.3 `Span`

A located region of a `Document`. **The atom of provenance** — the
thing Principle 4's "which section/page/chunk" resolves to.

`id`, `document_id`, `locator` (page/char-offset/heading-path/line
range), `text`, `token_count`, `embedding_ref`, `chunk_strategy`.

Spans are produced **deterministically** (structure-aware chunking on
heading boundaries). No LLM participates in chunking.

*This entity is not in the brief's list. It is added because without it,
"traceable to evidence" degrades to document-level attribution, which is
not traceability.*

### 2.4 `Concept`

A durable, named idea. The primary node of the graph, and the direct
descendant of the existing vault's "one canonical home per concept" rule.

| Field | Type | Notes |
|---|---|---|
| `id` | ULID | |
| `canonical_name` | string | one per concept — the invariant the vault maintains by hand |
| `aliases` | string[] | drives deterministic match before any LLM call |
| `kind` | enum | `concept`, `technology`, `pattern`, `algorithm`, `data_structure`, `project`, `person`, `experiment`, `decision`, `topic` |
| `definition` | text | short; itself a `Claim` in strict terms (§9, Q3) |
| `embedding_ref` | — | for similarity-based resolution |
| `vault_path` | string? | link back to the canonical Markdown file, when one exists |
| `created_at`, `updated_at`, `confidence` | — | |

`kind` exists so that `Technology`, `Project`, `Person`, `Experiment`,
and `Decision` need no separate tables (R4) while remaining queryable
and separately constrainable later.

### 2.5 `Claim`

An assertable statement. **The unit of understanding** — the thing that
can be supported, contradicted, refined, superseded, or doubted.

| Field | Type | Notes |
|---|---|---|
| `id` | ULID | |
| `statement` | text | normalized, single-assertion |
| `subject_concept_id` | FK | primary concept |
| `tier` | enum | provenance tier (§4) — **required** |
| `confidence` | float 0–1 | §4.4 |
| `status` | enum | `active`, `superseded`, `retracted`, `disputed` |
| `derivation` | enum | `deterministic` \| `model` (R6) |
| `extractor` / `model_id` / `prompt_version` | — | required when `derivation = model` |
| `valid_from`, `valid_to` | timestamp | supersession window, not deletion |
| `created_at` | timestamp | |

A `Claim` with no `EVIDENCED_BY` edge and `tier ≠ USER_ASSERTION` is a
**model invariant violation** and must fail validation. This is the
mechanical enforcement of Principle 10.

### 2.6 `EvidenceLink` (reified)

`Claim → Span`. Why a claim is believed.

`claim_id`, `span_id`, `relation` (`quotes` | `paraphrases` |
`infers_from`), `extractor`, `model_id?`, `confidence`, `created_at`.

`relation` is what distinguishes "the source says this" from "a model
concluded this from the source" at the level of individual evidence —
finer-grained than the claim's own tier, and the reason quoting can
never be confused with inference.

### 2.7 `ClaimLink` (reified)

`Claim → Claim`, carrying its own provenance (R3). This is the brief's
`Relationship` entity.

`from_claim_id`, `to_claim_id`, `type` (§5), `confidence`, `derivation`,
`model_id?`, `rationale` (text), `created_at`, `status`.

### 2.8 `Provenance` (embedded record)

Not a standalone table — an embedded, immutable record attached to every
assertable object (`Claim`, `ClaimLink`, `EvidenceLink`, `Synthesis`):

```
{ tier, derivation, agent, agent_version, model_id?, prompt_version?,
  inputs: [span_id|claim_id...], created_at, workflow_run_id }
```

`workflow_run_id` ties every object to the LangGraph run that produced
it, making the whole model replayable and auditable.

### 2.9 `Revision` (event log)

The append-only history spine, and the brief's `TimelineEvent`.

`id`, `entity_type`, `entity_id`, `op` (`create` | `supersede` |
`retract` | `merge` | `split` | `confidence_change`), `before`, `after`,
`cause` (`source_id` or `claim_id` that triggered it), `workflow_run_id`,
`created_at`.

**This is what makes "what changed in my understanding this month?"
answerable.** It is Phase 1 because it cannot be reconstructed
retroactively — a system that starts logging changes in Phase 9 has no
history for Phases 1–8.

---

## 3. Extended entities (Phase 4–5)

### `Contradiction` (reified, first-class)

`claim_a_id`, `claim_b_id`, `kind` (`direct` | `scope` | `temporal` |
`definitional`), `severity`, `status` (`open` | `resolved` |
`accepted_tension` | `false_positive`), `resolution_note`,
`detected_by`, `created_at`, `resolved_at`.

First-class rather than an edge type because a contradiction has a
lifecycle, needs a resolution, and is a thing the user is *notified
about*. `accepted_tension` matters: genuine unresolved disagreement in
a field is a valid terminal state, not a bug (Principle 12).

### `Synthesis`

A generated aggregate over claims — the highest-risk object in the
model, since it is the one most likely to be mistaken for evidence.

`id`, `scope` (concept/question/topic), `body` (Markdown),
`source_claim_ids[]`, `model_id`, `prompt_version`, `generated_at`,
`stale` (bool), `superseded_by`.

**Invariant:** a `Synthesis` is `stale = true` the moment any
constituent claim changes status or confidence. Staleness is computed
deterministically by graph traversal, never by an LLM. This is the
brief's "does this make an existing synthesis outdated?" — answered by
software, not judgment.

---

## 4. Provenance model

### 4.1 The five tiers

Ordered from strongest to weakest epistemic warrant:

| Tier | Means | Created by |
|---|---|---|
| `SOURCE_FACT` | Verbatim in a source; quotable | Deterministic extraction |
| `EXTRACTED_CLAIM` | Restatement of what a source asserts | LLM, bound to spans |
| `MODEL_INFERENCE` | Concluded from sources; not stated in any of them | LLM |
| `SYNTHESIS` | Aggregate across multiple claims | LLM over claim sets |
| `USER_ASSERTION` | The user said so | Human |

`USER_ASSERTION` is deliberately outside the strength ordering: it is
not *evidence*, but it is *authoritative* for this user's model, and it
is the only tier that may exist without an `EvidenceLink`.

### 4.2 The provenance floor rule

> **A derived object's tier can never be stronger than the weakest tier
> among its inputs.**

Inference over extracted claims is `MODEL_INFERENCE`, never
`SOURCE_FACT`. Synthesis over inferences is `SYNTHESIS`. Checked
deterministically at write time. This one rule is what structurally
prevents generated content from laundering itself into evidence — the
failure mode Principle 10 exists to forbid.

### 4.3 Source trust tiers

Independent of provenance tier; a faithful extraction from a weak source
is still a faithful extraction.

`peer_reviewed` | `official_docs` | `reputable_secondary` |
`community` | `unverified` | `user_authored`

Set deterministically from source metadata where possible, defaulting to
`unverified`. Feeds confidence, never overrides provenance.

### 4.4 Confidence

`confidence ∈ [0,1]`, and its **origin is always recorded**:

- deterministic (exact alias match on a concept → 1.0)
- model-reported (calibration is poor; treat as ordinal, not
  probability)
- aggregate (function of supporting/contradicting evidence and source
  trust)

Confidence is **not** a substitute for tier. A high-confidence
`MODEL_INFERENCE` is still an inference and must always be displayed as
one.

---

## 5. Relationship vocabulary

Domain and range are constrained — this is what prevents the graph
becoming an untyped mesh.

### Concept ↔ Concept (structural)

| Type | Meaning | Typical derivation |
|---|---|---|
| `PART_OF` | Composition/hierarchy | model, human-confirmable |
| `DEPENDS_ON` | Requires to function | model |
| `REQUIRES` | Prerequisite to understand | model |
| `IMPLEMENTS` | Concrete realization of an abstraction | model |
| `PRECEDES` | Temporal/sequential ordering | deterministic where dates exist |
| `EXPLAINS` | One concept accounts for another | model |
| `RELATED_TO` | Associated, relation unspecified | similarity |

> **`RELATED_TO` is a known hazard.** It is the edge every knowledge
> graph fills with noise until the graph means nothing. Constraint: it
> may only be created by similarity above a tuned threshold, must carry
> its similarity score, is excluded from reasoning traversals by
> default, and is a candidate for promotion to a specific type — never a
> fallback for "the model wasn't sure."

### Claim ↔ Claim (epistemic — always via `ClaimLink`)

| Type | Meaning |
|---|---|
| `SUPPORTS` | Increases warrant for the target |
| `CONTRADICTS` | Decreases warrant; may raise a `Contradiction` |
| `REFINES` | Same territory, more precise |
| `SUPERSEDES` | Replaces as current understanding; **target is retained** |
| `DERIVED_FROM` | Provenance lineage |

### Cross-type

`MENTIONS` (Span → Concept, deterministic: alias/NER match) ·
`EVIDENCED_BY` (Claim → Span, §2.6) · `ABOUT` (Claim → Concept) ·
`ANSWERS` (Claim → Question, Phase 9)

### Determinism split (Principle 7, made concrete)

| Deterministic | Model-derived |
|---|---|
| `MENTIONS`, `DERIVED_FROM`, `PRECEDES` (dated), `RELATED_TO` (similarity), all hashing/lineage | `SUPPORTS`, `CONTRADICTS`, `REFINES`, `SUPERSEDES`, `PART_OF`, `DEPENDS_ON`, `REQUIRES`, `IMPLEMENTS`, `EXPLAINS` |

Roughly half the edge vocabulary needs no LLM at all.

---

## 6. Deferred entities (Phase 9 — semantics reserved now)

Defined here so Phase-1 tables don't preclude them; not populated yet.

- **`Question`** — a research question with `status` (`open` |
  `partially_answered` | `answered`). Answered *by claims*, which is why
  claims must be first-class first.
- **`KnowledgeGap`** — a *derived* observation (concept with no claims;
  question with no answering claims; claim with only one source;
  contradiction unresolved past a threshold). Computed by deterministic
  graph queries, not generated.
- **`Topic`** — a *derived cluster* of concepts, not a hand-maintained
  taxonomy. Not an entity in Phase 1 (R4); reintroduced as a
  materialized view if clustering proves useful.
- **`Insight`** — deliberately **not** a separate entity. An insight is
  a `Claim` with `tier ∈ {MODEL_INFERENCE, SYNTHESIS}` and high
  salience. Making it separate would create a second, parallel truth
  path that escapes claim-level provenance rules — precisely what R1
  forbids.

---

## 7. Mapping the brief's 19 entities

| Brief entity | Disposition |
|---|---|
| Document, Source, Concept, Claim, Question, Relationship, Contradiction, KnowledgeGap, Synthesis | Kept (`Relationship` → reified `ClaimLink`) |
| Evidence | **Reified edge** `EvidenceLink` (R2) |
| Topic | Derived cluster of `Concept`, deferred |
| Project, Person, Technology, Experiment, Decision | `Concept.kind` values (R4) |
| Insight | `Claim` + tier + salience |
| TimelineEvent | `Revision` event log |
| *(added)* | **`Span`** — required for real provenance |

---

## 8. Mapping the existing vault

How the 620 existing files enter the model on first ingest.

| Vault content | Maps to |
|---|---|
| Any `.md` file | `Source(kind=markdown, trust_tier=user_authored)` + `Document` |
| Heading section | `Span` (structure-aware chunking — headings are already highly regular, see audit §4.3) |
| `DSA/01_Patterns/*.md` | `Concept(kind=pattern)`, `canonical_name` = filename stem |
| `DSA/02_Algorithms/`, `03_DataStructures/` | `Concept(kind=algorithm | data_structure)` |
| `Technologies/Docs/*.md` | `Concept(kind=technology)` + claims at `USER_ASSERTION` |
| `Projects/*/` | `Concept(kind=project)` + pack docs as sources |
| `Technologies/Templates/`, `Playbooks/`, `Prompt-Library/` | Sources; **not** concept-bearing (procedures, not assertions) |
| `Resources/*.md` | Pointers to *external* sources — seeds `Source` rows to ingest later |
| Wikilink `[[X]]` | `MENTIONS` → resolve to `Concept` by `canonical_name`/alias |
| Frontmatter `related:` | `RELATED_TO` — **only after the §6.2 repair in the audit**; currently unparseable in all 283 files |
| Frontmatter `tags:` | `Concept.aliases` / kind hints |
| Frontmatter `canonical: true` | Asserts this file is the canonical home — a direct, pre-existing statement of the invariant |
| Git history | `Revision` seed (`created_at`, authorship) |

**Two consequences worth stating plainly:**

1. **The existing corpus is `USER_ASSERTION`, not `SOURCE_FACT`.** It is
   hand-written by the user, mostly without citations. It is
   authoritative for what the user believes and is *not* evidence for
   what is true. Getting this wrong at ingest would poison the tiering
   of everything built on top.
2. **Filename stem as `canonical_name` works because the vault already
   enforces it.** "One concept, one canonical home" means the existing
   file structure *is* a concept table. That is an unusually clean
   bootstrap and a direct payoff from the corpus's existing discipline.

---

## 9. Open questions — human decision required

Marked open rather than guessed.

**Q1 — Claim granularity.** How atomic? "RAG reduces hallucination" vs
"RAG reduces hallucination *in open-domain QA* by grounding generation
in retrieved passages." Too atomic → explosion and lost context. Too
coarse → claims that are partly supported and partly contradicted, which
breaks the whole epistemic layer. *Recommendation: single-assertion,
context-preserving, with the source span always attached; tune against
the DSA corpus before locking.*

**Q2 — Concept identity across scale.** Is "attention" in a transformer
paper the same concept as "attention" in a UX note? Recommend
embedding-similarity + alias match with a **human confirmation step for
merges above a threshold** — merges are destructive-ish and must be
reviewable (Principle 11).

**Q3 — Is `Concept.definition` a field or a claim?** Strictly it is a
claim and should carry provenance. Pragmatically a denormalized field is
much easier to query. *Recommendation: field that is a
`current_definition_claim_id` pointer — keeps provenance, keeps
queryability.*

**Q4 — Confidence arithmetic.** Any principled aggregation (Bayesian,
Dempster-Shafer) rests on independence assumptions that are false here.
*Recommendation: no arithmetic in Phase 1. Store evidence counts by
tier and show them. An honest count beats a fabricated posterior.*

**Q5 — Does the vault get written back to?** Determines whether
`Concept.vault_path` is read-only or bidirectional. **Blocks
implementation** — see [ADR-001](../decisions/001-forge-knowledge-os.md).

**Q6 — Contradiction sensitivity.** Aggressive detection produces false
positives that erode trust faster than missed contradictions do. Needs a
labeled evaluation set before tuning — the existing corpus can provide
one, since it certainly contains real internal disagreements.

**Q7 — Namespaced vocabularies.** Audit §6.5 found two conflicting
convention systems in the vault. Should `Concept.kind` and tags be
global or per-namespace? *Recommendation: per-namespace, because it
requires no content rewriting.*

---

## Related

- [Current-state audit](../architecture/forge-current-state.md) — §6.2 (frontmatter repair), §8 (open decisions)
- [Target architecture](../architecture/target-architecture.md)
- [ADR-001](../decisions/001-forge-knowledge-os.md)
- [Roadmap](../roadmap.md)
