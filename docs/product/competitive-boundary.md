# Forge: Competitive Boundary

*The eight things Forge must not become, the specific decision that turns Forge into each one, and the observable symptom that says it already happened.*

---

## How to use this document

Every product on this list is a *reasonable* product. The danger is not
that someone will decide to build a ChatGPT wrapper on purpose. It is
that each one is **one plausible engineering decision away**, and each
decision looks locally correct at the time it is made.

So each entry below has four parts:

- **The drift decision.** The specific, tempting choice that causes it.
- **Symptom.** What you would observe once it has happened.
- **The boundary.** What to do instead.
- **Structural defense.** The thing in the architecture that makes the
  drift hard rather than merely discouraged.

Review this list at every phase boundary in the
[roadmap](../roadmap.md).

---

## 1. Not another chatbot

**Drift decision:** making conversation the primary interface, because
a chat box is the fastest thing to demo.

**Symptom:** the main screen is a message thread. The graph is a
secondary tab nobody opens. Product discussions are about response
quality rather than model correctness.

**Boundary:** the primary artifact is the **knowledge model**,
browsable, inspectable, and useful with no conversation at all. Natural
language is one query method among several (graph traversal, filters,
lexical search, similarity).

**Structural defense:** build the graph explorer before any chat
surface. If the model is only legible through generated prose, the
model is not legible.

---

## 2. Not a ChatGPT wrapper

**Drift decision:** letting the system prompt become where behavior
lives, solving problems by editing prompts rather than writing code.

**Symptom:** the interesting logic is in a `prompts/` directory. Removing
the LLM removes the product. Changing model versions changes behavior in
ways nobody can predict or test.

**Boundary:** the LLM is a **component with a narrow, typed contract**:
structured input, schema-validated output, deterministic post-processing.
Most of the pipeline should run with the model stubbed out.

**Structural defense:** provider abstraction plus a mock provider used
in tests. If the test suite cannot run offline against a deterministic
fake, the LLM is load-bearing in the wrong way.

---

## 3. Not another generic RAG chatbot

**Drift decision:** stopping at chunk → embed → retrieve → generate,
because that pipeline works and produces demoable answers quickly.

**Symptom:** ingestion's only output is embeddings. The graph is
decorative. Nothing in the system can answer "did this source change
anything?", because ingestion has no concept of change.

**Boundary:** retrieval is **one subsystem**, feeding concept
resolution, relationship discovery, contradiction analysis, and model
update. RAG is plumbing here, not the product.

**Structural defense:** the ingestion workflow's completion criterion is
*a diff to the knowledge model*, not *a vector written*. An ingest that
produced no model change is a distinct, reportable outcome.

---

## 4. Not a PDF summarizer

**Drift decision:** treating "user adds PDF → user gets summary" as the
core loop, because it is the most legible unit of value.

**Symptom:** each document has a summary; nothing relates documents to
each other. Value is per-document and does not compound.

**Boundary:** a document is **evidence**, not an output unit. Its value
is what it changes in the model. Two documents about the same concept
must converge on one concept, not produce two summaries.

**Structural defense:** the MVP's acceptance test is explicitly the
*second* document, detecting overlap and updating the graph instead of
duplicating. A summarizer passes document one and fails document two.

---

## 5. Not an Obsidian clone

**Drift decision:** rebuilding editing, panes, and vault management,
because Obsidian's UX is proven and users ask for familiar things.

**Symptom:** effort goes into a Markdown editor. Forge competes on
editing ergonomics, a fight it cannot win and does not need.

**Boundary:** Obsidian **stays** the editor. Forge adds the layer
Obsidian structurally cannot have: inference, provenance, contradiction
detection, evolution. The plugin surfaces engine capabilities inside the
editor the user already likes.

**Structural defense:** no editing surface in Forge's own interfaces
until the engine is complete. Read, navigate, inspect: not compose.

---

## 6. Not a NotebookLM clone

**Drift decision:** scoping intelligence to a *session*, "here are my
sources, answer questions about them", because bounded context is much
easier to make accurate.

**Symptom:** knowledge resets between sessions. There is no persistent
belief state. "What changed this month?" is unanswerable because nothing
persists to change.

**Boundary:** Forge maintains **one continuous, persistent model** that
accumulates across sources and time. This is the single sharpest
distinction on this list: NotebookLM answers *about documents*; Forge
maintains *a position* that documents modify.

**Structural defense:** temporal versioning and supersession in the
knowledge model from Phase 1, before any query interface exists.
Session-scoped systems cannot be retrofitted with history.

---

## 7. Not another "AI notes" app

**Drift decision:** shipping auto-tagging and auto-linking and calling
the intelligence layer done, because those demo well and are genuinely
useful.

**Symptom:** the AI's entire job is metadata. Notes are still notes;
nothing has an opinion; nothing can disagree.

**Boundary:** tags and links are **byproducts**. The deliverables are
claims, evidence, contradictions, confidence, and gaps.

**Structural defense:** `Claim`, `Evidence`, and `Contradiction` are
first-class entities in the canonical model, not attributes hung off a
document.

---

## 8. Not an autonomous coding agent

**Drift decision:** letting "Forge understands my repos" slide into
"Forge changes my repos," because the capability is adjacent and the
demo is impressive.

**Symptom:** Forge writes code, opens PRs, or executes tasks. Blast
radius and trust requirements change completely.

**Boundary:** Forge **reads** code as a knowledge source and models what
it means. It does not act on systems. Ingesting a repository produces
understanding, not commits.

**Structural defense:** no write credentials to anything except its own
derived stores, and the vault only under the write-back policy of
[ADR-001](../decisions/001-forge-knowledge-os.md).

---

## The two failure modes underneath all eight

Every entry above is a surface form of one of these:

**A. Collapsing the model into the interface.** Entries 1, 5, 6. The
knowledge model stops being an independent asset and becomes whatever
the current UI can display. Defense: the layering rule in
[positioning](./product-positioning.md), interfaces present, the core
decides.

**B. Collapsing understanding into retrieval.** Entries 2, 3, 4, 7. The
system's only real operation becomes "find relevant text and generate
prose about it," and every claim about evolution becomes marketing.
Defense: ingestion is measured by *model change*, and provenance tiers
keep generated content permanently distinguishable from evidence.

Entry 8 is the odd one out. It is a scope failure rather than an
architecture failure, which is why it is the easiest to hold the line
on and the least likely to happen by accident.

---

## Boundary review checklist

Run at each phase boundary. Any "yes" requires an explicit, recorded
decision before proceeding.

- [ ] Is the primary interface now a chat thread?
- [ ] Does core behavior live in prompt text rather than code?
- [ ] Can ingestion complete without producing a model change?
- [ ] Is per-document output the main unit of value?
- [ ] Are we building editing surfaces?
- [ ] Is knowledge scoped to a session rather than persistent?
- [ ] Is the AI's contribution mostly metadata?
- [ ] Does Forge write to anything outside its own stores?
- [ ] Can generated content be mistaken for source evidence?
- [ ] Does any core path require a paid API?

---

## Related

- [Vision](./vision.md)
- [Product positioning](./product-positioning.md)
- [Roadmap](../roadmap.md)
