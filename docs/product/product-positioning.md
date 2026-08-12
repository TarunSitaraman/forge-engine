# Forge — Product Positioning

*Where Forge sits in the stack, what owns the intelligence, and how to tell when a feature is being built in the wrong layer.*

---

## One sentence

**Forge is an AI-native knowledge infrastructure layer**: an engine that
maintains an evolving, evidence-traceable model of understanding, and
exposes it through multiple interchangeable interfaces.

Not an app that happens to have an API. An engine that happens to have
apps.

---

## The layering rule

```
  INTERFACES     Obsidian plugin | Web UI | CLI | MCP server
     (thin)      ------------------------------------------
                                    |
                            HTTP / local API
                                    |
  FORGE CORE     Ingestion | Retrieval | Graph | Evolution | Provenance
     (thick)     ------------------------------------------
                                    |
  SUBSTRATE      Markdown vault (truth) + derived indexes (rebuildable)
```

**The rule:** an interface may *present*, *navigate*, and *collect
input*. It may not *decide*. Any logic that determines what is true,
what is related, what is contradictory, or what is worth surfacing
belongs in the core.

**The test:** if a capability disappears when you switch from the web
UI to MCP, it was built in the wrong layer. Every interface should be
replaceable in a week without touching the engine.

This is why Obsidian is positioned as an interface, not the system.
Obsidian is an excellent Markdown editor and graph *viewer*. It is not
an inference engine, it cannot express provenance tiers or confidence,
and it cannot detect that two of your notes disagree. Forge's
philosophy has said "Obsidian as an interface rather than the
intelligence layer" from the start; this architecture is what makes
that statement enforceable rather than aspirational.

---

## Why "infrastructure layer" and not "app"

Three consequences, each of which changes what gets built:

**1. The model outlives the interfaces.** Interfaces are fashion; the
knowledge model is the asset. A user who spends two years feeding Forge
should be able to throw away every UI and still own everything valuable.
This is also why the Markdown vault stays authoritative — see
[ADR-001](../decisions/001-forge-knowledge-os.md).

**2. Multiple simultaneous consumers, none privileged.** The MCP server
is not a "Claude integration feature." It is the same core, addressed by
an agent instead of a person. Designing for that from the start prevents
the usual failure where the API is a degraded afterthought of the UI.

**3. Correctness beats surface area.** Infrastructure is judged on
whether it can be trusted, not how much it does. One traceable,
non-destructive ingestion path is worth more than nine source connectors
that silently produce unattributable claims.

---

## Who this is for

Primary user: **an engineer or researcher who accumulates knowledge
faster than they can organize it**, and whose existing system is a
graveyard of good notes they can no longer retrieve or reconcile.

The concrete instance is this repository's owner: 620 files, ~48,700
lines, curated with real discipline — and 145 broken links, 42% of
files with no metadata, and no way to answer "does anything I've
written contradict this?"

Secondary users, later and only if the core is sound: research groups
sharing a model; agents consuming a knowledge base over MCP.

**Explicitly not for:** casual note-takers (Forge is heavy machinery for
a light problem), teams needing collaborative editing (real-time
multi-writer is a different product), or anyone wanting a chat interface
to their documents (see [competitive boundary](./competitive-boundary.md)).

---

## What Forge competes with — and what it actually competes on

| Category | Their core operation | Forge's difference |
|---|---|---|
| Obsidian / Logseq / Roam | Human writes, human links | The model links itself; Obsidian remains a view |
| NotebookLM / ChatPDF | Retrieve from a fixed document set, answer | Sources *change the model*; the model persists and evolves |
| Generic RAG stacks | Chunk → embed → retrieve → generate | Retrieval is one subsystem; the differentiator is evolution + provenance |
| Zotero / Mendeley | Reference management | Manages *claims*, not citations |
| Mem / Reflect / "AI notes" | Auto-tagging, auto-linking | Contradiction, confidence, supersession, and history |

The honest summary: **many products can retrieve from your documents.
None of them maintain a position on what you believe, tell you when new
evidence undermines it, and show you the trail.** That is the whole
wager.

---

## What makes it defensible

Not the stack — LangGraph, Qdrant, and Neo4j are commodities. The
defensible parts are the ones that are tedious and unglamorous:

1. **The provenance model.** Five distinct tiers (source fact,
   extracted claim, model inference, synthesis, user assertion), tracked
   end-to-end and never collapsed. Most systems flatten these
   immediately because keeping them separate is expensive.
2. **Non-destructive evolution.** Superseded understanding is retained,
   with the evidence that caused the change.
3. **Determinism discipline.** Most of the pipeline is ordinary
   software. That makes Forge cheap, fast, testable, and offline-capable
   — properties that LLM-centric competitors structurally cannot match.
4. **The accumulated model itself.** Switching cost grows with every
   ingested source, and the model is the user's, in plain Markdown.

---

## Non-goals for the first version

Deferred deliberately, so that the core gets built:

Autonomous research agents · browser automation · automatic paper
downloading · YouTube ingestion · multi-agent debate · autonomous
writing · autonomous task execution · social integrations · enterprise
collaboration · billing · cloud multi-tenancy · complex auth · mobile
apps.

Each is a plausible future extension. Each would also, if built first,
consume the effort that the provenance and evolution model needs — and
those are the parts that cannot be retrofitted.

---

## Positioning drift tests

Run these when reviewing any proposed feature. A "yes" is a warning.

- Does this put the LLM on the critical path for something deterministic
  software could do? *(Principle 7)*
- Does this let generated content become indistinguishable from source
  evidence? *(Principle 10)*
- Does this make the chat box the primary interface? *(→ chatbot)*
- Does this require a paid API to work at all? *(Principle 9)*
- Does this add a source connector before provenance is solid?
  *(surface area over correctness)*
- Does this put decision logic in an interface? *(layering rule)*
- Does this overwrite prior state without recording it? *(Principle 11)*

---

## Related

- [Vision](./vision.md)
- [Competitive boundary](./competitive-boundary.md)
- [Target architecture](../architecture/target-architecture.md)
