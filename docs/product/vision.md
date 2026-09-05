# Forge: Vision

*What Forge is becoming, and the constraints that decide whether a given change belongs in it.*

---

## The thesis

> **Forge does not merely store information. Forge maintains
> understanding.**

Every note-taking tool ever built optimizes the same operation: *put
information in, get the same information back out*. Storage is solved.
Search is solved. Neither produces understanding, because understanding
is not a retrieval result. It is a **model that changes when new
evidence arrives**.

The gap Forge targets is specific. Today, when you read a paper that
contradicts something you wrote six months ago, one of three things
happens:

1. You never notice the contradiction. Both beliefs coexist in your
   vault forever.
2. You notice, and overwrite the old note. The old understanding, and
   your reason for holding it, are gone.
3. You notice, and write a third note. Now three notes disagree and
   nothing says which is current.

All three are failures, and all three are what the *existing* Forge
corpus does today: 620 files with no mechanism to detect that any two
of them disagree.

Forge's job is to make the fourth outcome the default: the new evidence
is linked to the existing claim, the disagreement is recorded as a
first-class object, confidence is adjusted, the prior understanding is
retained as history, and you are told that something you believed has
been challenged.

---

## What Forge is

**An AI-native knowledge infrastructure layer**: the engine that turns
scattered sources into a traceable, evolving model of understanding,
and exposes that model through whatever interface you prefer.

```
                        SOURCES
                           |
       +--------+--------+--------+--------+
       |        |        |        |        |
     PDFs    GitHub  Obsidian    Web     Code
       |        |        |        |        |
       +--------+--------+--------+--------+
                           |
                      FORGE CORE
                           |
       +-----------+-----------+-----------+
       |           |           |           |
   Knowledge   Retrieval    Memory     Evolution
     Graph      Engine      Engine      Engine
       |           |           |           |
       +-----------+-----------+-----------+
                           |
                   KNOWLEDGE MODEL
                           |
       +-----------+-----------+-----------+
       |           |           |           |
   Obsidian     Web UI    /   API        MCP
    Plugin                              Server
```

The intelligence lives in the core. Obsidian, the web UI, the CLI, and
MCP are **interchangeable views onto one model**: not four products.
If a capability only works in one of them, it was built in the wrong
layer.

The storage layer is not the product. A graph database is not a
differentiator; a graph you can *trust, question, and watch change* is.

---

## The twelve constraints

These are architectural constraints, not aspirations. Each one is
falsifiable, the "violated when" column is the test.

| # | Principle | Violated when |
|---|---|---|
| 1 | Research is not a collection of documents | The model is a list of files with embeddings attached |
| 2 | Knowledge ≠ stored notes | Ingesting a source produces only a note, not a change to the model |
| 3 | New information can change the existing model | Ingestion is append-only |
| 4 | Information stays traceable to evidence | A claim exists that cannot be resolved to a source span |
| 5 | Minimize manual organization | The user must file, tag, or link things by hand for the system to work |
| 6 | The LLM is a component, not the source of truth | A model output is stored without being marked as a model output |
| 7 | Deterministic software does deterministic work | An LLM is asked to hash, parse, chunk, or traverse |
| 8 | LLMs used selectively, for semantic tasks | Token spend scales linearly with corpus size on re-ingest |
| 9 | Local-first, usable with no paid API | Any core path requires a cloud key |
| 10 | Every important claim traces to evidence | Synthesis is indistinguishable from source text |
| 11 | Knowledge is never silently overwritten | An update destroys the prior state without recording it |
| 12 | Preserve uncertainty, disagreement, provenance, history | Everything reads as equally certain |

### The two that will actually be under pressure

**Principle 7 (deterministic work stays deterministic)** is the one
that erodes quietly. It is always faster to ask a model than to write a
parser, and the result usually looks right. The discipline is
mechanical: PDF extraction → parser. Chunking → algorithm. Dedup →
similarity + metadata. Traversal → query. Hashing → hash. Source
identity → metadata. An LLM earns a call only when the task is
genuinely semantic, naming a concept, judging whether two statements
conflict, deciding whether B refines A.

**Principle 11 (never silently overwrite)** is the one that
distinguishes Forge from every competitor, and it is the most expensive
to honor, because "the old value is gone" is the default behavior of
every storage system. Superseding must be an explicit, recorded,
reversible operation with both states retained.

---

## What "maintaining understanding" concretely means

When a new source arrives, Forge asks nine questions. These are the
functional spec of the evolution engine:

1. Is this already known?
2. Does it support an existing claim?
3. Does it contradict an existing claim?
4. Does it refine an existing concept?
5. Does it introduce a new concept?
6. Does it change confidence in an existing understanding?
7. Does it raise a new research question?
8. Does it expose a knowledge gap?
9. Does it make an existing synthesis outdated?

```
    OLD UNDERSTANDING
            |
       NEW EVIDENCE
            |
      CHANGE ANALYSIS
            |
   UPDATED UNDERSTANDING
            |
     HISTORY PRESERVED
```

The last box is not optional and not an audit log bolted on afterwards.
It is what makes the questions above answerable *over time* rather than
only at write time.

---

## The questions Forge should eventually answer

These are the product's real acceptance criteria. They are deliberately
not answerable by a search box.

- *What do I currently believe about agent memory?*
- *Which sources support that belief? Which disagree?*
- *What changed in my understanding this month?*
- *What questions remain unanswered?*
- *What concepts am I missing?*
- *Which papers are most relevant to this unresolved question?*

Note what each one requires:

| Question | Requires |
|---|---|
| What do I believe? | Claims with confidence, not documents |
| Which sources support/disagree? | Typed `SUPPORTS`/`CONTRADICTS` edges with provenance |
| What changed this month? | Temporal versioning of the model itself |
| What's unanswered? | Questions as first-class entities |
| What am I missing? | Gap detection over graph structure |
| Most relevant paper? | Retrieval scoped by an open question, not a keyword |

None of these are on the MVP path. All of them constrain it: **if the
Phase-1 knowledge model cannot in principle express these, the model is
wrong**: which is exactly why the model is designed before the
pipeline is built.

---

## Relationship to the existing Forge

Forge already exists as a 620-file, ~48,700-line curated Markdown
corpus (see [current-state audit](../architecture/forge-current-state.md)).
That corpus is not legacy to be migrated away from. It is:

- **the first ingestion source**, and a far better evaluation set than
  synthetic fixtures;
- **an existing specification of the invariants.** The Forge
  Engineering Constitution, `CONVENTIONS.md`, and the Validation
  Checklist already define canonical homes, typed relationships, and a
  12-point quality gate;
- **an existing proof of the problem**. The audit measured 145 broken
  links, 42% of files with no metadata, and 283 malformed relationship
  fields: all in a repository whose written rules forbid exactly those
  things.

That last point is the honest argument for this whole project. The
rules were good and a disciplined author still could not hold them by
hand at 620 files. The engine's first job is to enforce mechanically
what the corpus currently maintains by intention.

**Success is not "Forge has a graph database."** Success is: a year
from now, the user asks what they believe about something they studied
six months ago, and gets an answer with sources, dissent, confidence,
and a record of how that belief changed: without ever having filed a
note by hand.

---

## Related

- [Product positioning](./product-positioning.md)
- [Competitive boundary](./competitive-boundary.md)
- [Current-state audit](../architecture/forge-current-state.md)
- [Canonical knowledge model](../knowledge-model/canonical-model.md)
