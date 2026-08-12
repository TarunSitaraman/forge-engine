# Forge Knowledge OS — Documentation

*Engineering documentation for the Forge engine. Distinct from the Markdown vault in the rest of this repository, which is Forge's knowledge content.*

---

## Read in this order

| # | Document | Answers |
|---|---|---|
| 1 | [product/vision.md](./product/vision.md) | What are we building and why |
| 2 | [product/product-positioning.md](./product/product-positioning.md) | Where does it sit in the stack |
| 3 | [product/competitive-boundary.md](./product/competitive-boundary.md) | What must it never become |
| 4 | [architecture/forge-current-state.md](./architecture/forge-current-state.md) | What exists today (audited, measured) |
| 5 | [architecture/target-architecture.md](./architecture/target-architecture.md) | What we're building toward |
| 6 | [architecture/technology-decisions.md](./architecture/technology-decisions.md) | What we're building it with |
| 7 | [knowledge-model/canonical-model.md](./knowledge-model/canonical-model.md) | How knowledge is represented |
| 8 | [roadmap.md](./roadmap.md) | In what order, with what exit gates |
| 9 | [decisions/001-forge-knowledge-os.md](./decisions/001-forge-knowledge-os.md) | What was decided, and what is still open |

**If you read only one:** the [current-state audit](./architecture/forge-current-state.md).
It is the factual basis for everything else, and §8 lists the decisions
that block implementation.

---

## Why there are nine documents and not thirty

The brief sketched a wider tree (`ingestion/`, `retrieval/`, `agents/`,
`langgraph/`, `api/`, `ux/`, `deployment/`) and then said: *"Do not
create dozens of documents unnecessarily. Group closely related
concepts."*

Those topics are covered, grouped where they belong rather than split
across near-empty files:

| Topic | Currently lives in |
|---|---|
| Ingestion pipeline | [target-architecture §4](./architecture/target-architecture.md) |
| Retrieval | [target-architecture §6](./architecture/target-architecture.md) |
| LangGraph workflows / node design | [target-architecture §5](./architecture/target-architecture.md) |
| "Agents" | Same — Forge has workflow nodes, not agents, deliberately |
| Deployment | [target-architecture §10](./architecture/target-architecture.md), [technology-decisions §8](./architecture/technology-decisions.md) |
| API | Not yet designed — Phase 6 |
| UX | Not yet designed — Phase 6 |

A folder is created when its content exists. Writing `docs/api/` before
an API exists would produce a placeholder that goes stale before it is
ever true — precisely the drift the audit measured in the vault (§6.4).

---

## Conventions in this tree

- **Relative Markdown links**, not wikilinks — these docs are read on
  GitHub as often as in Obsidian. Permitted by
  [`CONVENTIONS.md`](../CONVENTIONS.md) for exactly this case.
- **kebab-case filenames**, per `CONVENTIONS.md`.
- **ADRs are immutable once accepted.** Supersede with a new numbered
  ADR; never rewrite history.
- **Measured claims cite their measurement.** Numbers in the audit were
  taken from the filesystem, not from existing documentation — the
  audit found three separate cases of stale hand-maintained counts, and
  this tree should not add a fourth.

---

## Status

Phase 0 complete. **No implementation has begun**, and none should
until decisions D1 (repository layout) and D2 (write-back policy) in
[ADR-001 §6](./decisions/001-forge-knowledge-os.md) are approved.
