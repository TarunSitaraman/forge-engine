# Forge CLI

*Every Phase 1 command, what it actually does, and what it will not do.*

**All commands are read-only with respect to the Markdown vault.** The only
directory Forge writes is `.forge/`, which is derived state and can be deleted
at any time — `forge index` rebuilds it.

---

## Install

```bash
pip install -e ".[dev]"     # from the repository root
forge --help
```

Python 3.11+. No model, no database server, and no API key is required for
anything below except `forge model-test`.

---

## Configuration

Environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_VAULT_PATH` | repo root (nearest `.git`) | Vault to index |
| `FORGE_STATE_DIR` | `<vault>/.forge` | Derived state |
| `FORGE_LLM_PROVIDER` | `ollama` | `ollama` or `mock` |
| `FORGE_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `FORGE_MODEL_DEFAULT` | `llama3.1:8b` | Model for all roles |
| `FORGE_MODEL_EXTRACTION` / `_ANALYSIS` / `_RESOLUTION` / `_SYNTHESIS` | — | Per-role override |
| `FORGE_LOG_LEVEL` / `FORGE_LOG_FORMAT` | `INFO` / `console` | `console` or `json` |

No API keys, ever. Configuration is validated at startup: a bad vault path or
an unbound model role fails immediately rather than mid-run.

---

## `forge index`

Walk the vault, hash every file, parse structure and metadata, resolve links,
detect changes, persist what changed, write reports.

```bash
forge index                 # normal run
forge index --json          # machine-readable
forge index --reset         # drop derived state first (safe: it rebuilds)
forge index --no-persist    # analyse without writing derived state
forge index --no-reports    # skip .forge/reports/
forge index -v              # verbose logging
```

```
Indexed 630 files in 0.7s
  fingerprint : 76ef7f11a21f5db7...
  changes     : 0 new, 0 modified, 630 unchanged, 0 deleted
  persisted   : 0 sources, 0 documents, 0 spans
  LLM calls   : 0
```

**`LLM calls: 0` is reported on every run, not only under test.** Indexing is
entirely deterministic; the `corpus` package cannot import `llm`.

Re-running with no edits persists nothing. Editing one file marks exactly that
file modified and reprocesses only it.

Writes to `.forge/reports/`: `corpus-stats.json`, `frontmatter-report.json`,
`link-report.json`, `convention-report.json`.

---

## `forge status`

Engine, derived-state, and provider status. Safe to run before anything is
indexed.

```
vault          : /home/user/forge
derived state  : /home/user/forge/.forge (exists: True)
markdown files : 630
indexed sources: 630
  spans=7003 documents=630 concepts=0 claims=0 revisions=630
llm provider   : ollama (UNAVAILABLE)
  cannot reach Ollama at http://localhost:11434: [Errno 111] Connection refused
```

`concepts=0 claims=0` is expected in Phase 1. Indexing describes the corpus; it
does not assert anything about it. Extraction is Phase 2+.

An unreachable provider is reported, never raised — status must work when
nothing is installed.

---

## `forge corpus-stats`

Statistics computed from the filesystem on every run. Nothing here is
maintained by hand, because the Phase 0 audit found stale counts in three
separate files.

```bash
forge corpus-stats
forge corpus-stats --json --top 20
```

Reports file/line/byte counts, frontmatter coverage, `canonical: true` count,
headings, code blocks, wikilink and Markdown-link totals, duplicate content
hashes, per-folder breakdown, and filename-style distribution.

---

## `forge diagnostics`

Report metadata, link, and convention problems. **Reports only — nothing is
modified.**

```bash
forge diagnostics                      # all
forge diagnostics frontmatter
forge diagnostics links --limit 40
forge diagnostics conventions
forge diagnostics --json               # full detail, including repair proposals
```

```
FRONTMATTER
  362/630 files have frontmatter; 294 valid, 68 invalid
    FM001: 68     YAML parse failure
    FM002: 215    nested-list wikilinks
    FM003: 268    no frontmatter
    FM008: 18     truncated final wikilink
  283 file(s) have verified repair proposals (NOT applied — approval required)

LINKS
  4656 total (4113 wiki, 589 markdown)
    ambiguous: 180   case_mismatch: 1   missing: 102   resolved: 4373
  unresolved: 282 occurrences across 89 distinct targets
     74x [ambiguous] 'Heap'  candidates=[...Patterns/Heap.md, ...DataStructures/Heap.md]

CONVENTIONS — UNRESOLVED — requires human decision (ADR-001 D3)
```

"Verified" means the proposed repair was applied in memory and re-parsed
successfully. It does **not** mean it was written. Applying repairs is a future
human-approved workflow (ADR-001 D2).

---

## `forge inspect <path>`

Everything deterministically known about one file.

```bash
forge inspect "DSA/01_Patterns/DFS.md"
forge inspect "Technologies/Docs/rag.md" --spans
forge inspect README.md --json
```

Shows hash, size, frontmatter state and keys, tags, recovered `related:` links,
heading count, code blocks, link counts, diagnostics, repair proposals as a
diff, unresolved links with candidates, and — with `--spans` — the derived
span breakdown with heading paths.

Useful for seeing a defect and its proposed fix side by side:

```
diagnostics    :
  [warning] FM008: Field 'related' ends with a truncated wikilink '[[Tree Traversal]'
  [error]   FM001: YAML parse failed: ...
repair proposals (NOT applied):
  line 6 verified=True
    - related: [[Binary Search Tree]], [[Tree Traversal]
    + related: ["Binary Search Tree", "Tree Traversal"]
```

---

## `forge model-test`

Run the local-model capability spike: structured concept extraction, claim
extraction, relationship extraction, and a small synthesis task, each repeated
so reliability is measured rather than sampled once.

```bash
forge model-test                          # writes docs/research/...
forge model-test --repetitions 5
forge model-test --no-write --json
forge model-test --note "M2 Pro, 32GB"    # repeatable
```

Writes `docs/research/local-model-capability-spike.md`.

**Exits non-zero when no model is reachable**, and the generated document says
so plainly rather than reporting an empty success. Contradiction detection is
deliberately not tested — it is not a required Phase 1 capability.

Requires Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
export FORGE_MODEL_DEFAULT=llama3.1:8b
forge model-test
```

---

## What the CLI will not do

No command writes to the vault, applies a repair, deletes a note, or calls a
paid API. `test_cli_never_writes_to_the_vault` byte-compares every Markdown
file before and after running the read commands.

---

## Related

- [Phase 1 implementation architecture](./architecture/phase-1-implementation.md)
- [Test strategy](./test-strategy.md)
- [ADR-001](./decisions/001-forge-knowledge-os.md)
