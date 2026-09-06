# Forge Engine

The Forge Knowledge OS, phases 1-4: a canonical knowledge model with enforced
provenance and revision tracking, deterministic corpus indexing, source
ingestion, human-approved proposals, a SQLite knowledge graph, and an agentic
evolution workflow that evaluates new evidence against what is already known.

- **Install & usage:** [`docs/cli.md`](../docs/cli.md)
- **Architecture:** phase [1](../docs/architecture/phase-1-implementation.md),
  [2](../docs/architecture/phase-2-implementation.md),
  [3](../docs/architecture/phase-3-implementation.md),
  [4](../docs/architecture/phase-4-implementation.md)
- **Tests:** [`docs/test-strategy.md`](../docs/test-strategy.md)

Read-only with respect to the Markdown vault. Everything it writes lives in
`.forge/`, which is derived state and can be deleted at any time.

```bash
pip install -e ".[dev]"     # Python 3.10+
forge index
forge diagnostics
python -m pytest tests      # 1,247 tests, offline, no model required
```

## Layout

| Path | What |
|---|---|
| `forge/domain/` | Pure domain model. No storage, no HTTP, no LLM. |
| `forge/corpus/`, `parsing/` | Deterministic vault indexing and Markdown parsing. |
| `forge/sources/`, `ingestion/` | PDF/Markdown acquisition, chunking into spans. |
| `forge/extraction/`, `matching/` | LLM candidate extraction; concept matching. |
| `forge/proposals/`, `activation/` | Proposed changes; approved changes becoming canonical. |
| `forge/graph/`, `retrieval/` | SQLite knowledge graph; FTS5 search. |
| `forge/evolution/` | The LangGraph workflow that evaluates new evidence. |
| `forge/llm/` | Provider abstraction: ollama / cloud / mock. |
| `forge/cli/` | The `forge` command. |
