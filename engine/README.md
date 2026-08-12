# Forge Engine

Phase 1 implementation of the Forge Knowledge OS: canonical knowledge model,
provenance, revision tracking, and deterministic corpus indexing.

- **Install & usage:** [`docs/cli.md`](../docs/cli.md)
- **Architecture:** [`docs/architecture/phase-1-implementation.md`](../docs/architecture/phase-1-implementation.md)
- **Tests:** [`docs/test-strategy.md`](../docs/test-strategy.md)

Read-only with respect to the Markdown vault. Everything it writes lives in
`.forge/`, which is derived state and can be deleted at any time.

```bash
pip install -e ".[dev]"
forge index
forge diagnostics
```
