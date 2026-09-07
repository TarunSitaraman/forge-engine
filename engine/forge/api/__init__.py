"""The read-only HTTP API and graph explorer (Phase 6).

**Read-only is a design constraint, not a stage.** Nothing in this package
writes to the store. Knowledge changes through the proposal and activation
path, which requires a human decision; an HTTP endpoint that could mutate the
graph would be a second way in, without that gate. Every route here is a GET.

FastAPI is an **optional extra** (`pip install forge-kb[api]`). Importing this
package without it raises a clear message naming the extra, rather than a bare
ImportError from three frames down. `forge index` on a fresh clone must not
require a web framework.
"""

from __future__ import annotations

API_VERSION = "api/0.1.0"

_MISSING = (
    "The HTTP API needs FastAPI, which is an optional extra:\n"
    "    pip install 'forge-kb[api]'\n"
    "Everything else in Forge works without it."
)


def create_app(*args, **kwargs):
    """Build the FastAPI application. Imported lazily so the extra stays optional."""
    try:
        from .app import create_app as _create_app
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the CLI
        if exc.name in {"fastapi", "starlette"}:
            raise ModuleNotFoundError(_MISSING) from exc
        raise
    return _create_app(*args, **kwargs)


__all__ = ["API_VERSION", "create_app"]
