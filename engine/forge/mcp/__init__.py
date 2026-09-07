"""The MCP server (Phase 8), exposing the knowledge queries to an agent.

**Read-only, like the HTTP API, and for the same reason.** Knowledge changes
through proposal and activation, which require a human decision. An agent that
could write would be a second way in without that gate, and an agent is exactly
the caller you least want holding it.

The MCP SDK is an **optional extra** (`pip install forge-kb[mcp]`). Importing
this package without it raises a message naming the extra rather than a bare
ImportError from three frames down.
"""

from __future__ import annotations

MCP_SERVER_VERSION = "mcp/0.1.0"

_MISSING = (
    "The MCP server needs the MCP SDK, which is an optional extra:\n"
    "    pip install 'forge-kb[mcp]'\n"
    "Everything else in Forge works without it."
)


def create_server(*args, **kwargs):
    """Build the MCP server. Imported lazily so the extra stays optional."""
    try:
        from .server import create_server as _create
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the CLI
        if exc.name and exc.name.split(".")[0] == "mcp":
            raise ModuleNotFoundError(_MISSING) from exc
        raise
    return _create(*args, **kwargs)


__all__ = ["MCP_SERVER_VERSION", "create_server"]
