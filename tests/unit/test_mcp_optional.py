"""The MCP server is an optional extra, and its absence must read as a sentence.

Deliberately not guarded by `importorskip`: this file must run in an
environment *without* the MCP SDK, which is the case it is about.
"""

from __future__ import annotations

import builtins
import sys

import forge.mcp
import pytest


def test_the_package_imports_without_the_sdk():
    """`forge.mcp` must never import the SDK at module scope.

    If it did, `forge --help` would fail on a clone that only wants to index a
    vault, which is why the extras exist at all.
    """
    assert forge.mcp.MCP_SERVER_VERSION.startswith("mcp/")


def test_a_missing_extra_names_the_extra(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "mcp":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.startswith(("mcp", "forge.mcp.server"))]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        forge.mcp.create_server()

    message = str(excinfo.value)
    assert "forge-kb[mcp]" in message
    assert "optional extra" in message


def test_an_unrelated_import_error_is_not_relabelled(monkeypatch):
    """A different missing module is a real bug, not a packaging problem.

    Relabelling it would send someone to install an extra that was never the
    cause.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "mcp":
            raise ModuleNotFoundError(
                "No module named 'some_unrelated_dep'", name="some_unrelated_dep"
            )
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.startswith("forge.mcp.server")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        forge.mcp.create_server()

    assert "forge-kb[mcp]" not in str(excinfo.value)
    assert excinfo.value.name == "some_unrelated_dep"
