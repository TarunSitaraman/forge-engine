"""The API is an optional extra, and its absence must read as a sentence.

Deliberately not guarded by `importorskip`: this file must run in an
environment *without* FastAPI, which is the case it is about. A bare
`ModuleNotFoundError: No module named 'fastapi'` from three frames inside the
package tells a user nothing about what to install.
"""

from __future__ import annotations

import builtins
import sys

import forge.api
import pytest


def test_the_package_imports_without_fastapi():
    """`forge.api` itself must never import the framework at module scope.

    If it did, `forge --help` would fail on a clone that only wants to index a
    vault, which is the reason the extras exist at all.
    """
    assert forge.api.API_VERSION.startswith("api/")


def test_a_missing_extra_names_the_extra(monkeypatch):
    """The message must say `pip install 'forge-kb[api]'`, not just the module."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("fastapi", "starlette")):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.startswith(("fastapi", "starlette", "forge.api.app"))]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        forge.api.create_app()

    message = str(excinfo.value)
    assert "forge-kb[api]" in message
    assert "optional extra" in message


def test_an_unrelated_import_error_is_not_relabelled(monkeypatch):
    """Only FastAPI's absence means "install the extra".

    A different missing module discovered while loading the app is a real bug,
    and dressing it up as a packaging problem sends someone to fix the wrong
    thing. Simulated by failing the app import with a *different* module name,
    which is exactly the discrimination `create_app` makes: it keys on
    `exc.name`, not on the fact that an import failed.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("fastapi"):
            raise ModuleNotFoundError(
                "No module named 'some_unrelated_dep'", name="some_unrelated_dep"
            )
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.startswith("forge.api.app")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        forge.api.create_app()

    assert "forge-kb[api]" not in str(excinfo.value)
    assert excinfo.value.name == "some_unrelated_dep"
