#!/usr/bin/env python3
"""Thin wrapper around `forge proposals audit-grounding`.

Prefer the CLI:

    forge proposals audit-grounding

The audit lives in `forge.proposals.grounding_audit` and is reachable from the
installed command, which is the form that works on a machine where the engine
was installed with pipx — there, the `python3` on PATH is not the interpreter
that owns the engine's dependencies, and running this file directly fails with
ModuleNotFoundError. This wrapper stays for scripted use from inside a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.cli.main import app  # noqa: E402

if __name__ == "__main__":
    app(["proposals", "audit-grounding", *sys.argv[1:]])
