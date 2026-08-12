"""Structured logging.

Every log line carries a ``run_id`` so that a single indexing run (and, later,
a LangGraph workflow run) can be reconstructed from logs alone.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog once per process. Idempotent."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # Route through stdlib logging rather than structlog's PrintLogger.
        # PrintLogger binds the output stream at construction time and, with
        # cache_logger_on_first_use, holds it for the process lifetime — so any
        # caller that swaps sys.stderr (test runners and CLI harnesses do)
        # leaves logging writing to a closed file. Logging must never be able
        # to crash the program that is logging.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def new_run_id() -> str:
    """Fresh correlation id for one engine run."""
    return uuid.uuid4().hex[:16]


def bind_run(run_id: str) -> None:
    structlog.contextvars.bind_contextvars(run_id=run_id)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
