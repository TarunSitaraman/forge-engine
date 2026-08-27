"""Deterministic staleness detection for documented external repositories."""

from .check import (
    CHECKED_KEY,
    COMMIT_KEY,
    UPSTREAM_CHECK_VERSION,
    URL_KEY,
    UpstreamError,
    UpstreamStatus,
    check,
    declared_upstreams,
    head_commit,
)

__all__ = [
    "check",
    "declared_upstreams",
    "head_commit",
    "UpstreamStatus",
    "UpstreamError",
    "UPSTREAM_CHECK_VERSION",
    "URL_KEY",
    "COMMIT_KEY",
    "CHECKED_KEY",
]
