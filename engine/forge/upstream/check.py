"""Detect when a documented external repository has moved on.

`Projects/` holds knowledge packs describing repositories that live elsewhere.
Those repositories change; the packs do not. Nothing in the vault could say
whether `Projects/quickcover/` still described QuickCover, so the docs aged
invisibly — the failure mode the whole vault exists to avoid.

**Detection is deterministic and needs no model and no API token.**
`git ls-remote <url> HEAD` returns the current head commit. Comparing it with
the commit a pack recorded is a string comparison. That choice matters:

* No GitHub token to store, rotate, or leak — git already holds the user's
  credentials, so private repositories work exactly as public ones do.
* Not GitHub-specific. Any git remote answers `ls-remote`.
* No API rate limit worth the name.

Rewriting a drifted pack is a separate, later step that does need judgement.
This module only ever *reports*; it never edits the vault.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..logging import get_logger

log = get_logger(__name__)

UPSTREAM_CHECK_VERSION = "upstream/0.1.0"

#: Frontmatter keys a pack uses to declare and pin its upstream.
URL_KEY = "upstream_repo"
COMMIT_KEY = "upstream_commit"
CHECKED_KEY = "upstream_checked"


class UpstreamError(Exception):
    """The remote could not be reached or did not answer with a head commit."""


@dataclass(frozen=True)
class UpstreamStatus:
    """One pack, checked against its upstream."""

    path: str
    url: str
    recorded: str | None
    current: str | None
    error: str | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "unreachable"
        if self.recorded is None:
            return "unpinned"
        if self.current == self.recorded:
            return "current"
        return "drifted"

    @property
    def drifted(self) -> bool:
        return self.state == "drifted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "url": self.url,
            "state": self.state,
            "recorded": self.recorded,
            "current": self.current,
            "error": self.error,
        }


def head_commit(url: str, *, timeout: float = 30.0) -> str:
    """Current head commit of a remote, via `git ls-remote`.

    Uses the ambient git credentials rather than an API token, which is what
    makes private repositories work without Forge ever handling a secret.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - git is a hard dep
        raise UpstreamError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise UpstreamError(f"timed out after {timeout:.0f}s") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise UpstreamError(detail[-1] if detail else f"git exited {result.returncode}")

    line = (result.stdout or "").strip().split("\n", 1)[0]
    sha = line.split("\t", 1)[0].strip()
    if not sha:
        raise UpstreamError("remote returned no HEAD")
    return sha


def declared_upstreams(index, vault_path: Path) -> list[tuple[str, str, str | None]]:
    """(path, url, recorded_commit) for every page declaring an upstream.

    The corpus index records which frontmatter *keys* a file has but not their
    values, so candidates are found from the index — cheap, no I/O — and only
    those few files are re-read for the values. Storing every frontmatter value
    in the index to save four reads would be the wrong trade.
    """
    from ..parsing.frontmatter import parse_frontmatter
    from ..parsing.markdown import parse_markdown

    found: list[tuple[str, str, str | None]] = []
    for f in index.files:
        if URL_KEY not in f.frontmatter_keys:
            continue
        try:
            raw = (vault_path / f.path).read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("upstream_declaration_unreadable", path=f.path, error=str(exc))
            continue
        parsed = parse_markdown(raw)
        data = (parse_frontmatter(parsed.frontmatter_raw or "").data) or {}
        url = data.get(URL_KEY)
        if not url:
            continue
        found.append((f.path, str(url), _as_str(data.get(COMMIT_KEY))))
    return found


def _as_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def check(
    upstreams: Iterable[tuple[str, str, str | None]],
    fetch: Callable[[str], str] = head_commit,
) -> list[UpstreamStatus]:
    """Compare each declared upstream against its recorded commit.

    `fetch` is injectable so the whole path is testable without a network —
    the same discipline the LLM providers follow.
    """
    statuses: list[UpstreamStatus] = []
    for path, url, recorded in upstreams:
        try:
            current = fetch(url)
        except UpstreamError as exc:
            statuses.append(
                UpstreamStatus(path=path, url=url, recorded=recorded, current=None, error=str(exc))
            )
            continue
        statuses.append(
            UpstreamStatus(path=path, url=url, recorded=recorded, current=current)
        )
        log.info(
            "upstream_checked",
            path=path,
            state=statuses[-1].state,
            recorded=(recorded or "")[:12],
            current=current[:12],
        )
    return statuses
