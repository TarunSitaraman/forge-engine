"""Deterministic wikilink resolution and classification.

No LLM. Where a link is ambiguous, this module **produces candidates rather
than guessing** — picking a target on the user's behalf would silently rewrite
the meaning of their corpus.

Obsidian resolves ``[[Some Note]]`` by filename stem across the whole vault,
which is why the resolver indexes by stem, by lowercased stem, and by
normalized stem (punctuation and spacing removed) in that order of confidence.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable

from .markdown import MarkdownLink, WikiLink

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
#: Residue that indicates the "link" is a parsing artifact, not a real link.
_MALFORMED_RE = re.compile(r"^[\s,\d'\"\[\]{}()<>|:;=+*/\\-]*$")


class LinkStatus(str, Enum):
    RESOLVED = "resolved"
    CASE_MISMATCH = "case_mismatch"
    PATH_MISMATCH = "path_mismatch"
    RENAMED_CANDIDATE = "renamed_candidate"
    AMBIGUOUS = "ambiguous"
    MALFORMED = "malformed"
    MISSING = "missing"
    UNKNOWN = "unknown"


#: Statuses that still point at a real file (link works, or works in Obsidian).
RESOLVING_STATUSES = frozenset(
    {LinkStatus.RESOLVED, LinkStatus.CASE_MISMATCH, LinkStatus.PATH_MISMATCH}
)

STATUS_DESCRIPTIONS: dict[LinkStatus, str] = {
    LinkStatus.RESOLVED: "Exact match on a vault file stem or path.",
    LinkStatus.CASE_MISMATCH: "Target exists but differs in letter case.",
    LinkStatus.PATH_MISMATCH: "Target names a path that does not exist, though the basename does.",
    LinkStatus.RENAMED_CANDIDATE: (
        "No exact match, but a file matches after normalizing punctuation and "
        "spacing — consistent with a rename."
    ),
    LinkStatus.AMBIGUOUS: "Several files share this stem; the intended target is undetermined.",
    LinkStatus.MALFORMED: "Not a usable link target (empty or punctuation/digits only).",
    LinkStatus.MISSING: "No file matches; the target page does not exist.",
    LinkStatus.UNKNOWN: "Could not be classified.",
}


@dataclass(frozen=True)
class ResolvedLink:
    source_path: str
    target: str
    status: LinkStatus
    line: int
    in_frontmatter: bool = False
    resolved_path: str | None = None
    #: Ordered suggestions when resolution is ambiguous or failed. Never applied.
    candidates: tuple[str, ...] = ()
    anchor: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "target": self.target,
            "status": self.status.value,
            "line": self.line,
            "in_frontmatter": self.in_frontmatter,
            "resolved_path": self.resolved_path,
            "candidates": list(self.candidates),
            "anchor": self.anchor,
        }


@dataclass
class LinkIndex:
    """Lookup tables over vault paths. Built once, reused for every file."""

    paths: tuple[str, ...]
    by_stem: dict[str, list[str]] = field(default_factory=dict)
    by_lower_stem: dict[str, list[str]] = field(default_factory=dict)
    by_normalized: dict[str, list[str]] = field(default_factory=dict)
    by_relpath: dict[str, str] = field(default_factory=dict)
    by_lower_relpath: dict[str, str] = field(default_factory=dict)
    #: Every directory containing at least one indexed file. Markdown links may
    #: legitimately target a directory (``](../personal-agent/)``), which
    #: resolves fine on GitHub and in Obsidian but has no ``.md`` path.
    directories: set[str] = field(default_factory=set)
    #: Non-Markdown files that exist in the repository — config, code, images.
    #:
    #: The index is built from indexed `.md` paths, so a Markdown link to
    #: `config/concept-identity.yaml` or a PNG has no entry and was reported
    #: MISSING even though it resolves fine on GitHub and in Obsidian. That is
    #: a property of the index, not a defect in the link. Populated by the
    #: caller, which is the only layer that may touch the filesystem.
    other_files: set[str] = field(default_factory=set)
    #: Normalized bare name -> vault path, for names a human has explicitly
    #: disambiguated. Keyed through `normalize()` on both sides, so a decision
    #: recorded for "Binary Search" also settles `[[binary search]]`.
    #:
    #: Two files may legitimately share a stem — `DSA/01_Patterns/Heap.md` and
    #: `DSA/03_DataStructures/Heap.md` both exist and both should. A bare
    #: `[[Heap]]` is then genuinely ambiguous, and the engine must not guess.
    #: But once a human has recorded what the bare name means, continuing to
    #: report it as unresolved is the engine ignoring an answer it was given.
    #: `forge identity decide` writes that answer; this is where it is read.
    decided: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        paths: Iterable[str],
        decided: dict[str, str] | None = None,
        other_files: set[str] | None = None,
    ) -> LinkIndex:
        ordered = tuple(sorted(paths))
        idx = cls(
            paths=ordered,
            decided=dict(decided or {}),
            other_files=set(other_files or ()),
        )
        for p in ordered:
            parent = PurePosixPath(p).parent
            while str(parent) not in (".", "/", ""):
                idx.directories.add(parent.as_posix())
                parent = parent.parent
            pp = PurePosixPath(p)
            stem = pp.stem
            idx.by_stem.setdefault(stem, []).append(p)
            idx.by_lower_stem.setdefault(stem.casefold(), []).append(p)
            idx.by_normalized.setdefault(normalize(stem), []).append(p)
            idx.by_relpath[p] = p
            idx.by_lower_relpath[p.casefold()] = p
            # Also index without the .md suffix, since links usually omit it.
            no_ext = p[:-3] if p.endswith(".md") else p
            idx.by_relpath.setdefault(no_ext, p)
            idx.by_lower_relpath.setdefault(no_ext.casefold(), p)
        return idx


def normalize(name: str) -> str:
    """Collapse to comparable form: lowercase, punctuation and spacing removed."""
    return _NORMALIZE_RE.sub("", name.casefold())


def resolve_wikilink(link: WikiLink, source_path: str, index: LinkIndex) -> ResolvedLink:
    """Classify one wikilink against the vault index."""
    target = link.target.strip()

    def make(status: LinkStatus, **kw: object) -> ResolvedLink:
        return ResolvedLink(
            source_path=source_path,
            target=target,
            status=status,
            line=link.line,
            in_frontmatter=link.in_frontmatter,
            anchor=link.anchor,
            **kw,  # type: ignore[arg-type]
        )

    if not target or _MALFORMED_RE.match(target):
        return make(LinkStatus.MALFORMED)

    # 1. Exact stem match.
    if (hits := index.by_stem.get(target)) is not None:
        if len(hits) == 1:
            return make(LinkStatus.RESOLVED, resolved_path=hits[0])
        # A recorded decision outranks the ambiguity it was recorded to settle.
        if (chosen := index.decided.get(normalize(target))) is not None and chosen in hits:
            return make(LinkStatus.RESOLVED, resolved_path=chosen)
        return make(LinkStatus.AMBIGUOUS, candidates=tuple(hits))

    # 2. Exact relative-path match (with or without .md).
    cleaned = target[:-3] if target.endswith(".md") else target
    if (hit := index.by_relpath.get(target) or index.by_relpath.get(cleaned)) is not None:
        return make(LinkStatus.RESOLVED, resolved_path=hit)

    # 3. Case-insensitive stem.
    if (hits := index.by_lower_stem.get(target.casefold())) is not None:
        if len(hits) == 1:
            return make(LinkStatus.CASE_MISMATCH, resolved_path=hits[0])
        if (chosen := index.decided.get(normalize(target))) is not None and chosen in hits:
            return make(LinkStatus.CASE_MISMATCH, resolved_path=chosen)
        return make(LinkStatus.AMBIGUOUS, candidates=tuple(hits))

    # 4. Case-insensitive relative path.
    if (
        hit := index.by_lower_relpath.get(target.casefold())
        or index.by_lower_relpath.get(cleaned.casefold())
    ) is not None:
        return make(LinkStatus.CASE_MISMATCH, resolved_path=hit)

    # 5. Target names a path whose directory is wrong but basename exists.
    if "/" in target:
        base = PurePosixPath(cleaned).name
        if (hits := index.by_stem.get(base)) is not None:
            if len(hits) == 1:
                return make(LinkStatus.PATH_MISMATCH, resolved_path=hits[0])
            return make(LinkStatus.AMBIGUOUS, candidates=tuple(hits))

    # 6. Normalized match — consistent with a rename (punctuation/spacing drift).
    if (hits := index.by_normalized.get(normalize(target))) is not None:
        if len(hits) == 1:
            return make(LinkStatus.RENAMED_CANDIDATE, candidates=(hits[0],))
        return make(LinkStatus.AMBIGUOUS, candidates=tuple(hits))

    # 7. Nothing matched. Offer close names as *candidates only*.
    close = difflib.get_close_matches(target, list(index.by_stem.keys()), n=3, cutoff=0.75)
    candidates = tuple(index.by_stem[c][0] for c in close)
    return make(LinkStatus.MISSING, candidates=candidates)


def resolve_markdown_link(
    link: MarkdownLink, source_path: str, index: LinkIndex
) -> ResolvedLink | None:
    """Classify a relative Markdown link. External URLs return ``None``.

    URL-decodes the target first. The Phase 0 audit initially reported two
    false-positive broken links in README.md because ``DSA%20Home.md`` was
    compared against the filesystem without decoding — correct for GitHub,
    wrong for a naive checker.
    """
    target = link.target
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None

    decoded = urllib.parse.unquote(target.split("#", 1)[0])
    if not decoded:
        return None

    base = PurePosixPath(source_path).parent
    candidate = _normalize_relative(base / decoded)

    if candidate in index.by_relpath:
        status, resolved = LinkStatus.RESOLVED, index.by_relpath[candidate]
    elif candidate in index.directories:
        # A directory link. Valid on GitHub and in Obsidian; it simply has no
        # .md target to point at.
        status, resolved = LinkStatus.RESOLVED, candidate + "/"
    elif candidate in index.other_files:
        # A link to a real non-Markdown file. Not indexed, but not broken.
        status, resolved = LinkStatus.RESOLVED, candidate
    else:
        status, resolved = LinkStatus.MISSING, None

    return ResolvedLink(
        source_path=source_path,
        target=target,
        status=status,
        line=link.line,
        resolved_path=resolved,
        anchor=target.split("#", 1)[1] if "#" in target else None,
    )


def _normalize_relative(path: PurePosixPath) -> str:
    """Collapse ``.``/``..`` segments without touching the filesystem."""
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)
