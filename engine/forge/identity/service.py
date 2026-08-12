"""Identity service — bridges user decisions to the matcher.

Two jobs:

1. **Scaffold** the identity file from collisions actually present in the
   vault, describing each one *without deciding it*. The user then edits one
   line to record their decision.
2. **Resolve** a name against those decisions, returning an
   :class:`~forge.domain.IdentityState` that says how the answer was reached.

Deterministic throughout. No LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from ..domain import IdentityState
from ..logging import get_logger
from ..parsing.links import normalize
from .config import CollisionResolution, ConceptIdentity, IdentityConfig

log = get_logger(__name__)

#: Folder name -> namespace, for scaffolding collisions out of the DSA layout.
#: Only used to *suggest* a namespace in the generated file; the user decides.
_FOLDER_NAMESPACES: dict[str, tuple[str, str]] = {
    "01_Patterns": ("pattern", "pattern"),
    "02_Algorithms": ("algorithm", "algorithm"),
    "03_DataStructures": ("data-structure", "data_structure"),
    "04_Problems": ("problem", "problem"),
    "07_Interview": ("interview-guide", "interview_guide"),
    "09_CheatSheets": ("cheat-sheet", "cheat_sheet"),
}


@dataclass(frozen=True)
class IdentityResolution:
    """Outcome of resolving one name against the identity configuration."""

    name: str
    state: IdentityState
    identity: ConceptIdentity | None = None
    candidates: tuple[ConceptIdentity, ...] = ()
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.state in (
            IdentityState.EXACT_MATCH,
            IdentityState.ALIAS_MATCH,
            IdentityState.RESOLVED_BY_USER,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "reason": self.reason,
            "identity": self.identity.to_dict() if self.identity else None,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class IdentityService:
    """Applies persisted identity decisions."""

    def __init__(self, config: IdentityConfig | None = None) -> None:
        self.config = config or IdentityConfig()

    # -- resolution --------------------------------------------------------

    def resolve(self, name: str) -> IdentityResolution:
        """Resolve a bare name against the user's recorded decisions."""
        cleaned = name.strip()
        if not cleaned:
            return IdentityResolution(name=name, state=IdentityState.NEW, reason="empty name")

        if (canonical := self.config.alias_target(cleaned)) is not None:
            return IdentityResolution(
                name=cleaned,
                state=IdentityState.ALIAS_MATCH,
                identity=ConceptIdentity(canonical_name=canonical),
                reason=f"user-registered alias for {canonical!r}",
            )

        resolution = self.config.resolution_for(cleaned)
        if resolution is None:
            return IdentityResolution(
                name=cleaned,
                state=IdentityState.NEW,
                reason="no identity decision recorded for this name",
            )

        if not resolution.resolved:
            # The user knows about this collision and has deliberately left it
            # undecided. Still ambiguous — but now *knowingly* so.
            return IdentityResolution(
                name=cleaned,
                state=IdentityState.AMBIGUOUS,
                candidates=resolution.identities,
                reason=(
                    f"collision is documented in the identity config but no default is set; "
                    f"choose one of: {', '.join(i.qualified_name for i in resolution.identities)}"
                ),
            )

        identity = resolution.identity_for(resolution.default or "")
        return IdentityResolution(
            name=cleaned,
            state=IdentityState.RESOLVED_BY_USER,
            identity=identity,
            candidates=resolution.identities,
            reason=f"resolved by explicit user decision to {resolution.default!r}",
        )

    def resolved_names(self) -> dict[str, str]:
        """``normalized name -> qualified canonical name`` for decided collisions."""
        return {
            key: resolution.default
            for key, resolution in self.config.collisions.items()
            if resolution.default
        }

    def alias_map(self) -> dict[str, str]:
        return dict(self.config.aliases)

    def unresolved(self) -> list[CollisionResolution]:
        return [r for r in self.config.collisions.values() if not r.resolved]

    # -- scaffolding -------------------------------------------------------

    def scaffold(
        self, ambiguity_index: Mapping[str, Sequence[str]], *, overwrite: bool = False
    ) -> tuple[int, int]:
        """Record each vault collision in the config **without deciding it**.

        Returns ``(added, skipped)``. Existing entries are preserved unless
        ``overwrite`` is set — a user's decision must survive re-scaffolding,
        or the feature would silently undo their work every time the vault
        gained a file.
        """
        added = skipped = 0
        for _, paths in sorted(ambiguity_index.items()):
            if not paths:
                continue
            name = PurePosixPath(paths[0]).stem
            if not overwrite and self.config.resolution_for(name) is not None:
                skipped += 1
                continue
            self.config.record_collision(
                CollisionResolution(
                    name=name,
                    identities=tuple(_identity_for_path(p) for p in paths),
                    default=None,  # deliberately undecided
                )
            )
            added += 1
        log.info("identity_scaffold", added=added, skipped=skipped)
        return added, skipped

    def decide(
        self, name: str, qualified_name: str, *, by: str = "cli"
    ) -> CollisionResolution:
        """Record the user's decision for a collision."""
        resolution = self.config.resolution_for(name)
        if resolution is None:
            raise KeyError(
                f"no documented collision named {name!r}; run `forge identity scaffold` first"
            )
        available = {i.qualified_name for i in resolution.identities}
        if qualified_name not in available:
            raise ValueError(
                f"{qualified_name!r} is not an identity of {name!r}; choose one of {sorted(available)}"
            )
        decided = CollisionResolution(
            name=resolution.name,
            identities=resolution.identities,
            default=qualified_name,
            decided_by=by,
            decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.config.record_collision(decided)
        log.info("identity_decided", name=resolution.name, default=qualified_name)
        return decided

    def clear(self, name: str) -> CollisionResolution:
        """Return a decided collision to the undecided state."""
        resolution = self.config.resolution_for(name)
        if resolution is None:
            raise KeyError(f"no documented collision named {name!r}")
        cleared = CollisionResolution(name=resolution.name, identities=resolution.identities)
        self.config.record_collision(cleared)
        return cleared


def _identity_for_path(path: str) -> ConceptIdentity:
    """Derive a suggested identity from a vault path.

    The namespace is *suggested* from the containing folder, which is where
    the corpus already encodes the distinction (`01_Patterns/Heap.md` vs
    `03_DataStructures/Heap.md`). The user is free to rename it — Forge does
    not invent their preferred vocabulary, it proposes the one their own
    folder structure already implies.
    """
    parts = PurePosixPath(path).parts
    folder = parts[-2] if len(parts) >= 2 else ""
    namespace, kind = _FOLDER_NAMESPACES.get(folder, (normalize(folder) or None, "concept"))
    return ConceptIdentity(
        canonical_name=PurePosixPath(path).stem,
        namespace=namespace,
        kind=kind,
        vault_path=path,
    )
