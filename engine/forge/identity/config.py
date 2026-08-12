"""Concept identity configuration — the user's explicit naming decisions.

The corpus contains three genuine collisions: `Heap`, `Binary Search`, and
`Trie` each exist twice, once as a pattern and once as an algorithm or data
structure. Phase 2 correctly refused to guess between them. Phase 3 gives the
user a way to *decide*, and makes that decision durable.

Three properties matter:

* **It is configuration, not code.** The decisions live in a versioned YAML
  file in the repository, reviewable in a diff, editable without touching the
  matcher. Burying them in Python would make the user's naming choices
  invisible and untestable.
* **The matcher learns from it.** A resolved collision stops being ambiguous
  and starts resolving to whichever concept the user named — with
  ``IdentityState.RESOLVED_BY_USER`` recorded, so the reason is never lost.
* **Forge never invents an entry.** No default resolutions ship. An unresolved
  collision stays ambiguous forever until a human says otherwise, and the file
  starts out documenting the collisions rather than deciding them.

No LLM is involved at any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..logging import get_logger
from ..parsing.links import normalize

log = get_logger(__name__)

CONFIG_VERSION = 1
DEFAULT_CONFIG_PATH = Path("config") / "concept-identity.yaml"


class IdentityConfigError(Exception):
    """Raised when the identity configuration is malformed.

    Fatal rather than best-effort: a misread identity decision would silently
    change what the matcher believes, which is worse than refusing to start.
    """


@dataclass(frozen=True)
class ConceptIdentity:
    """One canonical concept the user has explicitly named."""

    canonical_name: str
    #: Namespace/context that disambiguates it, e.g. "pattern" or "data_structure".
    namespace: str | None = None
    kind: str = "concept"
    aliases: tuple[str, ...] = ()
    #: Vault path this identity corresponds to, when there is one.
    vault_path: str | None = None
    note: str | None = None

    @property
    def qualified_name(self) -> str:
        """Name including its namespace, e.g. ``pattern/Heap``.

        The namespace is what lets two concepts legitimately share a bare name
        without either one being wrong.
        """
        return f"{self.namespace}/{self.canonical_name}" if self.namespace else self.canonical_name

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"canonical_name": self.canonical_name, "kind": self.kind}
        if self.namespace:
            out["namespace"] = self.namespace
        if self.aliases:
            out["aliases"] = list(self.aliases)
        if self.vault_path:
            out["vault_path"] = self.vault_path
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class CollisionResolution:
    """A user's decision about one colliding bare name."""

    #: The colliding bare name, e.g. "Heap".
    name: str
    #: The distinct identities this name resolves to.
    identities: tuple[ConceptIdentity, ...]
    #: Which identity a bare, unqualified mention means. ``None`` means the
    #: user deliberately left it ambiguous — a valid, recorded decision.
    default: str | None = None
    decided_by: str | None = None
    decided_at: str | None = None

    @property
    def resolved(self) -> bool:
        """True only when a bare mention has a defined meaning."""
        return self.default is not None

    def identity_for(self, qualified_name: str) -> ConceptIdentity | None:
        for identity in self.identities:
            if identity.qualified_name == qualified_name:
                return identity
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "identities": [i.to_dict() for i in self.identities],
        }
        if self.default:
            out["default"] = self.default
        if self.decided_by:
            out["decided_by"] = self.decided_by
        if self.decided_at:
            out["decided_at"] = self.decided_at
        return out


@dataclass
class IdentityConfig:
    """The whole identity configuration."""

    version: int = CONFIG_VERSION
    collisions: dict[str, CollisionResolution] = field(default_factory=dict)
    #: Standalone aliases: alias -> canonical name, independent of collisions.
    aliases: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    # -- lookup ------------------------------------------------------------

    def resolution_for(self, name: str) -> CollisionResolution | None:
        return self.collisions.get(normalize(name))

    def alias_target(self, name: str) -> str | None:
        return self.aliases.get(normalize(name))

    def is_resolved(self, name: str) -> bool:
        resolution = self.resolution_for(name)
        return resolution is not None and resolution.resolved

    def known_names(self) -> set[str]:
        out = set(self.collisions)
        out.update(self.aliases)
        return out

    # -- mutation ----------------------------------------------------------

    def record_collision(self, resolution: CollisionResolution) -> None:
        self.collisions[normalize(resolution.name)] = resolution

    def record_alias(self, alias: str, canonical: str) -> None:
        self.aliases[normalize(alias)] = canonical

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "collisions": [r.to_dict() for r in _sorted_resolutions(self.collisions.values())],
            "aliases": dict(sorted(self.aliases.items())),
        }

    def save(self, path: Path | None = None) -> Path:
        """Write the configuration back, deterministically ordered.

        Sorted so that regenerating the file produces a clean diff rather than
        reshuffled lines — this file is meant to be reviewed in a pull request.
        """
        target = Path(path or self.path or DEFAULT_CONFIG_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render(self.to_dict()), encoding="utf-8")
        self.path = target
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> IdentityConfig:
        """Load configuration, returning an empty one when absent.

        Absence is normal — a vault with no decided collisions is a valid
        state, and the matcher simply keeps reporting them as ambiguous.
        """
        target = Path(path or DEFAULT_CONFIG_PATH)
        if not target.is_file():
            return cls(path=target)

        try:
            raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise IdentityConfigError(f"{target}: invalid YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise IdentityConfigError(f"{target}: expected a mapping, got {type(raw).__name__}")

        config = cls(version=int(raw.get("version", CONFIG_VERSION)), path=target)

        for entry in raw.get("collisions") or []:
            config.record_collision(_parse_resolution(entry, target))

        aliases = raw.get("aliases") or {}
        if not isinstance(aliases, dict):
            raise IdentityConfigError(f"{target}: 'aliases' must be a mapping")
        for alias, canonical in aliases.items():
            config.record_alias(str(alias), str(canonical))

        log.info(
            "identity_config_loaded",
            path=str(target),
            collisions=len(config.collisions),
            resolved=sum(1 for r in config.collisions.values() if r.resolved),
            aliases=len(config.aliases),
        )
        return config


# -- helpers ---------------------------------------------------------------


def _parse_resolution(entry: Any, path: Path) -> CollisionResolution:
    if not isinstance(entry, dict) or "name" not in entry:
        raise IdentityConfigError(f"{path}: each collision needs a 'name' key; got {entry!r}")

    identities = tuple(_parse_identity(i, path) for i in entry.get("identities") or ())
    if not identities:
        raise IdentityConfigError(f"{path}: collision {entry['name']!r} lists no identities")

    default = entry.get("default")
    if default is not None:
        qualified = {i.qualified_name for i in identities}
        if default not in qualified:
            raise IdentityConfigError(
                f"{path}: collision {entry['name']!r} defaults to {default!r}, which is not one "
                f"of its identities ({sorted(qualified)})"
            )

    return CollisionResolution(
        name=str(entry["name"]),
        identities=identities,
        default=default,
        decided_by=entry.get("decided_by"),
        decided_at=entry.get("decided_at"),
    )


def _parse_identity(entry: Any, path: Path) -> ConceptIdentity:
    if not isinstance(entry, dict) or "canonical_name" not in entry:
        raise IdentityConfigError(
            f"{path}: each identity needs a 'canonical_name' key; got {entry!r}"
        )
    return ConceptIdentity(
        canonical_name=str(entry["canonical_name"]),
        namespace=entry.get("namespace"),
        kind=str(entry.get("kind", "concept")),
        aliases=tuple(str(a) for a in entry.get("aliases") or ()),
        vault_path=entry.get("vault_path"),
        note=entry.get("note"),
    )


def _sorted_resolutions(values: Iterable[CollisionResolution]) -> list[CollisionResolution]:
    return sorted(values, key=lambda r: r.name.casefold())


def _render(data: dict[str, Any]) -> str:
    header = (
        "# Forge — concept identity configuration\n"
        "#\n"
        "# Explicit user decisions about concept naming. Forge never writes a\n"
        "# resolution here on its own: an unresolved collision stays ambiguous\n"
        "# until a human decides, and that is a valid state to leave it in.\n"
        "#\n"
        "# Managed by `forge identity` — hand edits are fine and are preserved.\n"
        "#\n"
        "# `default` names which identity a bare, unqualified mention means.\n"
        "# Omitting it means \"still ambiguous, deliberately\".\n\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
