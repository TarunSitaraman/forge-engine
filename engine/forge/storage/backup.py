"""Backup and restore for the derived store.

**Why this exists at all.** ADR-001's R5: *derived state that Markdown cannot
express is lost on rebuild*. Forge's rule is that everything it derives lives
in `.forge/` and is rebuildable from scratch, and that rule is *almost* true.
What a rebuild from Markdown cannot reproduce:

* **Decisions.** A proposal you rejected, and why. Re-running extraction
  proposes it again; nothing in the Markdown records that you said no.
* **History.** The revision log, which is the entire answer to "what changed in
  my understanding". Rebuilt state has no past.
* **Questions.** You asked them. They are not in the vault.
* **Syntheses**, and whether they have gone stale.
* **Model-derived claims**, which a rebuild can only *re-derive*, differently:
  a different model, or the same model on a different day, produces different
  text. The evidence is reproducible; the wording is not.

So the store is not disposable, and this module is the mitigation R5 asks for.

**A backup is a consistent copy, not a dump.** It uses SQLite's own backup API,
which takes a copy that is transactionally consistent even if something is
writing, rather than copying the file underneath a live WAL and hoping. The
manifest beside it records what was copied so a restore can refuse a file that
does not match the schema it is being restored into.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sqlite_store import SCHEMA_VERSION, SqliteStore

MANIFEST_NAME = "manifest.json"
DATABASE_NAME = "forge.db"
BACKUP_FORMAT = "forge-backup/1"


class BackupError(RuntimeError):
    """A backup could not be made, or a restore was refused."""


@dataclass
class BackupManifest:
    """What is in a backup, and enough to refuse restoring it into the wrong place."""

    format: str = BACKUP_FORMAT
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION
    forge_version: str = "unknown"
    source_db: str = ""
    sha256: str = ""
    bytes: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "forge_version": self.forge_version,
            "source_db": self.source_db,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "counts": self.counts,
        }

    @classmethod
    def load(cls, path: Path) -> BackupManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            format=raw.get("format", ""),
            created_at=raw.get("created_at", ""),
            schema_version=int(raw.get("schema_version", 0)),
            forge_version=raw.get("forge_version", "unknown"),
            source_db=raw.get("source_db", ""),
            sha256=raw.get("sha256", ""),
            bytes=int(raw.get("bytes", 0)),
            counts=dict(raw.get("counts", {})),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forge_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("forge-kb")
    except PackageNotFoundError:  # pragma: no cover - running from a checkout
        return "unknown"


def create_backup(db_path: Path | str, destination: Path | str) -> BackupManifest:
    """Copy the store at `db_path` into `destination`, with a manifest.

    Uses `sqlite3.Connection.backup`, so the copy is transactionally consistent
    even under a concurrent writer. Copying the file directly would capture a
    WAL mid-write and produce a backup that restores to a torn database, which
    is worse than no backup because it looks like one.
    """
    db_path = Path(db_path)
    destination = Path(destination)
    if not db_path.is_file():
        raise BackupError(f"no store to back up at {db_path}")

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / DATABASE_NAME

    source = sqlite3.connect(str(db_path))
    try:
        copy = sqlite3.connect(str(target))
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()

    store = SqliteStore(target)
    try:
        counts = store.counts()
        schema = store.schema_version or SCHEMA_VERSION
    finally:
        store.close()

    manifest = BackupManifest(
        created_at=datetime.now(UTC).isoformat(),
        schema_version=schema,
        forge_version=_forge_version(),
        source_db=str(db_path),
        sha256=_sha256(target),
        bytes=target.stat().st_size,
        counts=counts,
    )
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def restore_backup(
    source: Path | str, db_path: Path | str, *, force: bool = False
) -> BackupManifest:
    """Restore a backup over the store at `db_path`.

    Refuses rather than guesses in three cases, because each of them destroys
    knowledge if it goes ahead:

    * a backup whose contents do not match its manifest's checksum, which means
      it was corrupted or edited since it was written;
    * a backup written by a **newer** schema than this build understands, which
      would be a silent downgrade;
    * an existing store, unless `force`. Restoring is destructive by nature and
      the caller has to say so.
    """
    source = Path(source)
    db_path = Path(db_path)

    manifest_path = source / MANIFEST_NAME
    archive = source / DATABASE_NAME
    if not manifest_path.is_file() or not archive.is_file():
        raise BackupError(
            f"{source} is not a Forge backup: expected {MANIFEST_NAME} and {DATABASE_NAME}"
        )

    manifest = BackupManifest.load(manifest_path)
    if manifest.format != BACKUP_FORMAT:
        raise BackupError(
            f"unknown backup format {manifest.format!r}; this build writes {BACKUP_FORMAT}"
        )
    actual = _sha256(archive)
    if manifest.sha256 and actual != manifest.sha256:
        raise BackupError(
            "backup does not match its manifest checksum, so it has been corrupted "
            f"or edited since it was written (expected {manifest.sha256[:12]}, "
            f"found {actual[:12]}). Refusing to restore it over a working store."
        )
    if manifest.schema_version > SCHEMA_VERSION:
        raise BackupError(
            f"backup is schema v{manifest.schema_version} and this build understands "
            f"v{SCHEMA_VERSION}. Restoring would be a silent downgrade; upgrade Forge "
            "instead."
        )
    if db_path.exists() and not force:
        raise BackupError(
            f"{db_path} already exists. Restoring replaces it and everything in it; "
            "pass force to say that is what you want."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear the WAL sidecars too: leaving a stale -wal beside a replaced
    # database is how a restore appears to succeed and then serves the old
    # data back.
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    shutil.copy2(archive, db_path)

    # An upgrade in place, so a restore from an older schema is usable at once
    # rather than failing on the next command.
    store = SqliteStore(db_path)
    try:
        store.initialize()
    finally:
        store.close()
    return manifest
