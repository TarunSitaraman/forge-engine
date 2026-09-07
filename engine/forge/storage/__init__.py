"""Storage layer: protocols plus the Phase 1 SQLite implementation."""

from .base import KnowledgeStore, RevisionStore, SourceStore, Store
from .backup import BackupError, BackupManifest, create_backup, restore_backup
from .sqlite_store import SCHEMA_VERSION, SqliteStore

__all__ = [
    "Store",
    "SourceStore",
    "KnowledgeStore",
    "RevisionStore",
    "SqliteStore",
    "SCHEMA_VERSION",
    "BackupError",
    "BackupManifest",
    "create_backup",
    "restore_backup",
]
