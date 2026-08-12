"""Indexing pipeline: index -> detect changes -> persist -> report.

This is the seam the CLI drives. It is entirely deterministic and makes **zero
LLM calls**, by construction: nothing in this module, or anything it imports,
can reach :mod:`forge.llm`.

The ``llm_calls`` counter on :class:`IndexResult` is reported on every run so
the "unchanged corpus costs zero LLM calls" property is observable in normal
operation, not only under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..domain import ChangeStatus
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore
from .conventions import analyze_conventions
from .diagnostics import frontmatter_report, link_report
from .indexer import CorpusIndexer, detect_changes
from .model import ChangeSet, CorpusIndex
from .stats import compute_stats

log = get_logger(__name__)


@dataclass
class IndexResult:
    index: CorpusIndex
    changes: ChangeSet
    persisted_sources: int = 0
    persisted_documents: int = 0
    persisted_spans: int = 0
    #: Always 0 in Phase 1 — indexing is fully deterministic.
    llm_calls: int = 0
    reports_written: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "files_indexed": self.index.file_count,
            "fingerprint": self.index.fingerprint(),
            "changes": self.changes.summary(),
            "persisted": {
                "sources": self.persisted_sources,
                "documents": self.persisted_documents,
                "spans": self.persisted_spans,
            },
            "llm_calls": self.llm_calls,
            "duration_seconds": self.index.duration_seconds,
        }


class IndexPipeline:
    def __init__(self, settings: Settings, store: SqliteStore) -> None:
        self.settings = settings
        self.store = store
        self.indexer = CorpusIndexer(settings)

    def run(self, *, persist: bool = True, write_reports: bool = True) -> IndexResult:
        index = self.indexer.build_index()

        previous = {s.locator: s.content_hash for s in self.store.list_sources()}
        changes = detect_changes(index.files, previous)

        result = IndexResult(index=index, changes=changes)

        if persist:
            self._persist(index, changes, result)

        if write_reports:
            result.reports_written = self._write_reports(index)

        log.info("index_run_complete", **{k: v for k, v in result.summary().items() if k != "changes"})
        return result

    # -- persistence -------------------------------------------------------

    def _persist(self, index: CorpusIndex, changes: ChangeSet, result: IndexResult) -> None:
        """Persist only what changed.

        Unchanged sources are skipped entirely — no re-parse, no re-span, no
        write. This is the mechanism behind incremental processing.
        """
        by_path = index.by_path()
        sources_by_path = {s.locator: s for s in self.indexer.to_sources(index)}

        for change in changes.requires_processing:
            indexed = by_path.get(change.path)
            source = sources_by_path.get(change.path)
            if indexed is None or source is None:  # pragma: no cover - defensive
                continue

            self.store.put_source(source)
            result.persisted_sources += 1

            document, spans = self.indexer.to_document_and_spans(indexed, source)
            self.store.put_document(document)
            result.persisted_documents += 1
            if spans:
                self.store.put_spans(spans)
                result.persisted_spans += len(spans)

        # A file removed from the vault is invalidated in derived state, with a
        # revision recording it. The Markdown itself is never touched.
        for change in changes.of(ChangeStatus.DELETED):
            existing = self.store.get_source_by_locator(change.path)
            if existing is not None:
                self.store.delete_source(existing.id)

    # -- reports -----------------------------------------------------------

    def _write_reports(self, index: CorpusIndex) -> list[str]:
        out_dir = self.settings.reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = datetime.now(timezone.utc).isoformat()

        reports: dict[str, dict[str, Any]] = {
            "corpus-stats": compute_stats(index).to_dict(),
            "frontmatter-report": frontmatter_report(index).to_dict(),
            "link-report": link_report(index).to_dict(),
            "convention-report": analyze_conventions(index).to_dict(),
        }

        written: list[str] = []
        for name, payload in reports.items():
            path = out_dir / f"{name}.json"
            body = {
                "generated_at": generated,
                "vault_path": index.vault_path,
                "fingerprint": index.fingerprint(),
                "report": payload,
            }
            path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(str(path))
        return written


def load_store(settings: Settings) -> SqliteStore:
    store = SqliteStore(settings.db_path)
    store.initialize()
    return store


def previous_hashes(store: SqliteStore) -> dict[str, str]:
    return {s.locator: s.content_hash for s in store.list_sources()}


def reports_path(settings: Settings, name: str) -> Path:
    return settings.reports_dir / f"{name}.json"
