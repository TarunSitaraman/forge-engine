"""SQLite implementation of the Phase 1 storage protocols.

Why SQLite, and why it is not a commitment:

* **stdlib.** No dependency, no service, no container. Phase 1 must run on a
  laptop with nothing installed.
* **Transactional.** Provenance and revision writes must be atomic with the
  objects they describe, or history can end up describing a state that was
  never committed.
* **Single file, deletable.** ``.forge/forge.db`` is derived state. Deleting it
  loses nothing that cannot be rebuilt from the vault — which is exactly the
  property the architecture promises.
* **Replaceable.** Everything upstream depends on the protocols in
  :mod:`forge.storage.base`, not on this module.

Entities are stored as JSON documents in typed tables, with the few fields
needed for lookup and joins promoted to real columns. At ~600 sources this is
comfortably fast, and it means a domain model change does not require a schema
migration during Phase 1's high-churn period.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from ..domain import (
    Claim,
    ClaimLink,
    ClaimStatus,
    Concept,
    Document,
    EntityType,
    EvidenceLink,
    Revision,
    Source,
    Span,
    record_change,
    record_create,
    record_invalidate,
    record_supersede,
    utc_now,
    validate_claim,
    validate_supersession,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,
    locator      TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    trust_tier   TEXT NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    data         TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);

CREATE TABLE IF NOT EXISTS spans (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    data         TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_spans_document ON spans(document_id, ordinal);

CREATE TABLE IF NOT EXISTS concepts (
    id             TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL,
    vault_path     TEXT,
    data           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_concepts_kind ON concepts(kind);

CREATE TABLE IF NOT EXISTS claims (
    id                 TEXT PRIMARY KEY,
    subject_concept_id TEXT,
    tier               TEXT NOT NULL,
    status             TEXT NOT NULL,
    data               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_concept ON claims(subject_concept_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);

CREATE TABLE IF NOT EXISTS evidence_links (
    id       TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    span_id  TEXT NOT NULL,
    relation TEXT NOT NULL,
    data     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence_links(claim_id);

CREATE TABLE IF NOT EXISTS claim_links (
    id      TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id   TEXT NOT NULL,
    type    TEXT NOT NULL,
    active  INTEGER NOT NULL DEFAULT 1,
    data    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_from ON claim_links(from_id);
CREATE INDEX IF NOT EXISTS idx_links_to ON claim_links(to_id);

CREATE TABLE IF NOT EXISTS revisions (
    id          TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    op          TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    seq         INTEGER,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revisions_entity ON revisions(entity_type, entity_id, seq);
CREATE INDEX IF NOT EXISTS idx_revisions_seq ON revisions(seq);
"""


class SqliteStore:
    """Concrete :class:`forge.storage.base.Store`."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def reset(self) -> None:
        """Drop every table. Derived state only — nothing unrecoverable here."""
        with self._conn:
            for table in (
                "revisions",
                "claim_links",
                "evidence_links",
                "claims",
                "concepts",
                "spans",
                "documents",
                "sources",
                "meta",
            ):
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.initialize()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- sources -----------------------------------------------------------

    def put_source(self, source: Source) -> None:
        existing = self.get_source(source.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources(id, locator, kind, content_hash, trust_tier, data)"
                " VALUES(?,?,?,?,?,?)",
                (
                    source.id,
                    source.locator,
                    source.kind.value,
                    source.content_hash,
                    source.trust_tier.value,
                    _dump(source),
                ),
            )
            if existing is None:
                self._append(record_create(EntityType.SOURCE, source.id, _as_dict(source)))
            elif existing.content_hash != source.content_hash:
                self._append(
                    record_change(
                        EntityType.SOURCE,
                        source.id,
                        _as_dict(existing),
                        _as_dict(source),
                        cause=source.content_hash,
                        note="content hash changed",
                    )
                )

    def get_source(self, source_id: str) -> Source | None:
        row = self._one("SELECT data FROM sources WHERE id = ?", (source_id,))
        return Source.model_validate_json(row["data"]) if row else None

    def get_source_by_locator(self, locator: str) -> Source | None:
        row = self._one("SELECT data FROM sources WHERE locator = ?", (locator,))
        return Source.model_validate_json(row["data"]) if row else None

    def list_sources(self) -> Sequence[Source]:
        rows = self._all("SELECT data FROM sources ORDER BY locator")
        return [Source.model_validate_json(r["data"]) for r in rows]

    def delete_source(self, source_id: str) -> None:
        existing = self.get_source(source_id)
        if existing is None:
            return
        with self._conn:
            self._conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            self._append(
                record_invalidate(
                    EntityType.SOURCE,
                    source_id,
                    _as_dict(existing),
                    note="source removed from vault",
                )
            )

    # -- documents / spans -------------------------------------------------

    def put_document(self, document: Document) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents(id, source_id, version, content_hash, data)"
                " VALUES(?,?,?,?,?)",
                (
                    document.id,
                    document.source_id,
                    document.version,
                    document.content_hash,
                    _dump(document),
                ),
            )

    def get_document(self, document_id: str) -> Document | None:
        row = self._one("SELECT data FROM documents WHERE id = ?", (document_id,))
        return Document.model_validate_json(row["data"]) if row else None

    def documents_for_source(self, source_id: str) -> Sequence[Document]:
        rows = self._all(
            "SELECT data FROM documents WHERE source_id = ? ORDER BY version", (source_id,)
        )
        return [Document.model_validate_json(r["data"]) for r in rows]

    def put_spans(self, spans: Sequence[Span]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans(id, document_id, ordinal, content_hash, data)"
                " VALUES(?,?,?,?,?)",
                [(s.id, s.document_id, s.ordinal, s.content_hash, _dump(s)) for s in spans],
            )

    def get_span(self, span_id: str) -> Span | None:
        row = self._one("SELECT data FROM spans WHERE id = ?", (span_id,))
        return Span.model_validate_json(row["data"]) if row else None

    def spans_for_document(self, document_id: str) -> Sequence[Span]:
        rows = self._all(
            "SELECT data FROM spans WHERE document_id = ? ORDER BY ordinal", (document_id,)
        )
        return [Span.model_validate_json(r["data"]) for r in rows]

    # -- concepts ----------------------------------------------------------

    def put_concept(self, concept: Concept) -> None:
        existing = self.get_concept(concept.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO concepts(id, canonical_name, kind, vault_path, data)"
                " VALUES(?,?,?,?,?)",
                (
                    concept.id,
                    concept.canonical_name,
                    concept.kind.value,
                    concept.vault_path,
                    _dump(concept),
                ),
            )
            if existing is None:
                self._append(record_create(EntityType.CONCEPT, concept.id, _as_dict(concept)))

    def get_concept(self, concept_id: str) -> Concept | None:
        row = self._one("SELECT data FROM concepts WHERE id = ?", (concept_id,))
        return Concept.model_validate_json(row["data"]) if row else None

    def get_concept_by_name(self, canonical_name: str) -> Concept | None:
        row = self._one("SELECT data FROM concepts WHERE canonical_name = ?", (canonical_name,))
        return Concept.model_validate_json(row["data"]) if row else None

    def list_concepts(self) -> Sequence[Concept]:
        rows = self._all("SELECT data FROM concepts ORDER BY canonical_name")
        return [Concept.model_validate_json(r["data"]) for r in rows]

    # -- claims ------------------------------------------------------------

    def put_claim(self, claim: Claim, evidence: Sequence[EvidenceLink] = ()) -> None:
        # Enforcement point: a claim requiring evidence cannot be stored
        # without it, no matter which caller is asking.
        validate_claim(claim, evidence)
        existing = self.get_claim(claim.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO claims(id, subject_concept_id, tier, status, data)"
                " VALUES(?,?,?,?,?)",
                (
                    claim.id,
                    claim.subject_concept_id,
                    claim.provenance.tier.value,
                    claim.status.value,
                    _dump(claim),
                ),
            )
            for ev in evidence:
                self._conn.execute(
                    "INSERT OR REPLACE INTO evidence_links(id, claim_id, span_id, relation, data)"
                    " VALUES(?,?,?,?,?)",
                    (ev.id, ev.claim_id, ev.span_id, ev.relation.value, _dump(ev)),
                )
            if existing is None:
                self._append(record_create(EntityType.CLAIM, claim.id, _as_dict(claim)))

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self._one("SELECT data FROM claims WHERE id = ?", (claim_id,))
        return Claim.model_validate_json(row["data"]) if row else None

    def list_claims(self) -> Sequence[Claim]:
        rows = self._all("SELECT data FROM claims ORDER BY id")
        return [Claim.model_validate_json(r["data"]) for r in rows]

    def evidence_for_claim(self, claim_id: str) -> Sequence[EvidenceLink]:
        rows = self._all("SELECT data FROM evidence_links WHERE claim_id = ?", (claim_id,))
        return [EvidenceLink.model_validate_json(r["data"]) for r in rows]

    def supersede_claim(self, old_id: str, new_claim: Claim, *, cause: str | None = None) -> None:
        """Non-destructive replacement — Principle 11's enforcement point.

        The old claim is retained, marked SUPERSEDED, and a SUPERSEDE revision
        records both states.
        """
        old = self.get_claim(old_id)
        if old is None:
            raise KeyError(f"cannot supersede unknown claim {old_id}")

        before = _as_dict(old)
        retired = old.model_copy(
            update={
                "status": ClaimStatus.SUPERSEDED,
                "superseded_by": new_claim.id,
                "valid_to": utc_now(),
            }
        )
        validate_supersession(retired, new_claim)

        with self._conn:
            self._conn.execute(
                "UPDATE claims SET status = ?, data = ? WHERE id = ?",
                (retired.status.value, _dump(retired), retired.id),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO claims(id, subject_concept_id, tier, status, data)"
                " VALUES(?,?,?,?,?)",
                (
                    new_claim.id,
                    new_claim.subject_concept_id,
                    new_claim.provenance.tier.value,
                    new_claim.status.value,
                    _dump(new_claim),
                ),
            )
            self._append(
                record_supersede(
                    EntityType.CLAIM,
                    old_id,
                    before,
                    _as_dict(retired),
                    superseded_by=new_claim.id,
                    cause=cause or new_claim.id,
                )
            )
            self._append(record_create(EntityType.CLAIM, new_claim.id, _as_dict(new_claim)))

    # -- links -------------------------------------------------------------

    def put_link(self, link: ClaimLink) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO claim_links(id, from_id, to_id, type, active, data)"
                " VALUES(?,?,?,?,?,?)",
                (
                    link.id,
                    link.from_id,
                    link.to_id,
                    link.type.value,
                    int(link.active),
                    _dump(link),
                ),
            )

    def links_from(self, entity_id: str) -> Sequence[ClaimLink]:
        rows = self._all("SELECT data FROM claim_links WHERE from_id = ?", (entity_id,))
        return [ClaimLink.model_validate_json(r["data"]) for r in rows]

    def links_to(self, entity_id: str) -> Sequence[ClaimLink]:
        rows = self._all("SELECT data FROM claim_links WHERE to_id = ?", (entity_id,))
        return [ClaimLink.model_validate_json(r["data"]) for r in rows]

    def count_links(self) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM claim_links")["n"])  # type: ignore[index]

    # -- revisions ---------------------------------------------------------

    def append_revision(self, revision: Revision) -> None:
        with self._conn:
            self._append(revision)

    def _append(self, revision: Revision) -> None:
        """Append within an existing transaction.

        ``seq`` is a monotonic counter giving revisions a total order that does
        not depend on timestamp resolution — two revisions written in the same
        millisecond must still be orderable.
        """
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS nxt FROM revisions").fetchone()
        self._conn.execute(
            "INSERT OR REPLACE INTO revisions(id, entity_type, entity_id, op, created_at, seq, data)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                revision.id,
                revision.entity_type.value,
                revision.entity_id,
                revision.op.value,
                revision.created_at.isoformat(),
                int(row["nxt"]),
                _dump(revision),
            ),
        )

    def revisions_for(self, entity_type: EntityType, entity_id: str) -> Sequence[Revision]:
        rows = self._all(
            "SELECT data FROM revisions WHERE entity_type = ? AND entity_id = ? ORDER BY seq",
            (entity_type.value, entity_id),
        )
        return [Revision.model_validate_json(r["data"]) for r in rows]

    def recent_revisions(self, limit: int = 50) -> Sequence[Revision]:
        rows = self._all("SELECT data FROM revisions ORDER BY seq DESC LIMIT ?", (limit,))
        return [Revision.model_validate_json(r["data"]) for r in rows]

    def count_revisions(self) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM revisions")["n"])  # type: ignore[index]

    # -- helpers -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            table: int(self._one(f"SELECT COUNT(*) AS n FROM {table}")["n"])  # type: ignore[index]
            for table in (
                "sources",
                "documents",
                "spans",
                "concepts",
                "claims",
                "evidence_links",
                "claim_links",
                "revisions",
            )
        }

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()


def _dump(model: Any) -> str:
    return model.model_dump_json()


def _as_dict(model: Any) -> dict[str, Any]:
    """Round-trip through JSON so revision snapshots are plain, storable data."""
    return json.loads(model.model_dump_json())  # type: ignore[no-any-return]
