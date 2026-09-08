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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..domain import (
    Claim,
    ClaimLink,
    ClaimStatus,
    Concept,
    Document,
    EntityType,
    EvidenceLink,
    Proposal,
    ProposalStatus,
    ProposalType,
    Question,
    QuestionStatus,
    Revision,
    Source,
    Span,
    Synthesis,
    WorkflowRun,
    WorkflowStatus,
    record_change,
    record_create,
    record_invalidate,
    record_supersede,
    utc_now,
    validate_claim,
    validate_supersession,
)
from ..logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 6

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

-- Phase 3: `canonical_name` alone is NOT unique. Resolving the vault's
-- collisions means `pattern/Heap` and `data-structure/Heap` are two genuinely
-- different concepts that legitimately share a bare name; a UNIQUE constraint
-- on the name alone would force one to overwrite the other, which is exactly
-- the silent merge the whole design refuses.
CREATE TABLE IF NOT EXISTS concepts (
    id             TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    namespace      TEXT,
    kind           TEXT NOT NULL,
    vault_path     TEXT,
    data           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_concepts_kind ON concepts(kind);
CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_qualified
    ON concepts(canonical_name, IFNULL(namespace, ''));

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

-- ---------------------------------------------------------------- Phase 2

CREATE TABLE IF NOT EXISTS proposals (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    status     TEXT NOT NULL,
    safety     TEXT NOT NULL,
    target     TEXT NOT NULL,
    source_id  TEXT,
    created_at TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, type);
CREATE INDEX IF NOT EXISTS idx_proposals_source ON proposals(source_id);

-- Derivation cache. Keyed on everything that can invalidate a derived result:
-- source content, processor version, model, and prompt/schema version. If any
-- component changes the key changes, and the work is redone.
CREATE TABLE IF NOT EXISTS derivations (
    key          TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    source_id    TEXT,
    content_hash TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_derivations_source ON derivations(source_id, kind);

-- Lexical search over spans. FTS5 is stdlib-available and removes any need for
-- an external search service at this corpus size.
CREATE VIRTUAL TABLE IF NOT EXISTS span_fts USING fts5(
    span_id UNINDEXED,
    document_id UNINDEXED,
    text,
    tokenize = 'unicode61'
);

-- Optional embeddings. Absent rows simply mean vector search is unavailable;
-- lexical retrieval and deterministic matching continue regardless.
CREATE TABLE IF NOT EXISTS embeddings (
    owner_type TEXT NOT NULL,
    owner_id   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_type, owner_id, model)
);

-- ---------------------------------------------------------------- Phase 4

-- Knowledge-evolution runs. Distinct from the LangGraph checkpoint: a
-- checkpoint is resumption state and may be pruned, while this is the durable
-- answer to "why did Forge propose this?". Storing the run here also means
-- inspection works with the orchestrator uninstalled.
CREATE TABLE IF NOT EXISTS workflows (
    id          TEXT PRIMARY KEY,
    source_id   TEXT,
    status      TEXT NOT NULL,
    provider_id TEXT,
    model_id    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_source ON workflows(source_id);

-- Phase 9. `questions` and `syntheses` are stored; KnowledgeGap deliberately
-- is not, because a stored gap is wrong the moment the missing claim arrives
-- and nothing would notice. Gaps are recomputed from the graph on demand.
CREATE TABLE IF NOT EXISTS questions (
    id         TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);

CREATE TABLE IF NOT EXISTS syntheses (
    id           TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,
    scope_id     TEXT,
    stale        INTEGER NOT NULL DEFAULT 0,
    generated_at TEXT NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_syntheses_scope ON syntheses(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_syntheses_stale ON syntheses(stale);

-- One row per constituent claim, holding the claim's state *at generation*.
-- Staleness is then a comparison rather than a judgement: if the claim differs
-- from what is recorded here, the synthesis was written from something that
-- has since moved.
--
-- The canonical model says "status or confidence". Forge has no confidences:
-- `forge.evolution.impact` refuses to invent one, because a number a model
-- emits about its own certainty is not a measurement. So the fingerprint is
-- what a claim actually has and what actually changes its meaning: its status,
-- its tier, its text, and whether it has been superseded. That is strictly
-- more than status alone, and none of it is invented.
CREATE TABLE IF NOT EXISTS synthesis_claims (
    synthesis_id TEXT NOT NULL REFERENCES syntheses(id) ON DELETE CASCADE,
    claim_id     TEXT NOT NULL,
    status       TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    PRIMARY KEY (synthesis_id, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_synthesis_claims_claim ON synthesis_claims(claim_id);

-- Claims that answer a question. The edge is a human's or a workflow's
-- assertion that this claim bears on that question, kept separate from
-- claim_links because its endpoints are different kinds of thing.
CREATE TABLE IF NOT EXISTS question_answers (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    claim_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (question_id, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_question_answers_claim ON question_answers(claim_id);

-- How many pages link to each concept's page, counted over EVERY page in the
-- vault, including the ones that are not concepts and so have no node in the
-- graph.
--
-- It exists because the graph cannot answer "does anything point here?".
-- Edges run between concept pages, and `bootstrap.is_concept_page` excludes
-- `_index.md` hubs and other navigation as not-concepts — correctly, a hub is
-- not a concept. But a vault that links hub-and-spoke puts all of its inbound
-- links on exactly those pages: reviewed on the real corpus 2026-09-07, 115
-- links pointed at the 72 concepts the gap report called isolated, and not one
-- came from a page the graph counts. Without this table there is no way to
-- tell a page nothing links to from a page linked only from its index.
--
-- Written by `forge bootstrap --apply`, from the same index pass that builds
-- the edges, so it is exactly as fresh as the graph and no fresher.
CREATE TABLE IF NOT EXISTS concept_inbound_links (
    concept_id TEXT PRIMARY KEY REFERENCES concepts(id) ON DELETE CASCADE,
    inbound    INTEGER NOT NULL,
    sources    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
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
        """Create or upgrade the schema. Idempotent.

        Phase 1 -> Phase 2 is a purely additive migration: every v1 table is
        untouched and the new tables are created alongside. Upgrading a v1
        database preserves all of its sources, documents, spans, and revisions.
        """
        previous = self.schema_version

        # Structural migrations run BEFORE the schema script. The v3 script
        # creates an index over `concepts.namespace`, which does not exist on a
        # v2 table — and CREATE TABLE IF NOT EXISTS will not add it. The table
        # has to be reshaped first, or initialize() fails on an upgrade.
        if previous and previous < SCHEMA_VERSION:
            self._migrate_structure(previous)

        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

        if previous and previous < SCHEMA_VERSION:
            self._migrate_derived(previous)

    def _migrate_structure(self, previous: int) -> None:
        """Schema changes that must precede the idempotent schema script."""
        if previous < 3:
            self._migrate_concepts_to_namespaced()

    def _migrate_derived(self, previous: int) -> None:
        """Bring derived-only structures into line after a schema upgrade.

        The FTS index is derived from spans, so a v1 database arrives with it
        empty. Left alone that would make ``forge search`` silently return
        nothing on an upgraded database — a wrong answer rather than an error,
        which is the worse failure. Rebuilding is cheap and self-healing.
        """
        if previous < 2:
            spans = int(self._one("SELECT COUNT(*) AS n FROM spans")["n"])  # type: ignore[index]
            indexed = int(self._one("SELECT COUNT(*) AS n FROM span_fts")["n"])  # type: ignore[index]
            if spans and not indexed:
                self.rebuild_search_index()

    def _migrate_concepts_to_namespaced(self) -> None:
        """v2 -> v3: drop UNIQUE(canonical_name), add a namespace column.

        SQLite cannot alter a constraint in place, so the table is rebuilt.
        Existing rows are preserved and carry ``namespace = NULL``, which is
        correct: a concept created before namespaces existed was, by
        definition, not disambiguated.
        """
        columns = {
            row["name"] for row in self._all("PRAGMA table_info(concepts)")
        }
        if "namespace" in columns:
            return

        rows = self._all("SELECT id, canonical_name, kind, vault_path, data FROM concepts")
        with self._conn:
            self._conn.execute("ALTER TABLE concepts RENAME TO concepts_v2")
            self._conn.executescript(
                """
                CREATE TABLE concepts (
                    id             TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    namespace      TEXT,
                    kind           TEXT NOT NULL,
                    vault_path     TEXT,
                    data           TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_concepts_kind ON concepts(kind);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_qualified
                    ON concepts(canonical_name, IFNULL(namespace, ''));
                """
            )
            self._conn.executemany(
                "INSERT INTO concepts(id, canonical_name, namespace, kind, vault_path, data)"
                " VALUES(?,?,NULL,?,?,?)",
                [
                    (r["id"], r["canonical_name"], r["kind"], r["vault_path"], r["data"])
                    for r in rows
                ],
            )
            self._conn.execute("DROP TABLE concepts_v2")
        log.info("concepts_migrated_to_namespaced", rows=len(rows))

    def reset(self) -> None:
        """Drop every table. Derived state only — nothing unrecoverable here."""
        with self._conn:
            for table in (
                "workflows",
                "revisions",
                "claim_links",
                "evidence_links",
                "claims",
                "concepts",
                "spans",
                "documents",
                "sources",
                "proposals",
                "derivations",
                "embeddings",
                "span_fts",
                "meta",
            ):
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.initialize()

    @property
    def schema_version(self) -> int:
        """Stored schema version, or 0 when the database has no schema yet.

        Tolerates a missing ``meta`` table: ``sqlite3.connect`` creates the file
        on construction, so a brand-new database exists on disk but is empty.
        """
        try:
            row = self._one("SELECT value FROM meta WHERE key = 'schema_version'")
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

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
                # ON CONFLICT DO UPDATE, not INSERT OR REPLACE. SQLite implements
                # REPLACE as DELETE-then-INSERT, which fires ON DELETE CASCADE and
                # would silently destroy every document and span belonging to a
                # source the moment its content changed — exactly the historical
                # provenance Phase 2 must preserve.
                "INSERT INTO sources(id, locator, kind, content_hash, trust_tier, data)"
                " VALUES(?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   locator=excluded.locator, kind=excluded.kind,"
                "   content_hash=excluded.content_hash, trust_tier=excluded.trust_tier,"
                "   data=excluded.data",
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
                # Upsert for the same reason: documents cascade to spans.
                "INSERT INTO documents(id, source_id, version, content_hash, data)"
                " VALUES(?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   version=excluded.version, content_hash=excluded.content_hash,"
                "   data=excluded.data",
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
        """Persist spans and keep the lexical index in step.

        FTS rows are rewritten alongside the span in the same transaction, so
        the search index can never drift from the spans it indexes.
        """
        with self._conn:
            self._conn.executemany(
                "INSERT INTO spans(id, document_id, ordinal, content_hash, data)"
                " VALUES(?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   ordinal=excluded.ordinal, content_hash=excluded.content_hash,"
                "   data=excluded.data",
                [(s.id, s.document_id, s.ordinal, s.content_hash, _dump(s)) for s in spans],
            )
            ids = [s.id for s in spans]
            if ids:
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"DELETE FROM span_fts WHERE span_id IN ({placeholders})", ids
                )
            self._conn.executemany(
                "INSERT INTO span_fts(span_id, document_id, text) VALUES(?,?,?)",
                [(s.id, s.document_id, s.text) for s in spans],
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
                "INSERT INTO concepts(id, canonical_name, namespace, kind, vault_path, data)"
                " VALUES(?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   canonical_name=excluded.canonical_name, namespace=excluded.namespace,"
                "   kind=excluded.kind, vault_path=excluded.vault_path, data=excluded.data",
                (
                    concept.id,
                    concept.canonical_name,
                    concept.namespace,
                    concept.kind.value,
                    concept.vault_path,
                    _dump(concept),
                ),
            )
            if existing is None:
                self._append(record_create(EntityType.CONCEPT, concept.id, _as_dict(concept)))
            elif _as_dict(existing) != _as_dict(concept):
                # An activated proposal that changes a concept (adding an alias,
                # say) must leave a CHANGE revision, not silently overwrite.
                self._append(
                    record_change(
                        EntityType.CONCEPT,
                        concept.id,
                        _as_dict(existing),
                        _as_dict(concept),
                        note="concept updated",
                    )
                )

    def get_concept(self, concept_id: str) -> Concept | None:
        row = self._one("SELECT data FROM concepts WHERE id = ?", (concept_id,))
        return Concept.model_validate_json(row["data"]) if row else None

    def get_concept_by_name(
        self, canonical_name: str, namespace: str | None = None
    ) -> Concept | None:
        """Look up by name, optionally within a namespace.

        Without a namespace this returns the first match in a stable order.
        Callers that care about which `Heap` they mean must pass the namespace
        — the ambiguity is real and the API does not hide it.
        """
        if namespace is not None:
            row = self._one(
                "SELECT data FROM concepts WHERE canonical_name = ? AND namespace = ?",
                (canonical_name, namespace),
            )
        else:
            row = self._one(
                "SELECT data FROM concepts WHERE canonical_name = ?"
                " ORDER BY IFNULL(namespace, '') LIMIT 1",
                (canonical_name,),
            )
        return Concept.model_validate_json(row["data"]) if row else None

    def concepts_named(
        self, canonical_name: str, *, ignore_case: bool = False
    ) -> list[Concept]:
        """Every concept sharing this bare name, across namespaces.

        `ignore_case` because a vault names concepts after its filenames —
        `rag`, `llms`, `ai-agents` — and a human types RAG. The link resolver
        has always treated a case mismatch as resolved; this lookup did not, so
        `forge concept "RAG"` found nothing, fell through to a substring
        search, and reported the template `rag-architecture-review` as a second
        concept *named* RAG. Which it is not: it merely contains the letters.
        """
        sql = (
            "SELECT data FROM concepts WHERE canonical_name = ?"
            + (" COLLATE NOCASE" if ignore_case else "")
            + " ORDER BY IFNULL(namespace, '')"
        )
        rows = self._all(sql, (canonical_name,))
        return [Concept.model_validate_json(r["data"]) for r in rows]

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
        # "The moment any constituent claim changes status or confidence."
        # Outside the transaction above so a synthesis update cannot roll back
        # the claim write that caused it: the claim is the fact, staleness is
        # bookkeeping about it.
        self._mark_dependent_syntheses_stale(claim)

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
        # Supersession writes the retired claim through raw SQL rather than
        # `put_claim`, so the staleness hook there never sees it. A superseded
        # claim is the clearest case there is for staling work written from it,
        # and it was the one case the hook missed. Found by a Phase 9 test.
        self._mark_dependent_syntheses_stale(retired)

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

    def get_link(self, link_id: str) -> ClaimLink | None:
        row = self._one("SELECT data FROM claim_links WHERE id = ?", (link_id,))
        return ClaimLink.model_validate_json(row["data"]) if row else None

    def all_links(self, *, active_only: bool = True) -> list[ClaimLink]:
        """Every relationship. Small graph; a full read is cheap and simpler
        than paginating something that fits in memory many times over."""
        sql = "SELECT data FROM claim_links"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY id"
        return [ClaimLink.model_validate_json(r["data"]) for r in self._all(sql)]

    def count_links(self) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM claim_links")["n"])  # type: ignore[index]

    def delete_links(self, link_ids: Sequence[str]) -> int:
        """Remove relationships outright. Returns how many rows went.

        Deliberately a hard delete rather than `active = 0`: the only caller is
        bootstrap pruning edges it derived from wikilinks that no longer exist,
        and a deactivated row for a link the vault has forgotten is a tombstone
        nobody will ever read. Anything a human or a model asserted is
        superseded, never deleted, which is a different path entirely.
        """
        if not link_ids:
            return 0
        with self._conn:
            cur = self._conn.executemany(
                "DELETE FROM claim_links WHERE id = ?", [(i,) for i in link_ids]
            )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(link_ids)

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

    # -- proposals (Phase 2) -----------------------------------------------

    def put_proposal(self, proposal: Proposal) -> None:
        """Persist a proposal, recording creation and every status change.

        Re-generating an identical proposal is idempotent: identity is derived
        from the proposal's content, so a rejected proposal is not silently
        resurrected as a new PENDING one by the next ingestion run.
        """
        existing = self.get_proposal(proposal.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO proposals"
                "(id, type, status, safety, target, source_id, created_at, data)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (
                    proposal.id,
                    proposal.type.value,
                    proposal.status.value,
                    proposal.safety.value,
                    proposal.operation.target,
                    proposal.source_id,
                    proposal.created_at.isoformat(),
                    _dump(proposal),
                ),
            )
            if existing is None:
                self._append(record_create(EntityType.PROPOSAL, proposal.id, _as_dict(proposal)))
            elif existing.status is not proposal.status:
                self._append(
                    record_change(
                        EntityType.PROPOSAL,
                        proposal.id,
                        _as_dict(existing),
                        _as_dict(proposal),
                        cause=proposal.decided_by,
                        note=f"{existing.status.value} -> {proposal.status.value}",
                    )
                )

    def put_proposal_if_absent(self, proposal: Proposal) -> bool:
        """Store only if unseen. Returns True when it was actually created."""
        if self.get_proposal(proposal.id) is not None:
            return False
        self.put_proposal(proposal)
        return True

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        row = self._one("SELECT data FROM proposals WHERE id = ?", (proposal_id,))
        return Proposal.model_validate_json(row["data"]) if row else None

    def find_proposal(self, prefix: str) -> list[Proposal]:
        """Resolve a possibly-abbreviated id, so the CLI can accept short ids."""
        rows = self._all(
            "SELECT data FROM proposals WHERE id LIKE ? ORDER BY created_at LIMIT 10",
            (prefix + "%",),
        )
        return [Proposal.model_validate_json(r["data"]) for r in rows]

    def list_proposals(
        self,
        *,
        status: ProposalStatus | None = None,
        type: ProposalType | None = None,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Proposal]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self._all(
            f"SELECT data FROM proposals {where} ORDER BY created_at, id LIMIT ? OFFSET ?",
            tuple(params),
        )
        return [Proposal.model_validate_json(r["data"]) for r in rows]

    def count_proposals(self) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) AS n FROM proposals GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}

    # -- derivation cache (Phase 2) ----------------------------------------

    def get_derivation(self, key: str) -> dict[str, Any] | None:
        row = self._one("SELECT payload FROM derivations WHERE key = ?", (key,))
        return json.loads(row["payload"]) if row else None

    def put_derivation(
        self, key: str, kind: str, content_hash: str, payload: dict[str, Any], *, source_id: str | None = None
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO derivations(key, kind, source_id, content_hash, payload, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (key, kind, source_id, content_hash, json.dumps(payload, sort_keys=True),
                 utc_now().isoformat()),
            )

    def invalidate_derivations(self, source_id: str) -> int:
        with self._conn:
            cur = self._conn.execute("DELETE FROM derivations WHERE source_id = ?", (source_id,))
            return int(cur.rowcount)

    # -- lexical search (Phase 2) ------------------------------------------

    def search_spans(
        self, query: str, *, limit: int = 20, document_ids: Sequence[str] | None = None
    ) -> list[tuple[Span, float]]:
        """FTS5 lexical search. Returns (span, score) with lower score = better.

        Uses bm25(), whose sign convention is "more negative is more relevant";
        it is negated here so callers can sort ascending on a positive rank.
        """
        sql = (
            "SELECT span_fts.span_id AS sid, bm25(span_fts) AS rank FROM span_fts "
            "WHERE span_fts MATCH ?"
        )
        params: list[Any] = [query]
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            sql += f" AND span_fts.document_id IN ({placeholders})"
            params.extend(document_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        out: list[tuple[Span, float]] = []
        for row in self._all(sql, tuple(params)):
            span = self.get_span(row["sid"])
            if span is not None:
                out.append((span, float(row["rank"])))
        return out

    def rebuild_search_index(self) -> int:
        """Rebuild the FTS index from spans. Derived state; safe to run anytime."""
        with self._conn:
            self._conn.execute("DELETE FROM span_fts")
            rows = self._all("SELECT data FROM spans")
            spans = [Span.model_validate_json(r["data"]) for r in rows]
            self._conn.executemany(
                "INSERT INTO span_fts(span_id, document_id, text) VALUES(?,?,?)",
                [(s.id, s.document_id, s.text) for s in spans],
            )
        return len(spans)

    # -- workflow runs (Phase 4) -------------------------------------------

    def put_workflow(self, run: WorkflowRun) -> None:
        """Upsert a run. Append-only in spirit: runs are updated, never deleted."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO workflows(id, source_id, status, provider_id, model_id,"
                " created_at, updated_at, data) VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "  status = excluded.status,"
                "  provider_id = excluded.provider_id,"
                "  model_id = excluded.model_id,"
                "  updated_at = excluded.updated_at,"
                "  data = excluded.data",
                (
                    run.id,
                    run.source_id,
                    run.status.value,
                    run.provider_id,
                    run.model_id,
                    run.created_at.isoformat(),
                    utc_now().isoformat(),
                    run.model_dump_json(),
                ),
            )

    def get_workflow(self, workflow_id: str) -> WorkflowRun | None:
        row = self._one("SELECT data FROM workflows WHERE id = ?", (workflow_id,))
        return WorkflowRun.model_validate_json(row["data"]) if row else None

    def find_workflow(self, prefix: str) -> list[WorkflowRun]:
        """Resolve a possibly-abbreviated id, matching the proposal CLI's ergonomics."""
        rows = self._all(
            "SELECT data FROM workflows WHERE id LIKE ? ORDER BY created_at LIMIT 10",
            (prefix + "%",),
        )
        return [WorkflowRun.model_validate_json(r["data"]) for r in rows]

    def list_workflows(
        self,
        *,
        status: WorkflowStatus | None = None,
        source_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._all(
            f"SELECT data FROM workflows {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        return [WorkflowRun.model_validate_json(r["data"]) for r in rows]

    def count_workflows(self) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) AS n FROM workflows GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}

    # -- embeddings (Phase 2, optional) ------------------------------------

    def put_embedding(
        self, owner_type: str, owner_id: str, model: str, vector: Sequence[float]
    ) -> None:
        import struct

        blob = struct.pack(f"<{len(vector)}f", *vector)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings"
                "(owner_type, owner_id, model, dimensions, vector, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (owner_type, owner_id, model, len(vector), blob, utc_now().isoformat()),
            )

    def get_embeddings(self, owner_type: str, model: str) -> list[tuple[str, list[float]]]:
        import struct

        rows = self._all(
            "SELECT owner_id, dimensions, vector FROM embeddings WHERE owner_type = ? AND model = ?",
            (owner_type, model),
        )
        return [
            (r["owner_id"], list(struct.unpack(f"<{r['dimensions']}f", r["vector"])))
            for r in rows
        ]

    def count_embeddings(self) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM embeddings")["n"])  # type: ignore[index]

    def embedded_models(self) -> dict[str, int]:
        """model_id -> vector count, for reporting what is actually stored.

        `forge embeddings build --provider ollama` and `forge retrieval-eval
        --provider hashing` are one flag apart, and the mismatch shows up as
        "semantic skipped" with no hint that vectors exist under another name.
        """
        rows = self._all("SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model")
        return {str(r["model"]): int(r["n"]) for r in rows}

    # -- helpers -----------------------------------------------------------

    # -- Phase 9: questions ------------------------------------------------

    def put_question(self, question: Question) -> None:
        existing = self.get_question(question.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO questions(id, status, text, created_at, data)"
                " VALUES(?,?,?,?,?)",
                (
                    question.id,
                    question.status.value,
                    question.text,
                    question.created_at.isoformat(),
                    _dump(question),
                ),
            )
            if existing is None:
                self._append(record_create(EntityType.QUESTION, question.id, _as_dict(question)))
            elif _as_dict(existing) != _as_dict(question):
                self._append(
                    record_change(
                        EntityType.QUESTION,
                        question.id,
                        _as_dict(existing),
                        _as_dict(question),
                    )
                )

    def get_question(self, question_id: str) -> Question | None:
        row = self._one("SELECT data FROM questions WHERE id = ?", (question_id,))
        return Question.model_validate_json(row["data"]) if row else None

    def list_questions(self, status: QuestionStatus | None = None) -> Sequence[Question]:
        if status is None:
            rows = self._all("SELECT data FROM questions ORDER BY created_at")
        else:
            rows = self._all(
                "SELECT data FROM questions WHERE status = ? ORDER BY created_at",
                (status.value,),
            )
        return [Question.model_validate_json(r["data"]) for r in rows]

    def link_answer(self, question_id: str, claim_id: str) -> None:
        """Record that a claim bears on a question. Idempotent."""
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO question_answers(question_id, claim_id, created_at)"
                " VALUES(?,?,?)",
                (question_id, claim_id, utc_now().isoformat()),
            )

    def answers_for_question(self, question_id: str) -> Sequence[Claim]:
        rows = self._all(
            "SELECT c.data AS data FROM question_answers qa"
            " JOIN claims c ON c.id = qa.claim_id"
            " WHERE qa.question_id = ? ORDER BY qa.created_at",
            (question_id,),
        )
        return [Claim.model_validate_json(r["data"]) for r in rows]

    # -- inbound links, counted over the whole vault ------------------------

    def set_concept_inbound(
        self, counts: Mapping[str, tuple[int, Sequence[str]]]
    ) -> int:
        """Replace the inbound-link counts wholesale.

        Wholesale rather than row by row because the input is one pass over the
        vault: a concept whose last inbound link was deleted must lose its row,
        and an update that only writes what it saw would leave that row behind
        saying a page is linked when nothing links to it any more.
        """
        now = utc_now().isoformat()
        with self._conn:
            self._conn.execute("DELETE FROM concept_inbound_links")
            self._conn.executemany(
                "INSERT INTO concept_inbound_links(concept_id, inbound, sources, updated_at)"
                " VALUES(?,?,?,?)",
                [
                    (concept_id, int(count), json.dumps(list(sources)[:5]), now)
                    for concept_id, (count, sources) in counts.items()
                ],
            )
        return len(counts)

    def concept_inbound(self, concept_id: str) -> tuple[int, list[str]] | None:
        """How many pages link to this concept's page, and a few of them.

        `None` means no row, which is **not** the same as a row of zero: the
        counts are replaced wholesale on every bootstrap, so a concept without
        one was not seen in the last pass — its page has been deleted or
        renamed and the concept is a leftover. Saying "nothing links to this
        page" about a page that no longer exists is a false statement dressed
        as a finding, so callers must tell the two apart.
        """
        row = self._one(
            "SELECT inbound, sources FROM concept_inbound_links WHERE concept_id = ?",
            (concept_id,),
        )
        if row is None:
            return None
        return int(row["inbound"]), list(json.loads(row["sources"]))

    def unreferenced_concepts(self) -> list[str] | None:
        """Concepts no page in the vault links to. `None` if nobody has counted.

        `None` and `[]` are different answers and the caller must not collapse
        them: one means nothing links to any concept was ever checked, the
        other means everything is linked.
        """
        if not self.inbound_counted():
            return None
        rows = self._all(
            "SELECT concept_id FROM concept_inbound_links WHERE inbound = 0"
            " ORDER BY concept_id"
        )
        return [str(r["concept_id"]) for r in rows]

    def inbound_counted(self) -> bool:
        """Whether inbound links have ever been counted for this store.

        A store bootstrapped before this table existed, or never bootstrapped
        at all, cannot distinguish "nothing links here" from "nobody looked".
        Callers must say which of those they are reporting.
        """
        row = self._one("SELECT COUNT(*) AS n FROM concept_inbound_links")
        return bool(row and int(row["n"]))

    # -- Phase 9: syntheses and the staleness invariant ---------------------

    def put_synthesis(self, synthesis: Synthesis) -> None:
        """Store a synthesis and snapshot the state of every claim it rests on.

        The snapshot is what makes staleness a comparison rather than a
        judgement: a later check asks whether any constituent claim's status or
        confidence differs from what was recorded here.
        """
        existing = self.get_synthesis(synthesis.id)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO syntheses(id, scope, scope_id, stale, generated_at, data)"
                " VALUES(?,?,?,?,?,?)",
                (
                    synthesis.id,
                    synthesis.scope.value,
                    synthesis.scope_id,
                    int(synthesis.stale),
                    synthesis.generated_at.isoformat(),
                    _dump(synthesis),
                ),
            )
            self._conn.execute(
                "DELETE FROM synthesis_claims WHERE synthesis_id = ?", (synthesis.id,)
            )
            for claim_id in synthesis.source_claim_ids:
                claim = self.get_claim(claim_id)
                self._conn.execute(
                    "INSERT INTO synthesis_claims(synthesis_id, claim_id, status, fingerprint)"
                    " VALUES(?,?,?,?)",
                    (
                        synthesis.id,
                        claim_id,
                        claim.status.value if claim else "missing",
                        claim_fingerprint(claim) if claim else "missing",
                    ),
                )
            if existing is None:
                self._append(
                    record_create(EntityType.SYNTHESIS, synthesis.id, _as_dict(synthesis))
                )

    def get_synthesis(self, synthesis_id: str) -> Synthesis | None:
        row = self._one("SELECT data FROM syntheses WHERE id = ?", (synthesis_id,))
        return Synthesis.model_validate_json(row["data"]) if row else None

    def list_syntheses(self, *, stale: bool | None = None) -> Sequence[Synthesis]:
        if stale is None:
            rows = self._all("SELECT data FROM syntheses ORDER BY generated_at DESC")
        else:
            rows = self._all(
                "SELECT data FROM syntheses WHERE stale = ? ORDER BY generated_at DESC",
                (int(stale),),
            )
        return [Synthesis.model_validate_json(r["data"]) for r in rows]

    def syntheses_for_claim(self, claim_id: str) -> Sequence[Synthesis]:
        rows = self._all(
            "SELECT s.data AS data FROM synthesis_claims sc"
            " JOIN syntheses s ON s.id = sc.synthesis_id WHERE sc.claim_id = ?",
            (claim_id,),
        )
        return [Synthesis.model_validate_json(r["data"]) for r in rows]

    def _mark_dependent_syntheses_stale(self, claim: Claim) -> None:
        """Stale every synthesis whose snapshot of this claim no longer matches.

        Deterministic and cheap: a status string and a float, compared. Never a
        model call, which is the point of the invariant — "does this make an
        existing synthesis outdated?" is answered by software.
        """
        rows = self._all(
            "SELECT synthesis_id, status, fingerprint FROM synthesis_claims WHERE claim_id = ?",
            (claim.id,),
        )
        if not rows:
            return
        current = claim_fingerprint(claim)
        for row in rows:
            if row["fingerprint"] == current:
                continue
            self.mark_synthesis_stale(
                row["synthesis_id"], _describe_drift(claim.id, row["fingerprint"], current)
            )

    def mark_synthesis_stale(self, synthesis_id: str, reason: str) -> bool:
        """Mark one synthesis stale, recording why. Returns whether it changed."""
        synthesis = self.get_synthesis(synthesis_id)
        if synthesis is None or synthesis.stale:
            return False
        updated = synthesis.model_copy(update={"stale": True, "stale_reason": reason})
        with self._conn:
            self._conn.execute(
                "UPDATE syntheses SET stale = 1, data = ? WHERE id = ?",
                (_dump(updated), synthesis_id),
            )
            self._append(
                record_change(
                    EntityType.SYNTHESIS,
                    synthesis_id,
                    _as_dict(synthesis),
                    _as_dict(updated),
                    note=reason,
                )
            )
        return True

    def recheck_synthesis_staleness(self) -> list[str]:
        """Re-derive staleness for every synthesis. Returns the ids newly stale.

        The write-time hook catches claims that change through `put_claim`.
        This catches everything else: a claim deleted out from under a
        synthesis, a database restored from a backup, a bug in the hook. The
        invariant is worth being able to re-establish from the data alone.
        """
        newly: list[str] = []
        for synthesis in self.list_syntheses(stale=False):
            for row in self._all(
                "SELECT claim_id, fingerprint FROM synthesis_claims WHERE synthesis_id = ?",
                (synthesis.id,),
            ):
                claim = self.get_claim(row["claim_id"])
                if claim is None:
                    reason = f"claim {row['claim_id']} is gone"
                else:
                    current = claim_fingerprint(claim)
                    if row["fingerprint"] == current:
                        continue
                    reason = _describe_drift(claim.id, row["fingerprint"], current)
                if self.mark_synthesis_stale(synthesis.id, reason):
                    newly.append(synthesis.id)
                break
        return newly

    def counts(self) -> dict[str, int]:
        return {
            table: int(self._one(f"SELECT COUNT(*) AS n FROM {table}")["n"])  # type: ignore[index]
            for table in (
                "workflows",
                "sources",
                "documents",
                "spans",
                "concepts",
                "claims",
                "evidence_links",
                "claim_links",
                "revisions",
                "proposals",
                "derivations",
                "embeddings",
                "questions",
                "syntheses",
            )
        }

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()


def claim_fingerprint(claim: Claim) -> str:
    """What a synthesis was written from, reduced to a comparable string.

    Status, tier, statement and supersession. A change in any of them changes
    what the claim asserts or how strongly, which is exactly when text written
    from it stops being safe to trust.

    Not a content hash of the whole claim: `created_at` and provenance
    bookkeeping move without changing what the claim says, and staleness that
    fires on those would cry wolf until nobody read it.
    """
    return "|".join(
        [
            claim.status.value,
            claim.provenance.tier.value,
            claim.statement.strip(),
            claim.superseded_by or "",
        ]
    )


def _describe_drift(claim_id: str, before: str, after: str) -> str:
    """Name the parts that moved, so `stale_reason` is readable by a human."""
    labels = ("status", "tier", "statement", "superseded_by")
    old = before.split("|")
    new = after.split("|")
    moved = [
        f"{label} {o!r} -> {n!r}"
        for label, o, n in zip(labels, old, new)
        if o != n
    ]
    return f"claim {claim_id} changed: {', '.join(moved) if moved else 'content differs'}"


def _dump(model: Any) -> str:
    return model.model_dump_json()


def _as_dict(model: Any) -> dict[str, Any]:
    """Round-trip through JSON so revision snapshots are plain, storable data."""
    return json.loads(model.model_dump_json())  # type: ignore[no-any-return]
