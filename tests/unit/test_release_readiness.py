"""Phase 10 gates: rebuild determinism, and backup covering what rebuild loses.

The third gate, "fresh-machine setup works from documentation alone", is a
clean-room exercise rather than a unit test, and its findings are recorded in
`docs/roadmap.md`. What is testable here is what it uncovered: the vault-marker
rule and the bootstrap link reporting, both of which are pinned below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from forge.bootstrap import build_plan
from forge.config import VAULT_MARKERS, ConfigError, Settings, _find_vault_root
from forge.corpus.indexer import CorpusIndexer
from forge.domain import (
    Derivation,
    Provenance,
    ProvenanceTier,
    Question,
)
from forge.storage import SqliteStore, create_backup, restore_backup
from forge.storage.backup import BackupError, BackupManifest

HUMAN = Provenance(
    tier=ProvenanceTier.USER_ASSERTION, derivation=Derivation.HUMAN, agent="test"
)


def write_vault(root: Path, *, marker: str = ".git") -> Path:
    """A minimal vault: two pages, one wikilink between them."""
    (root / marker).mkdir(parents=True, exist_ok=True)
    notes = root / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "RAG.md").write_text(
        "# RAG\n\nRetrieval grounds generation.\n\nSee also [[Vector Databases]].\n",
        encoding="utf-8",
    )
    (notes / "Vector Databases.md").write_text(
        "# Vector Databases\n\nStores embeddings, retrieves by similarity.\n",
        encoding="utf-8",
    )
    return root


def rebuild(settings: Settings, db_path: Path) -> dict:
    """Index and seed from Markdown alone, the way a fresh machine would."""
    store = SqliteStore(db_path)
    store.initialize()
    indexer = CorpusIndexer(settings)
    index = indexer.build_index()
    plan = build_plan(index, decided=indexer._decided_targets())
    for concept in plan.concepts:
        store.put_concept(concept)
    for link in plan.links:
        store.put_link(link)
    result = {
        "concepts": sorted(c.id for c in store.list_concepts()),
        "links": sorted(link.id for link in store.all_links()),
        "counts": store.counts(),
    }
    store.close()
    return result


# -- gate: a full rebuild from Markdown reproduces the derived model --------


def test_gate_a_rebuild_from_markdown_reproduces_the_derived_model(tmp_path: Path):
    """Twice from the same Markdown, into two empty stores, must agree exactly.

    Ids are deterministic by construction, so this is really asking whether
    anything non-deterministic leaked into the derivation: a timestamp in an
    id, a set iterated in hash order, a dict that lost its ordering.
    """
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")

    first = rebuild(settings, tmp_path / "a.db")
    second = rebuild(settings, tmp_path / "b.db")

    assert first["concepts"] == second["concepts"]
    assert first["links"] == second["links"]
    assert first["counts"] == second["counts"]
    assert first["concepts"], "the rebuild produced no concepts at all"


def test_a_rebuild_into_a_store_that_already_has_the_data_adds_nothing(tmp_path: Path):
    """Re-running the deterministic path is idempotent, not additive."""
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"

    once = rebuild(settings, db)
    twice = rebuild(settings, db)

    assert once["counts"]["concepts"] == twice["counts"]["concepts"]
    assert once["counts"]["claim_links"] == twice["counts"]["claim_links"]


# -- gate: backup covers state Markdown cannot express (ADR-001 R5) --------


def test_gate_a_rebuild_loses_exactly_what_markdown_cannot_express(tmp_path: Path):
    """Names the gap this gate exists to close, before closing it.

    A question is the cleanest example: the user asked it, it is not in the
    vault, and no amount of re-indexing brings it back. If this ever fails
    because a rebuild *does* restore it, the backup story needs revisiting, not
    the test.
    """
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)

    store = SqliteStore(db)
    store.put_question(
        Question(id=Question.make_id("What is RAG?"), text="What is RAG?", provenance=HUMAN)
    )
    assert len(store.list_questions()) == 1
    store.close()

    db.unlink()  # the "rebuild from scratch" a user is told is safe
    rebuild(settings, db)

    store = SqliteStore(db)
    assert store.list_questions() == [], "a rebuild is expected to lose this"
    assert store.list_concepts(), "but it must still reproduce the deterministic part"
    store.close()


def test_gate_backup_and_restore_returns_what_a_rebuild_loses(tmp_path: Path):
    """The mitigation R5 asks for, end to end."""
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)

    store = SqliteStore(db)
    question = Question(
        id=Question.make_id("What is RAG?"), text="What is RAG?", provenance=HUMAN
    )
    store.put_question(question)
    before = store.counts()
    store.close()

    manifest = create_backup(db, tmp_path / "backup")
    assert manifest.counts["questions"] == 1
    assert manifest.sha256

    db.unlink()
    rebuild(settings, db)
    restore_backup(tmp_path / "backup", db, force=True)

    store = SqliteStore(db)
    assert [q.id for q in store.list_questions()] == [question.id]
    assert store.counts() == before
    store.close()


def test_a_restore_will_not_overwrite_without_being_told_to(tmp_path: Path):
    """Restoring is destructive by nature, so the caller has to say so."""
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)
    create_backup(db, tmp_path / "backup")

    with pytest.raises(BackupError, match="already exists"):
        restore_backup(tmp_path / "backup", db)


def test_a_corrupted_backup_is_refused_rather_than_restored(tmp_path: Path):
    """A backup that looks like one and is not is worse than none.

    Checked against the manifest's own checksum, so a truncated copy or an
    edited file is caught before it replaces a working store.
    """
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)
    create_backup(db, tmp_path / "backup")

    # Flip real bytes rather than overwriting the tail: a small SQLite file is
    # already zero-padded at the end, so writing zeros there changes nothing
    # and the first version of this test corrupted nothing at all.
    archive = tmp_path / "backup" / "forge.db"
    raw = bytearray(archive.read_bytes())
    raw[200:216] = b"corrupted-bytes!"
    archive.write_bytes(bytes(raw))

    with pytest.raises(BackupError, match="checksum"):
        restore_backup(tmp_path / "backup", db, force=True)


def test_a_backup_from_a_newer_schema_is_refused(tmp_path: Path):
    """Restoring it would be a silent downgrade into a schema that cannot hold it."""
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)
    create_backup(db, tmp_path / "backup")

    manifest_path = tmp_path / "backup" / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["schema_version"] = 999
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(BackupError, match="silent downgrade"):
        restore_backup(tmp_path / "backup", db, force=True)


def test_a_directory_that_is_not_a_backup_says_so(tmp_path: Path):
    (tmp_path / "random").mkdir()
    with pytest.raises(BackupError, match="not a Forge backup"):
        restore_backup(tmp_path / "random", tmp_path / "forge.db")


def test_a_restore_clears_stale_wal_sidecars(tmp_path: Path):
    """Otherwise a restore appears to work and then serves the old data back.

    SQLite reads a `-wal` beside the database it is opening. Replacing only the
    `.db` leaves the previous write-ahead log in place, and the first read
    replays it over the restored file.
    """
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)
    create_backup(db, tmp_path / "backup")

    stale = db.with_name(db.name + "-wal")
    stale.write_bytes(b"stale write-ahead log")

    restore_backup(tmp_path / "backup", db, force=True)

    assert not stale.exists() or stale.read_bytes() != b"stale write-ahead log"


def test_a_backup_manifest_round_trips(tmp_path: Path):
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    db = tmp_path / "forge.db"
    rebuild(settings, db)
    written = create_backup(db, tmp_path / "backup")

    loaded = BackupManifest.load(tmp_path / "backup" / "manifest.json")

    assert loaded.sha256 == written.sha256
    assert loaded.counts == written.counts
    assert loaded.schema_version == written.schema_version


# -- what the clean-room run found -----------------------------------------


@pytest.mark.parametrize("marker", VAULT_MARKERS)
def test_any_vault_marker_is_accepted(tmp_path: Path, marker: str):
    """`.git` alone turned a plain notes folder into a configuration error.

    Found by installing into a clean venv and following the README: a folder of
    Markdown notes is not a git repository, and the first documented command
    failed before reaching anything Forge does.
    """
    vault = tmp_path / f"vault-{marker.strip('.')}"
    write_vault(vault, marker=marker)
    assert _find_vault_root(vault / "notes") == vault.resolve()


def test_an_unmarked_directory_is_still_not_a_vault(tmp_path: Path):
    """The guarantee the marker rule exists for, unchanged.

    Without it `forge index` would treat an arbitrary directory as a vault,
    write a `.forge/` into it and report success: silently indexing the wrong
    thing instead of saying it could not find the right thing.
    """
    plain = tmp_path / "not-a-vault"
    (plain / "notes").mkdir(parents=True)
    assert _find_vault_root(plain / "notes") is None


def test_the_missing_vault_error_names_every_marker_and_a_fix(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_VAULT_PATH", raising=False)
    monkeypatch.setattr(
        "forge.config._find_vault_root", lambda start: None
    )

    with pytest.raises(ConfigError) as excinfo:
        Settings.load()

    message = str(excinfo.value)
    for marker in VAULT_MARKERS:
        assert marker in message
    assert "FORGE_VAULT_PATH" in message


def test_bootstrap_reports_links_it_declined_to_turn_into_edges(tmp_path: Path):
    """The clean-room run's worst moment: `edges: 0` and no reason given.

    `[[Vector Databases]]` against `vector-databases.md` is a
    `renamed_candidate`, which bootstrap correctly refuses to act on because
    doing so would be guessing. Saying nothing about it left a first-time user
    with an empty graph and no way to learn why.
    """
    vault = tmp_path / "vault"
    (vault / ".git").mkdir(parents=True)
    notes = vault / "notes"
    notes.mkdir()
    (notes / "rag.md").write_text("# RAG\n\nSee [[Vector Databases]].\n", encoding="utf-8")
    (notes / "vector-databases.md").write_text("# Vector Databases\n\nText.\n", encoding="utf-8")

    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    indexer = CorpusIndexer(settings)
    plan = build_plan(indexer.build_index(), decided=indexer._decided_targets())

    assert plan.links == [], "the link is a rename candidate, not a resolved link"
    assert sum(plan.skipped_links.values()) == 1
    assert "renamed_candidate" in plan.skipped_links
    assert plan.to_dict()["skipped_links"] == plan.skipped_links


def test_a_resolved_link_still_becomes_an_edge(tmp_path: Path):
    """The reporting must not have changed what actually gets built."""
    vault = write_vault(tmp_path / "vault")
    settings = Settings.load(vault_path=vault, state_dir=tmp_path / "state")
    indexer = CorpusIndexer(settings)
    plan = build_plan(indexer.build_index(), decided=indexer._decided_targets())

    assert len(plan.links) == 1
    assert plan.skipped_links == {}


# -- the commands themselves ------------------------------------------------


def test_backup_and_restore_work_through_the_cli(tmp_path: Path, monkeypatch):
    """Covers the wiring, which the library tests above do not.

    The first version of `forge backup` raised `NameError: utc_now` on its
    first line, because a lint autofix had removed the import as unused before
    the command that needed it was written. Every test above passed: they call
    `create_backup` directly. Only running the command found it, which is the
    argument for the clean-room walkthrough this file records.
    """
    from forge.cli.main import app
    from typer.testing import CliRunner

    vault = write_vault(tmp_path / "vault")
    monkeypatch.setenv("FORGE_VAULT_PATH", str(vault))
    monkeypatch.setenv("FORGE_STATE_DIR", str(tmp_path / "state"))
    settings = Settings.load()
    rebuild(settings, settings.db_path)

    store = SqliteStore(settings.db_path)
    store.put_question(
        Question(id=Question.make_id("What is RAG?"), text="What is RAG?", provenance=HUMAN)
    )
    store.close()

    runner = CliRunner()
    made = runner.invoke(app, ["backup", "--out", str(tmp_path / "bk"), "--json"])
    assert made.exit_code == 0, made.output
    assert json.loads(made.output)["counts"]["questions"] == 1

    settings.db_path.unlink()
    rebuild(settings, settings.db_path)

    refused = runner.invoke(app, ["restore", str(tmp_path / "bk")])
    assert refused.exit_code == 2, "restoring over an existing store must need --force"
    assert "already exists" in refused.output

    done = runner.invoke(app, ["restore", str(tmp_path / "bk"), "--force"])
    assert done.exit_code == 0, done.output

    store = SqliteStore(settings.db_path)
    assert len(store.list_questions()) == 1
    store.close()


def test_the_version_command_reports_what_is_installed():
    """`forge --version` was `No such option` until 2026-09-07.

    Found in the first minute of the clean-room run, which is roughly how long
    it takes a new user to find it too.
    """
    from forge.cli.main import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "forge-kb" in result.output
    assert "store schema : v" in result.output
