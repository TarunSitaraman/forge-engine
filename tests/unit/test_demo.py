"""`forge demo`: does the sample vault still carry its defects, and are they found?

This file is not testing a presentation. `forge demo` plants five specific
defects and reports whether the engine came back with each one, so these tests
are a regression check on the five claims the README makes: that the tool finds
a link which resolves and is wrong, a page nothing links to, a dead link, an
ambiguous link, and frontmatter that does not parse without also calling a note
with no frontmatter a defect.

If one of them starts failing, the demo prints MISSED to whoever is watching and
exits non-zero. These tests catch it before that happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from forge.cli.demo import FILES, build, is_occupied, render, write_vault
from forge.config import Settings
from forge.llm.base import CALLS


@pytest.fixture
def demo_vault(tmp_path: Path) -> Settings:
    vault = write_vault(tmp_path / "demo")
    return Settings(vault_path=vault, state_dir=tmp_path / "state")


def test_the_sample_vault_is_written_whole(tmp_path: Path) -> None:
    write_vault(tmp_path / "v")
    on_disk = {str(p.relative_to(tmp_path / "v")) for p in (tmp_path / "v").rglob("*.md")}
    assert on_disk == set(FILES)


def test_every_planted_defect_is_reported(demo_vault: Settings) -> None:
    checks, counts = build(demo_vault)
    missed = [c.title for c in checks if not c.found]
    assert not missed, f"the engine no longer finds: {missed}"
    assert counts["found"] == counts["total"] == 5


def test_the_demo_makes_no_model_calls(demo_vault: Settings) -> None:
    """The whole claim. A demo that needs an API key demonstrates nothing."""
    CALLS.reset()
    build(demo_vault)
    assert CALLS.count == 0


def test_the_wrong_link_is_found_by_reachability_not_by_resolution(
    demo_vault: Settings,
) -> None:
    """The headline finding: every link in the vault resolves except the one dead one.

    If this vault's cheat-sheet defect were detectable as an unresolved link it
    would prove nothing, because a link checker would find it too.
    """
    from forge.corpus.diagnostics import link_report
    from forge.corpus.indexer import CorpusIndexer

    report = link_report(CorpusIndexer(demo_vault).build_index())
    assert report.unresolved_targets.keys() == {"Monotonic Stack", "Heap"}
    assert "Binary Search Cheat Sheet" not in report.unresolved_targets

    checks, _ = build(demo_vault)
    assert checks[0].found


def test_a_note_without_frontmatter_is_not_called_a_defect(demo_vault: Settings) -> None:
    from forge.corpus.diagnostics import frontmatter_report
    from forge.corpus.indexer import CorpusIndexer

    report = frontmatter_report(CorpusIndexer(demo_vault).build_index())
    assert report.without_frontmatter == 1
    assert report.by_severity.get("error", 0) == 1  # only the unparseable one


def test_the_demo_refuses_to_write_over_a_directory_it_did_not_create(
    tmp_path: Path,
) -> None:
    notes = tmp_path / "someones-notes"
    notes.mkdir()
    (notes / "important.md").write_text("mine", encoding="utf-8")
    assert is_occupied(notes)
    assert not is_occupied(tmp_path / "empty-and-absent")


def test_the_walkthrough_names_the_vault_and_leaves_no_trailing_space(
    demo_vault: Settings,
) -> None:
    checks, counts = build(demo_vault)
    lines = list(render(checks, counts, demo_vault.vault_path, calls=0))
    assert all(line == line.rstrip() for line in lines)
    assert any(str(demo_vault.vault_path) in line for line in lines)
    assert any("Model calls: 0" in line for line in lines)
