"""Hash-based change detection and content hashing.

The target property: re-indexing an unchanged corpus performs zero LLM calls.
"""

from __future__ import annotations

from forge.corpus.indexer import detect_changes
from forge.corpus.model import IndexedFile
from forge.domain import ChangeStatus
from forge.ids import content_hash, deterministic_id, text_hash


def f(path: str, h: str) -> IndexedFile:
    return IndexedFile(
        path=path,
        content_hash=h,
        byte_size=0,
        line_count=0,
        title=None,
        frontmatter_present=False,
        frontmatter_valid=False,
    )


class TestHashing:
    def test_same_text_same_hash(self):
        assert text_hash("hello") == text_hash("hello")

    def test_different_text_different_hash(self):
        assert text_hash("hello") != text_hash("hello ")

    def test_crlf_and_lf_hash_identically(self):
        """The corpus was authored on Windows; a line-ending difference between
        checkouts must not read as a content change."""
        assert text_hash("a\r\nb\r\n") == text_hash("a\nb\n")
        assert text_hash("a\rb") == text_hash("a\nb")

    def test_content_hash_is_sha256_hex(self):
        h = content_hash(b"x")
        assert len(h) == 64 and int(h, 16) >= 0

    def test_deterministic_id_is_stable_and_namespaced(self):
        assert deterministic_id("source", "a.md") == deterministic_id("source", "a.md")
        assert deterministic_id("source", "a.md") != deterministic_id("concept", "a.md")

    def test_deterministic_id_separator_is_unambiguous(self):
        """('a','bc') must not collide with ('ab','c')."""
        assert deterministic_id("n", "a", "bc") != deterministic_id("n", "ab", "c")


class TestChangeDetection:
    def test_all_new_on_first_run(self):
        cs = detect_changes([f("a.md", "h1"), f("b.md", "h2")], {})
        assert len(cs.new) == 2
        assert cs.summary() == {"new": 2, "modified": 0, "unchanged": 0, "deleted": 0}

    def test_all_unchanged_on_second_run(self):
        current = [f("a.md", "h1"), f("b.md", "h2")]
        cs = detect_changes(current, {"a.md": "h1", "b.md": "h2"})
        assert len(cs.unchanged) == 2
        assert cs.requires_processing == [], "nothing to process means zero LLM calls"

    def test_single_modification_is_isolated(self):
        """Editing one file must mark exactly that file changed."""
        current = [f("a.md", "h1"), f("b.md", "CHANGED"), f("c.md", "h3")]
        cs = detect_changes(current, {"a.md": "h1", "b.md": "h2", "c.md": "h3"})
        assert [c.path for c in cs.modified] == ["b.md"]
        assert {c.path for c in cs.unchanged} == {"a.md", "c.md"}
        assert [c.path for c in cs.requires_processing] == ["b.md"]

    def test_modified_records_both_hashes(self):
        cs = detect_changes([f("a.md", "new")], {"a.md": "old"})
        change = cs.modified[0]
        assert change.old_hash == "old" and change.new_hash == "new"

    def test_deletion_detected(self):
        cs = detect_changes([f("a.md", "h1")], {"a.md": "h1", "gone.md": "h9"})
        assert [c.path for c in cs.deleted] == ["gone.md"]
        assert cs.deleted[0].old_hash == "h9"

    def test_new_and_deleted_together(self):
        cs = detect_changes([f("new.md", "h")], {"old.md": "h"})
        assert cs.summary() == {"new": 1, "modified": 0, "unchanged": 0, "deleted": 1}

    def test_deleted_files_are_not_processing_work(self):
        cs = detect_changes([], {"gone.md": "h"})
        assert cs.requires_processing == []

    def test_output_is_sorted_deterministically(self):
        cs = detect_changes([f("z.md", "1"), f("a.md", "2"), f("m.md", "3")], {})
        assert [c.path for c in cs.changes] == ["a.md", "m.md", "z.md"]

    def test_identical_content_at_different_paths_both_tracked(self):
        """Duplicate content is not the same source."""
        cs = detect_changes([f("a.md", "same"), f("b.md", "same")], {})
        assert len(cs.new) == 2
