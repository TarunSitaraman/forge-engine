"""A convention system only applies where its defining document exists.

The rulesets in `forge.corpus.conventions` are transcriptions of two specific
Markdown files in *this* vault. Asserting them against someone else's notes
measures conformance to rules its author never agreed to, and reports a
conflict between two documents that are not there.

Before 2026-09-02 a three-file test vault was told its filenames were 33%
conforming and that `dsa-local` claimed 0 files. That is the bug these lock.
"""

from __future__ import annotations

from forge.corpus.conventions import (
    DSA_LOCAL,
    REPO_WIDE,
    analyze_conventions,
    applicable_systems,
)
from forge.corpus.model import CorpusIndex, IndexedFile


def _file(path: str, *, tags=(), frontmatter=False) -> IndexedFile:
    return IndexedFile(
        path=path,
        content_hash="h",
        byte_size=1,
        line_count=1,
        title=None,
        doc_type=None,
        status=None,
        canonical=False,
        tags=tuple(tags),
        related=(),
        frontmatter_present=frontmatter,
        frontmatter_valid=True,
        frontmatter_keys=(),
        heading_count=0,
        wikilink_count=0,
        markdown_link_count=0,
        code_block_count=0,
        code_languages=(),
        links=(),
        diagnostics=(),
        repairs=(),
    )


def _index(*paths: str) -> CorpusIndex:
    return CorpusIndex(vault_path=".", files=[_file(p) for p in paths])


class TestApplicableSystems:
    def test_a_vault_with_neither_document_gets_neither_system(self):
        assert applicable_systems(_index("notes/meeting.md", "notes/apollo.md")) == ()

    def test_only_the_system_whose_document_is_present_applies(self):
        got = applicable_systems(_index("CONVENTIONS.md", "notes/a.md"))
        assert got == (REPO_WIDE,)

    def test_both_apply_when_both_documents_exist(self):
        got = applicable_systems(_index("CONVENTIONS.md", "DSA/Documentation Standards.md"))
        assert got == (REPO_WIDE, DSA_LOCAL)

    def test_a_similarly_named_file_elsewhere_does_not_count(self):
        """Presence is an exact vault-relative path, not a basename match:
        `archive/CONVENTIONS.md` is a copy, not the governing document."""
        assert applicable_systems(_index("archive/CONVENTIONS.md")) == ()


class TestAnalyzeOnAForeignVault:
    def test_no_conformance_is_reported(self):
        report = analyze_conventions(_index("notes/a.md", "notes/b.md"))
        assert report.conformance == {}
        assert report.systems == []
        assert report.applied == []

    def test_no_conflict_is_invented(self):
        """The damaging half: two documents that are absent cannot disagree."""
        assert analyze_conventions(_index("notes/a.md")).conflicts == []

    def test_the_status_says_why_rather_than_claiming_a_verdict(self):
        status = analyze_conventions(_index("notes/a.md")).resolution_status
        assert "NOT APPLICABLE" in status
        assert "CONVENTIONS.md" in status, "name what was looked for"

    def test_it_survives_an_empty_vault(self):
        report = analyze_conventions(_index())
        assert report.conflicts == []
        assert "NOT APPLICABLE" in report.resolution_status


class TestAnalyzeWithOneSystem:
    def test_conformance_is_measured_for_the_present_system_only(self):
        report = analyze_conventions(_index("CONVENTIONS.md", "notes/a.md"))
        assert set(report.conformance) == {"repo-wide"}
        assert report.applied == ["repo-wide"]

    def test_no_conflict_with_nothing_to_conflict_against(self):
        report = analyze_conventions(_index("CONVENTIONS.md", "notes/a.md"))
        assert report.conflicts == []
        assert "SINGLE SYSTEM" in report.resolution_status


class TestAnalyzeWithBothSystems:
    """This vault's own case, which must not change."""

    def _report(self):
        return analyze_conventions(
            _index("CONVENTIONS.md", "DSA/Documentation Standards.md", "DSA/01_Patterns/Heap.md")
        )

    def test_both_are_measured(self):
        assert set(self._report().conformance) == {"repo-wide", "dsa-local"}

    def test_the_conflicts_are_reported(self):
        report = self._report()
        assert len(report.conflicts) == 3
        assert {c["kind"] for c in report.conflicts} == {"filename", "tags", "frontmatter"}

    def test_it_still_refuses_to_choose(self):
        assert "UNRESOLVED" in self._report().resolution_status
