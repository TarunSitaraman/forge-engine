"""Rendering diagnostics for people who are not at a terminal.

Two properties carry the weight: the numbers must match the payload they came
from, and vault content must never execute. A note called `<script>.md` is a
legal filename, and its own health report must not run it.
"""

from __future__ import annotations

import pytest

from forge.reporting import render_html, render_markdown
from forge.reporting.summary import headline

CLEAN = {
    "links": {
        "summary": {
            "total_links": 100,
            "wikilinks": 90,
            "markdown_links": 10,
            "unresolved_occurrences": 0,
            "unresolved_distinct_targets": 0,
        },
        "unresolved_targets": {},
    },
    "frontmatter": {
        "summary": {"total_files": 20, "with_frontmatter": 20, "valid": 20, "invalid": 0,
                    "without_frontmatter": 0, "repairable_files": 0},
        "by_code": {},
        "code_descriptions": {},
        "coverage_by_folder": {},
    },
    "conventions": {"resolution_status": "NOT APPLICABLE", "conflicts": [], "conformance": {}},
}

BROKEN = {
    "links": {
        "summary": {
            "total_links": 50,
            "wikilinks": 50,
            "markdown_links": 0,
            "unresolved_occurrences": 7,
            "unresolved_distinct_targets": 2,
        },
        "unresolved_targets": {
            "Missing Page": {"count": 5, "sources": ["a.md", "b.md", "c.md", "d.md"]},
            "Other": {"count": 2, "sources": ["e.md"]},
        },
    },
    "frontmatter": {
        "summary": {"total_files": 10, "with_frontmatter": 4, "valid": 3, "invalid": 1,
                    "without_frontmatter": 6, "repairable_files": 0},
        "by_code": {"FM003": 6, "FM001": 1},
        "code_descriptions": {"FM003": "No frontmatter block.", "FM001": "Invalid YAML."},
        "coverage_by_folder": {"notes": {"total": 8, "with_fm": 2}},
    },
    "conventions": {
        "resolution_status": "UNRESOLVED",
        "conflicts": [
            {"kind": "filename", "repo_wide": "kebab-case.md", "dsa_local": "Title Case.md"}
        ],
        "conformance": {},
    },
}


class TestHeadline:
    def test_it_reads_the_numbers_off_the_payload(self):
        facts = headline(BROKEN)
        assert facts.files == 10
        assert facts.links == 50
        assert facts.dead_links == 7
        assert facts.dead_link_targets == 2
        assert facts.without_frontmatter == 6
        assert facts.invalid_frontmatter == 1
        assert facts.convention_conflicts == 1

    def test_a_clean_vault_is_clean(self):
        assert headline(CLEAN).clean is True

    def test_a_broken_one_is_not(self):
        assert headline(BROKEN).clean is False

    def test_an_empty_payload_does_not_divide_by_zero(self):
        facts = headline({})
        assert facts.files == 0
        assert facts.clean is True

    def test_graph_rows_are_omitted_when_the_graph_was_never_populated(self):
        """`forge index` alone creates no concepts. Reporting '0 errors' over
        nothing checked reads as a pass when nothing was examined."""
        payload = dict(CLEAN)
        payload["graph"] = {"errors": 0, "checked": {"concepts": 0, "claims": 0}}
        assert not any("Graph" in label for label, _, _ in headline(payload).rows())

    def test_graph_rows_appear_once_it_has_content(self):
        payload = dict(CLEAN)
        payload["graph"] = {"errors": 2, "checked": {"concepts": 5, "claims": 9}}
        facts = headline(payload)
        assert any("Graph" in label for label, _, _ in facts.rows())
        assert facts.clean is False


class TestHtmlIsSelfContained:
    def test_no_external_resources_are_referenced(self):
        """A report that needs the network to render is not a report you can
        attach to an email or open in two years."""
        out = render_html(BROKEN)
        for token in ("src=", "<link", "@import", "cdn.", "http://"):
            assert token not in out, f"{token!r} makes the report non-portable"

    def test_it_is_a_complete_document(self):
        out = render_html(BROKEN)
        assert out.startswith("<!doctype html>")
        assert "</html>" in out
        assert "<style>" in out

    def test_it_names_the_vault(self):
        assert "my-notes" in render_html(BROKEN, vault_name="my-notes")


class TestVaultContentIsEscaped:
    """Filenames and link targets come from an arbitrary folder on disk."""

    @pytest.fixture
    def hostile(self):
        payload = {
            "links": {
                "summary": {
                    "total_links": 1,
                    "unresolved_occurrences": 1,
                    "unresolved_distinct_targets": 1,
                },
                "unresolved_targets": {
                    "<script>alert(1)</script>": {
                        "count": 1,
                        "sources": ['a" onerror="alert(2)'],
                    }
                },
            },
            "frontmatter": {"summary": {"total_files": 1}, "by_code": {}, "code_descriptions": {}},
        }
        return render_html(payload, vault_name="<img src=x onerror=alert(3)>")

    def test_a_script_tag_in_a_link_target_does_not_survive(self, hostile):
        assert "<script>alert(1)" not in hostile
        assert "&lt;script&gt;" in hostile

    def test_an_attribute_break_out_is_escaped(self, hostile):
        assert 'a" onerror="alert(2)' not in hostile

    def test_the_vault_name_is_escaped_too(self, hostile):
        assert "<img src=x" not in hostile

    def test_the_only_script_free_output_is_not_achieved_by_dropping_content(self, hostile):
        """Escaping, not deletion: the reader still needs to see the target."""
        assert "alert(1)" in hostile


class TestMarkdown:
    def test_it_reports_the_dead_links(self):
        out = render_markdown(BROKEN)
        assert "Missing Page" in out
        assert "## Dead links" in out

    def test_a_clean_vault_has_no_dead_link_section(self):
        assert "## Dead links" not in render_markdown(CLEAN)

    def test_it_carries_the_install_line(self):
        """The report is the distribution channel; a reader who likes it must
        be able to find the tool from the artifact alone."""
        assert "pip install forge-kb" in render_markdown(CLEAN)

    def test_both_renderers_agree_on_the_numbers(self):
        """One summary, two skins -- they must not drift."""
        facts = headline(BROKEN)
        md, html = render_markdown(BROKEN), render_html(BROKEN)
        for text in (md, html):
            assert f"{facts.without_frontmatter} of {facts.files}" in text
