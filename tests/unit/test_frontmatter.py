"""Frontmatter parsing, diagnostics, and repair proposals.

The two malformed shapes tested here are the exact strings found in the real
corpus, not invented examples.
"""

from __future__ import annotations

import pytest
import yaml

from forge.parsing.frontmatter import (
    DiagnosticCode,
    Severity,
    extract_wikilink_values,
    parse_frontmatter,
)

# Verbatim from DSA/09_CheatSheets/DFS Cheat Sheet.md
PARSE_ERROR_FM = """type: cheat-sheet
status: stable
tags: [dsa/cheat-sheet, dsa/dfs]
canonical: true
related: [[Pattern Index]], [[Template Index]], [[Complexities Cheat Sheet]]"""

# Verbatim from DSA/01_Patterns/Greedy.md
NESTED_LIST_FM = """type: pattern
status: stable
tags: [dsa/pattern, dsa/greedy]
canonical: true
related: [[[Python - Greedy]], [[Heap]], [[Sorting]]]"""

VALID_FM = """type: pattern
status: stable
canonical: true"""


class TestParseErrors:
    def test_real_corpus_shape_fails_yaml(self):
        """Confirms the defect exists as characterized, not just as claimed."""
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(PARSE_ERROR_FM)

    def test_parse_error_is_reported_as_error(self):
        result = parse_frontmatter(PARSE_ERROR_FM)
        assert not result.valid
        assert result.has_errors
        codes = {d.code for d in result.diagnostics}
        assert DiagnosticCode.YAML_PARSE_ERROR in codes

    def test_parse_error_yields_verified_repair(self):
        result = parse_frontmatter(PARSE_ERROR_FM)
        assert len(result.repairs) == 1
        repair = result.repairs[0]
        assert repair.key == "related"
        assert repair.verified
        assert repair.proposed.strip() == (
            'related: ["[[Pattern Index]]", "[[Template Index]]", "[[Complexities Cheat Sheet]]"]'
        )

    def test_repaired_text_actually_parses(self):
        result = parse_frontmatter(PARSE_ERROR_FM)
        assert result.repaired_text is not None
        data = yaml.safe_load(result.repaired_text)
        assert data["related"] == ["[[Pattern Index]]", "[[Template Index]]", "[[Complexities Cheat Sheet]]"]


class TestNestedLists:
    def test_nested_list_parses_but_is_wrong(self):
        """The 215-file case: valid YAML, semantically useless.

        `related: [[[A]], [[B]]]` yields a list whose every element is itself a
        singleton list containing a singleton list — the wikilink brackets are
        read as two levels of YAML flow sequence.
        """
        data = yaml.safe_load(NESTED_LIST_FM)
        assert data["related"] == [[["Python - Greedy"]], [["Heap"]], [["Sorting"]]]

    def test_reported_as_warning_not_error(self):
        result = parse_frontmatter(NESTED_LIST_FM)
        assert result.valid  # it does parse
        codes = {d.code for d in result.diagnostics}
        assert DiagnosticCode.NESTED_LIST_WIKILINKS in codes
        severities = {d.severity for d in result.diagnostics}
        assert Severity.ERROR not in severities

    def test_flattening_repair_is_proposed_and_verified(self):
        result = parse_frontmatter(NESTED_LIST_FM)
        assert result.repairs and result.repairs[0].verified
        data = yaml.safe_load(result.repaired_text)
        assert data["related"] == ["[[Python - Greedy]]", "[[Heap]]", "[[Sorting]]"]


class TestValidAndAbsent:
    def test_valid_frontmatter_has_no_repairs(self):
        result = parse_frontmatter(VALID_FM)
        assert result.valid and not result.repairs and not result.has_errors

    def test_absent_frontmatter_is_info_not_error(self):
        result = parse_frontmatter(None)
        assert not result.present
        assert result.diagnostics[0].code is DiagnosticCode.NO_FRONTMATTER
        assert result.diagnostics[0].severity is Severity.INFO

    def test_empty_frontmatter(self):
        result = parse_frontmatter("   \n  ")
        assert result.present and not result.valid
        assert result.diagnostics[0].code is DiagnosticCode.EMPTY_FRONTMATTER

    def test_duplicate_keys_are_detected(self):
        result = parse_frontmatter("type: a\nstatus: x\ntype: b")
        codes = {d.code for d in result.diagnostics}
        assert DiagnosticCode.DUPLICATE_KEY in codes

    def test_non_mapping_frontmatter(self):
        result = parse_frontmatter("- just\n- a\n- list")
        assert not result.valid
        assert {d.code for d in result.diagnostics} == {DiagnosticCode.NOT_A_MAPPING}


class TestTruncatedWikilinks:
    """A third defect shape, found in 18 corpus files during Phase 1.

    The Phase 0 audit characterized only the parse-error and nested-list forms.
    Verbatim from DSA/03_DataStructures/AVL Tree.md — note the single closing
    bracket on the final link.
    """

    TRUNCATED_FM = """type: data-structure
status: stable
canonical: true
related: [[Binary Search Tree]], [[Tree Traversal]"""

    def test_reported_as_its_own_code(self):
        result = parse_frontmatter(self.TRUNCATED_FM)
        assert DiagnosticCode.TRUNCATED_WIKILINK in {d.code for d in result.diagnostics}

    def test_truncated_link_is_recovered_not_dropped(self):
        result = parse_frontmatter(self.TRUNCATED_FM)
        assert result.repairs[0].verified
        data = yaml.safe_load(result.repaired_text)
        assert data["related"] == ["[[Binary Search Tree]]", "[[Tree Traversal]]"]

    def test_text_extraction_also_recovers_it(self):
        assert extract_wikilink_values(self.TRUNCATED_FM, "related") == [
            "Binary Search Tree",
            "Tree Traversal",
        ]

    def test_single_truncated_link(self):
        assert extract_wikilink_values("related: [[Only One]", "related") == ["Only One"]

    def test_truncation_mid_value_is_not_recovered(self):
        """Only a trailing truncation is unambiguous; anything else is left alone."""
        result = parse_frontmatter("related: [[A] and [[B]]")
        assert result.repairs == []


class TestRepairSafety:
    def test_no_repair_when_value_has_other_content(self):
        """A mechanical rewrite must not silently drop mixed-in content."""
        result = parse_frontmatter('related: [[A]] and also some prose')
        assert result.repairs == []

    def test_quotes_in_names_are_escaped(self):
        result = parse_frontmatter('related: [[He said "hi"]]')
        assert result.repairs
        assert yaml.safe_load(result.repaired_text)["related"] == ['[[He said "hi"]]']


class TestWikilinkRecovery:
    def test_recovers_links_from_unparseable_yaml(self):
        """The corpus's related: graph is readable without repairing any file."""
        assert extract_wikilink_values(PARSE_ERROR_FM, "related") == [
            "Pattern Index",
            "Template Index",
            "Complexities Cheat Sheet",
        ]

    def test_recovers_from_nested_form(self):
        assert extract_wikilink_values(NESTED_LIST_FM, "related") == [
            "Python - Greedy",
            "Heap",
            "Sorting",
        ]

    def test_missing_key_returns_empty(self):
        assert extract_wikilink_values(VALID_FM, "related") == []


class TestRepairPreservesTheLinkGraph:
    """The repair must not destroy the links it is repairing.

    `related:` is read in two places by *text extraction* of `[[...]]` from the
    raw frontmatter — `extract_wikilink_values`, which is how `CorpusIndexer`
    builds the related graph, and `parse_markdown`, which counts frontmatter
    wikilinks. Neither falls back to the parsed YAML.

    An earlier repair emitted `related: ["A", "B"]`. Valid YAML, and it silently
    returned both readers to zero: measured on the corpus, applying it would
    have dropped 746 `related:` edges. Quoting the wikilink whole keeps every
    reader working, and keeps the field an Obsidian property link.
    """

    BROKEN = "type: pattern\nrelated: [[[Python - BFS]], [[Graph Traversal]]]"

    def _repaired(self):
        result = parse_frontmatter(self.BROKEN)
        assert result.repairs and result.repairs[0].verified
        return result.repaired_text

    def test_the_repair_is_valid_yaml_and_a_real_list(self):
        import yaml

        data = yaml.safe_load(self._repaired())
        assert data["related"] == ["[[Python - BFS]]", "[[Graph Traversal]]"]

    def test_text_extraction_still_finds_the_links(self):
        assert extract_wikilink_values(self._repaired(), "related") == [
            "Python - BFS",
            "Graph Traversal",
        ]

    def test_the_markdown_parser_still_sees_frontmatter_wikilinks(self):
        from forge.parsing.markdown import parse_markdown

        doc = f"---\n{self._repaired()}\n---\n\n# Title\n\nbody\n"
        found = [w.target for w in parse_markdown(doc).wikilinks if w.in_frontmatter]
        assert found == ["Python - BFS", "Graph Traversal"]

    def test_the_repaired_file_reports_no_diagnostics(self):
        assert parse_frontmatter(self._repaired()).diagnostics == []

    def test_repairing_twice_changes_nothing(self):
        """The repaired form must be a fixed point, or repeated runs churn."""
        once = self._repaired()
        assert parse_frontmatter(once).repairs == []
