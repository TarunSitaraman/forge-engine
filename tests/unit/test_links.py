"""Wikilink and Markdown-link resolution and classification.

Includes the known-hard stem-collision cases (Heap, Binary Search, Trie) that
the real corpus contains, and the URL-decoding case that produced two false
positives during the Phase 0 audit.
"""

from __future__ import annotations

import pytest

from forge.parsing.links import LinkIndex, LinkStatus, normalize, resolve_markdown_link, resolve_wikilink
from forge.parsing.markdown import MarkdownLink, WikiLink

PATHS = [
    "DSA/01_Patterns/DFS.md",
    "DSA/01_Patterns/BFS.md",
    "DSA/01_Patterns/Graph Traversal.md",
    "DSA/01_Patterns/Heap.md",
    "DSA/03_DataStructures/Heap.md",
    "DSA/01_Patterns/Binary Search.md",
    "DSA/02_Algorithms/Binary Search.md",
    "Notes/plain-note.md",
    "README.md",
]


@pytest.fixture
def index() -> LinkIndex:
    return LinkIndex.build(PATHS)


def wl(target: str, line: int = 1) -> WikiLink:
    return WikiLink(raw=target, target=target, anchor=None, alias=None, line=line)


class TestResolution:
    def test_exact_stem(self, index):
        r = resolve_wikilink(wl("DFS"), "Notes/plain-note.md", index)
        assert r.status is LinkStatus.RESOLVED
        assert r.resolved_path == "DSA/01_Patterns/DFS.md"

    def test_stem_with_spaces(self, index):
        r = resolve_wikilink(wl("Graph Traversal"), "Notes/plain-note.md", index)
        assert r.status is LinkStatus.RESOLVED

    def test_case_mismatch(self, index):
        r = resolve_wikilink(wl("dfs"), "Notes/plain-note.md", index)
        assert r.status is LinkStatus.CASE_MISMATCH
        assert r.resolved_path == "DSA/01_Patterns/DFS.md"

    def test_full_path_target(self, index):
        r = resolve_wikilink(wl("DSA/01_Patterns/DFS"), "README.md", index)
        assert r.status is LinkStatus.RESOLVED

    def test_path_mismatch(self, index):
        """Right file, wrong folder."""
        r = resolve_wikilink(wl("Wrong/Folder/DFS"), "README.md", index)
        assert r.status is LinkStatus.PATH_MISMATCH
        assert r.resolved_path == "DSA/01_Patterns/DFS.md"

    def test_renamed_candidate_via_normalization(self, index):
        r = resolve_wikilink(wl("graph-traversal"), "README.md", index)
        assert r.status is LinkStatus.RENAMED_CANDIDATE
        assert r.candidates == ("DSA/01_Patterns/Graph Traversal.md",)

    def test_missing_offers_candidates_but_does_not_guess(self, index):
        r = resolve_wikilink(wl("Graph Traversals"), "README.md", index)
        assert r.status in (LinkStatus.MISSING, LinkStatus.RENAMED_CANDIDATE)
        assert r.resolved_path is None  # never silently picked

    def test_completely_missing(self, index):
        r = resolve_wikilink(wl("Nonexistent Page Xyzzy"), "README.md", index)
        assert r.status is LinkStatus.MISSING
        assert r.resolved_path is None

    @pytest.mark.parametrize("bad", ["", "  ", "1,2,3", "[[", "0-1", "''", "{}", "2,2,2,2"])
    def test_malformed_targets(self, index, bad):
        """Punctuation/digit-only targets are code residue, not links.

        These shapes appear in the real corpus when a naive parser reads Python
        literals as wikilinks; classifying them as MALFORMED keeps them out of
        the "missing page" report, which is meant to list pages worth writing.
        """
        r = resolve_wikilink(wl(bad), "README.md", index)
        assert r.status is LinkStatus.MALFORMED


class TestKnownHardCases:
    """Stem collisions that violate the vault's own one-canonical-home rule."""

    @pytest.mark.parametrize(
        "target,expected",
        [
            ("Heap", ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]),
            (
                "Binary Search",
                ["DSA/01_Patterns/Binary Search.md", "DSA/02_Algorithms/Binary Search.md"],
            ),
        ],
    )
    def test_collisions_are_ambiguous_not_arbitrary(self, index, target, expected):
        r = resolve_wikilink(wl(target), "README.md", index)
        assert r.status is LinkStatus.AMBIGUOUS
        assert list(r.candidates) == expected
        assert r.resolved_path is None, "must not pick one of two canonical homes"

    def test_graph_traversal_dfs_bfs_all_resolve_distinctly(self, index):
        results = {
            t: resolve_wikilink(wl(t), "README.md", index)
            for t in ("Graph Traversal", "DFS", "BFS")
        }
        assert all(r.status is LinkStatus.RESOLVED for r in results.values())
        paths = {r.resolved_path for r in results.values()}
        assert len(paths) == 3, "these three must never collapse onto one another"


class TestMarkdownLinks:
    def test_relative_link(self, index):
        r = resolve_markdown_link(
            MarkdownLink(text="x", target="../DSA/01_Patterns/BFS.md", line=1),
            "Notes/plain-note.md",
            index,
        )
        assert r.status is LinkStatus.RESOLVED

    def test_url_encoded_link_resolves(self, index):
        """The Phase 0 audit's false positive: %20 must be decoded first."""
        r = resolve_markdown_link(
            MarkdownLink(text="x", target="DSA/01_Patterns/Graph%20Traversal.md", line=1),
            "README.md",
            index,
        )
        assert r.status is LinkStatus.RESOLVED
        assert r.resolved_path == "DSA/01_Patterns/Graph Traversal.md"

    def test_directory_link_is_not_broken(self, index):
        """](../DSA/) is a valid directory link with no .md target."""
        r = resolve_markdown_link(
            MarkdownLink(text="x", target="../DSA/", line=1), "Notes/plain-note.md", index
        )
        assert r.status is LinkStatus.RESOLVED

    def test_external_links_ignored(self, index):
        for target in ("https://example.com/x.md", "http://a.b", "mailto:x@y.z", "#anchor"):
            assert (
                resolve_markdown_link(
                    MarkdownLink(text="x", target=target, line=1), "README.md", index
                )
                is None
            )

    def test_broken_relative_link(self, index):
        r = resolve_markdown_link(
            MarkdownLink(text="x", target="./nope.md", line=1), "README.md", index
        )
        assert r.status is LinkStatus.MISSING


class TestNormalize:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("Graph Traversal", "graph-traversal"),
            ("Fast & Slow Pointers", "fast--slow--pointers"),
            ("A_B", "a b"),
        ],
    )
    def test_equivalent_after_normalization(self, a, b):
        assert normalize(a) == normalize(b)
