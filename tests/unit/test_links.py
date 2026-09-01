"""Wikilink and Markdown-link resolution and classification.

Includes the known-hard stem-collision cases (Heap, Binary Search, Trie) that
the real corpus contains, and the URL-decoding case that produced two false
positives during the Phase 0 audit.
"""

from __future__ import annotations

import pytest

from forge.corpus.indexer import CorpusIndexer
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


class TestIdentityDecisionsResolveAmbiguity:
    """A recorded decision must outrank the ambiguity it was recorded to settle.

    `DSA/01_Patterns/Heap.md` and `DSA/03_DataStructures/Heap.md` both exist and
    both should, so a bare `[[Heap]]` is genuinely ambiguous and the engine must
    not guess. But `forge identity decide` exists precisely so a human can say
    what the bare name means — and until 2026-08-27 the link resolver never read
    that answer. On the real corpus this accounted for 180 of 274 unresolved
    link occurrences: Heap 74, Binary Search 66, Trie 40.
    """

    PATHS = [
        "DSA/01_Patterns/Heap.md",
        "DSA/03_DataStructures/Heap.md",
        "DSA/01_Patterns/DFS.md",
    ]

    def _link(self, target, line=1):
        return wl(target, line)

    def test_without_a_decision_it_stays_ambiguous(self):
        index = LinkIndex.build(self.PATHS)
        got = resolve_wikilink(self._link("Heap"), "src.md", index)
        assert got.status is LinkStatus.AMBIGUOUS
        assert len(got.candidates) == 2

    def test_a_decision_resolves_it(self):
        index = LinkIndex.build(self.PATHS, decided={"heap": "DSA/01_Patterns/Heap.md"})
        got = resolve_wikilink(self._link("Heap"), "src.md", index)
        assert got.status is LinkStatus.RESOLVED
        assert got.resolved_path == "DSA/01_Patterns/Heap.md"

    def test_the_decision_is_matched_regardless_of_casing(self):
        index = LinkIndex.build(self.PATHS, decided={"heap": "DSA/01_Patterns/Heap.md"})
        got = resolve_wikilink(self._link("heap"), "src.md", index)
        assert got.resolved_path == "DSA/01_Patterns/Heap.md"

    def test_a_decision_naming_a_file_that_is_not_a_candidate_is_ignored(self):
        """Guard against a stale decision silently retargeting a link."""
        index = LinkIndex.build(self.PATHS, decided={"heap": "DSA/99_Gone/Heap.md"})
        got = resolve_wikilink(self._link("Heap"), "src.md", index)
        assert got.status is LinkStatus.AMBIGUOUS

    def test_unrelated_links_are_unaffected(self):
        index = LinkIndex.build(self.PATHS, decided={"heap": "DSA/01_Patterns/Heap.md"})
        got = resolve_wikilink(self._link("DFS"), "src.md", index)
        assert got.status is LinkStatus.RESOLVED
        assert got.resolved_path == "DSA/01_Patterns/DFS.md"


class TestLinksToNonMarkdownFiles:
    """A link to real config, code or an image is a link, not a defect.

    The index is built from `.md` paths, so `](config/concept-identity.yaml)`
    had no entry and was reported MISSING even though it resolves on GitHub and
    in Obsidian. A checker that reports working links as broken trains people to
    ignore it, which costs more than the false positive itself.
    """

    def _md_link(self, target):
        from forge.parsing.markdown import MarkdownLink

        return MarkdownLink(text="t", target=target, line=1)

    def test_a_real_non_markdown_file_resolves(self):
        index = LinkIndex.build(
            ["README.md"], other_files={"config/concept-identity.yaml"}
        )
        got = resolve_markdown_link(self._md_link("config/concept-identity.yaml"), "README.md", index)
        assert got.status is LinkStatus.RESOLVED

    def test_a_non_markdown_file_that_does_not_exist_is_still_missing(self):
        index = LinkIndex.build(["README.md"], other_files={"config/real.yaml"})
        got = resolve_markdown_link(self._md_link("config/imaginary.yaml"), "README.md", index)
        assert got.status is LinkStatus.MISSING

    def test_markdown_targets_are_unaffected(self):
        index = LinkIndex.build(["README.md", "docs/a.md"], other_files={"x.png"})
        got = resolve_markdown_link(self._md_link("docs/a.md"), "README.md", index)
        assert got.status is LinkStatus.RESOLVED
        assert got.resolved_path == "docs/a.md"


class TestDecidedIdentitiesAreReadFromTheVault:
    """The identity config is found relative to the vault, not to the cwd.

    `DEFAULT_CONFIG_PATH` is a relative path. `CorpusIndexer._decided_targets`
    used to call `IdentityConfig.load()` with no argument, which resolved it
    against the *process working directory* — so `forge index` run from
    anywhere but the vault root read no config, and every collision a human had
    already decided went back to being reported AMBIGUOUS. Every other caller
    passed an explicit vault-relative path; this was the one that did not.

    Found when the engine moved to its own repository and the corpus tests
    started running with a cwd outside the vault for the first time.
    """

    DECIDED = """version: 1
collisions:
- name: Heap
  identities:
  - canonical_name: Heap
    kind: pattern
    namespace: pattern
    vault_path: DSA/01_Patterns/Heap.md
  - canonical_name: Heap
    kind: data_structure
    namespace: data-structure
    vault_path: DSA/03_DataStructures/Heap.md
  default: pattern/Heap
  decided_by: test
  decided_at: '2026-09-01T00:00:00+00:00'
"""

    def _heap_link(self, index):
        source = index.by_path()["DSA/01_Patterns/Graph Traversal.md"]
        return next(link for link in source.links if link.target == "Heap")

    def test_the_decision_is_honoured_from_an_unrelated_cwd(
        self, settings, fixture_vault, tmp_path, monkeypatch
    ):
        config = fixture_vault / "config" / "concept-identity.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(self.DECIDED, encoding="utf-8")

        elsewhere = tmp_path / "somewhere-else"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        got = self._heap_link(CorpusIndexer(settings).build_index())

        assert got.status is LinkStatus.RESOLVED
        assert got.resolved_path == "DSA/01_Patterns/Heap.md"

    def test_an_undecided_collision_still_stays_ambiguous(
        self, settings, fixture_vault, tmp_path, monkeypatch
    ):
        """The fix must not make the engine start guessing."""
        elsewhere = tmp_path / "somewhere-else"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        got = self._heap_link(CorpusIndexer(settings).build_index())

        assert got.status is LinkStatus.AMBIGUOUS
