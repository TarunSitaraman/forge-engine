"""Deterministic graph seeding from vault structure.

The premise: this vault's concepts are its filenames. A human decided
`Binary Search` deserves one canonical home and created the page, and that is
the judgement LLM extraction was failing to reproduce — measured 2026-08-20, it
returned `RAM`, `Answer`, `Fluency` and `VARCHAR(n)` as concepts.
"""

from __future__ import annotations

import pytest

from forge.bootstrap import build_plan, is_concept_page, kind_for
from forge.bootstrap.seed import BOOTSTRAP_VERSION
from forge.corpus.model import CorpusIndex, IndexedFile
from forge.domain import ConceptKind, Derivation, LinkType, ProvenanceTier
from forge.parsing.links import LinkStatus, ResolvedLink


def _file(path, links=()):
    return IndexedFile(
        path=path,
        content_hash="h",
        byte_size=1,
        line_count=1,
        title=None,
        doc_type=None,
        status=None,
        canonical=False,
        tags=(),
        related=(),
        frontmatter_present=False,
        frontmatter_valid=True,
        frontmatter_keys=(),
        heading_count=0,
        wikilink_count=0,
        markdown_link_count=0,
        code_block_count=0,
        code_languages=(),
        links=tuple(links),
        diagnostics=(),
        repairs=(),
    )


def _link(target, resolved, status=LinkStatus.RESOLVED, in_fm=False):
    return ResolvedLink(
        source_path="src.md",
        target=target,
        status=status,
        line=1,
        resolved_path=resolved,
        in_frontmatter=in_fm,
    )


class TestPageSelection:
    @pytest.mark.parametrize(
        "path",
        [
            "DSA/01_Patterns/Heap.md",
            "Technologies/Docs/redis.md",
            "DSA/04_Problems/BFS - Level Order.md",
        ],
    )
    def test_content_pages_are_concepts(self, path):
        assert is_concept_page(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "Technologies/Docs/_index.md",          # navigation hub
            "DSA/00_Index/Pattern Index.md",        # a folder of hubs
            "README.md",
            "Projects/smartresq/01-overview.md",    # a chapter, not a concept
            "Projects/quickcover/10-roadmap.md",
            "FORGE_COMPLETION_STATUS.md",           # point-in-time artifact
            "Archive/old-thing.md",
            "Inbox/scratch.md",
        ],
    )
    def test_navigation_chapters_and_artifacts_are_not(self, path):
        assert is_concept_page(path) is False

    def test_numbered_chapters_would_otherwise_collide_across_packs(self):
        """`01-overview` exists in six project packs; none of them is a concept."""
        assert is_concept_page("Projects/a/01-overview.md") is False
        assert is_concept_page("Projects/b/01-overview.md") is False


class TestKinds:
    @pytest.mark.parametrize(
        "path,kind",
        [
            ("DSA/01_Patterns/Heap.md", ConceptKind.PATTERN),
            ("DSA/02_Algorithms/Quick Sort.md", ConceptKind.ALGORITHM),
            ("DSA/03_DataStructures/Trie.md", ConceptKind.DATA_STRUCTURE),
            ("Technologies/Docs/redis.md", ConceptKind.TECHNOLOGY),
            ("Technologies/Playbooks/deployment.md", ConceptKind.PLAYBOOK),
            ("Projects/smartresq/architecture.md", ConceptKind.PROJECT),
        ],
    )
    def test_folder_implies_kind(self, path, kind):
        assert kind_for(path) is kind

    def test_an_unmapped_folder_falls_back_rather_than_raising(self):
        assert kind_for("Somewhere/New/thing.md") is ConceptKind.CONCEPT


class TestProvenance:
    def test_concepts_are_user_assertions_derived_deterministically(self):
        plan = build_plan(CorpusIndex(vault_path=".", files=[_file("DSA/01_Patterns/Heap.md")]))
        c = plan.concepts[0]
        assert c.provenance.tier is ProvenanceTier.USER_ASSERTION
        assert c.provenance.derivation is Derivation.DETERMINISTIC
        assert c.provenance.agent == BOOTSTRAP_VERSION
        assert c.vault_path == "DSA/01_Patterns/Heap.md"

    def test_no_model_is_involved(self):
        """The whole point: this replaces 3,372 model calls with zero."""
        from forge.llm.base import CALLS

        CALLS.reset()
        build_plan(CorpusIndex(vault_path=".", files=[_file("DSA/01_Patterns/Heap.md")]))
        assert CALLS.count == 0


class TestCollisions:
    PAGES = ["DSA/01_Patterns/Heap.md", "DSA/03_DataStructures/Heap.md"]

    def test_an_undecided_collision_creates_no_concept(self):
        """The engine must not pick which `Heap` the user meant."""
        plan = build_plan(CorpusIndex(vault_path=".", files=[_file(p) for p in self.PAGES]))
        assert plan.concepts == []
        assert plan.undecided_collisions == {"Heap": sorted(self.PAGES)}

    def test_a_decided_collision_creates_both_under_namespaces(self):
        plan = build_plan(
            CorpusIndex(vault_path=".", files=[_file(p) for p in self.PAGES]),
            decided={"heap": "DSA/01_Patterns/Heap.md"},
        )
        assert plan.undecided_collisions == {}
        namespaces = sorted(c.namespace for c in plan.concepts)
        assert namespaces == ["data-structure", "pattern"]
        assert len({c.id for c in plan.concepts}) == 2

    def test_an_uncollided_name_gets_no_namespace(self):
        plan = build_plan(CorpusIndex(vault_path=".", files=[_file("DSA/01_Patterns/Heap.md")]))
        assert plan.concepts[0].namespace is None


class TestEdges:
    def test_a_resolved_link_becomes_a_scored_related_to_edge(self):
        index = CorpusIndex(
            vault_path=".",
            files=[
                _file("A.md", [_link("B", "B.md")]),
                _file("B.md"),
            ]
        )
        plan = build_plan(index)
        assert len(plan.links) == 1
        edge = plan.links[0]
        assert edge.type is LinkType.RELATED_TO
        assert edge.score == 1.0
        assert "not a computed similarity" in edge.rationale

    def test_related_to_is_the_only_type_that_is_both_graph_valid_and_deterministic(self):
        """Documents why the edge type is not a free choice."""
        from forge.domain.enums import DETERMINISTIC_LINK_TYPES
        from forge.graph.graph import SUPPORTED_GRAPH_TYPES

        assert SUPPORTED_GRAPH_TYPES & DETERMINISTIC_LINK_TYPES == {LinkType.RELATED_TO}

    def test_repeated_links_between_two_pages_make_one_edge(self):
        index = CorpusIndex(
            vault_path=".",
            files=[
                _file("A.md", [_link("B", "B.md"), _link("B", "B.md"), _link("B", "B.md")]),
                _file("B.md"),
            ]
        )
        assert len(build_plan(index).links) == 1

    def test_self_links_are_dropped(self):
        index = CorpusIndex(vault_path=".", files=[_file("A.md", [_link("A", "A.md")])])
        assert build_plan(index).links == []

    def test_unresolved_links_make_no_edge(self):
        index = CorpusIndex(
            vault_path=".",
            files=[_file("A.md", [_link("Nope", None, LinkStatus.MISSING)]), _file("B.md")]
        )
        assert build_plan(index).links == []

    def test_links_into_excluded_pages_make_no_edge(self):
        """A link to an index hub must not put the hub in the graph."""
        index = CorpusIndex(
            vault_path=".",
            files=[
                _file("A.md", [_link("_index", "Technologies/Docs/_index.md")]),
                _file("Technologies/Docs/_index.md"),
            ]
        )
        plan = build_plan(index)
        assert plan.links == []
        assert all(c.vault_path != "Technologies/Docs/_index.md" for c in plan.concepts)

    def test_frontmatter_links_are_marked_as_such_in_the_rationale(self):
        index = CorpusIndex(
            vault_path=".",
            files=[_file("A.md", [_link("B", "B.md", in_fm=True)]), _file("B.md")]
        )
        assert "related: field" in build_plan(index).links[0].rationale
