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


class TestInboundLinks:
    """Counting what points at each concept, over every page in the vault.

    The graph cannot answer "does anything link here?", because the pages that
    do the linking in a hub-and-spoke vault are not concepts. Measured on the
    real corpus 2026-09-07: 115 links pointed at the 72 concepts the gap report
    called isolated, and not one came from a page the graph counts.
    """

    def test_a_link_from_a_page_that_is_not_a_concept_still_counts(self):
        """The whole point. `_index.md` is not a node, but it is a page, and a
        doc reachable from its folder's index is reachable."""
        index = CorpusIndex(
            vault_path=".",
            files=[
                _file("Technologies/Docs/_index.md", [_link("azure", "Technologies/Docs/azure.md")]),
                _file("Technologies/Docs/azure.md"),
            ],
        )
        plan = build_plan(index)
        concept = next(c for c in plan.concepts if c.vault_path.endswith("azure.md"))

        assert plan.links == [], "a hub must not become a node"
        assert plan.inbound_links[concept.id][0] == 1
        assert plan.inbound_links[concept.id][1] == ("Technologies/Docs/_index.md",)
        assert plan.unreferenced() == 0

    def test_distinct_pages_are_counted_not_link_occurrences(self):
        """The question is whether anything points here, and a page linking
        five times knows about it once."""
        index = CorpusIndex(
            vault_path=".",
            files=[
                _file("A.md", [_link("B", "B.md"), _link("B", "B.md"), _link("B", "B.md")]),
                _file("B.md"),
            ],
        )
        plan = build_plan(index)
        b = next(c for c in plan.concepts if c.vault_path == "B.md")
        assert plan.inbound_links[b.id][0] == 1

    def test_a_page_linking_to_itself_does_not_make_itself_reachable(self):
        index = CorpusIndex(vault_path=".", files=[_file("A.md", [_link("A", "A.md")])])
        plan = build_plan(index)
        a = plan.concepts[0]
        assert plan.inbound_links[a.id] == (0, ())
        assert plan.unreferenced() == 1

    def test_an_unresolved_link_points_at_nothing_and_counts_as_nothing(self):
        index = CorpusIndex(
            vault_path=".",
            files=[_file("A.md", [_link("B", None, LinkStatus.MISSING)]), _file("B.md")],
        )
        plan = build_plan(index)
        b = next(c for c in plan.concepts if c.vault_path == "B.md")
        assert plan.inbound_links[b.id][0] == 0

    def test_every_concept_gets_a_row_including_the_unlinked_ones(self):
        """A concept with no row would be indistinguishable from one nobody
        counted, which is the distinction the whole table exists to make."""
        index = CorpusIndex(vault_path=".", files=[_file("A.md"), _file("B.md")])
        plan = build_plan(index)
        assert set(plan.inbound_links) == {c.id for c in plan.concepts}
        assert plan.to_dict()["concepts_nothing_links_to"] == 2


class TestTheCommandWritesThem:
    """Driven through the CLI, not the library.

    Every unit test above calls `build_plan` directly, so none of them would
    notice if `forge bootstrap --apply` stored the concepts and forgot the
    counts — the same shape of gap that let `forge backup` ship with a
    NameError while every unit test passed.
    """

    def _run(self, vault):
        from forge.cli.main import app
        from typer.testing import CliRunner

        runner = CliRunner()
        for argv in (
            ["index", "--vault", str(vault)],
            ["bootstrap", "--vault", str(vault), "--apply"],
        ):
            result = runner.invoke(app, argv)
            assert result.exit_code == 0, result.output
        return result.output

    def test_apply_records_the_inbound_counts(self, fixture_vault):
        from forge.config import Settings
        from forge.storage import SqliteStore

        self._run(fixture_vault)

        store = SqliteStore(Settings.load(fixture_vault).db_path)
        store.initialize()
        try:
            assert store.inbound_counted(), "bootstrap --apply stored no counts"
            assert store.unreferenced_concepts() is not None
        finally:
            store.close()

    def test_a_preview_reports_what_nothing_links_to(self, fixture_vault):
        from forge.cli.main import app
        from typer.testing import CliRunner

        self._run(fixture_vault)
        output = CliRunner().invoke(
            app, ["bootstrap", "--vault", str(fixture_vault)]
        ).output

        assert "unreferenced" in output
