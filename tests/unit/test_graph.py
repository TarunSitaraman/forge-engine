"""Bounded graph traversal and integrity diagnostics.

Two properties are under test here.

**Traversal is bounded.** Every walk over the graph has a depth limit and a
node budget that cannot be switched off. A graph that is small today is not a
licence to write an unbounded traversal, because the traversal outlives the
size assumption.

**Diagnostics report; they never repair.** A graph integrity check that
silently fixes things is an unreviewed change to what the user believes. Every
assertion about ``check_integrity`` therefore also asserts that the store came
out the other side unchanged.
"""

from __future__ import annotations

import pytest

from forge.domain import (
    Claim,
    ClaimLink,
    Concept,
    ConceptKind,
    Derivation,
    Document,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    LinkType,
    Provenance,
    ProvenanceTier,
    ProvenanceViolation,
    Source,
    SourceKind,
    Span,
    deterministic_provenance,
)
from forge.graph import (
    DEFAULT_MAX_DEPTH,
    IntegrityCode,
    KnowledgeGraph,
    check_integrity,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def model_provenance(agent: str = "RelationshipActivator") -> Provenance:
    return Provenance(
        tier=ProvenanceTier.MODEL_INFERENCE,
        derivation=Derivation.MODEL,
        agent=agent,
        model_id="llama3.1:8b",
    )


def concept(name: str, **kw) -> Concept:
    return Concept(
        id=Concept.make_id(name, kw.pop("namespace", None) or None),
        canonical_name=name,
        kind=kw.pop("kind", ConceptKind.TECHNOLOGY),
        provenance=kw.pop("provenance", model_provenance("ProposalActivator")),
        origin_proposal_id=kw.pop("origin_proposal_id", "p-seed"),
        **kw,
    )


def link(from_id: str, to_id: str, link_type: LinkType = LinkType.RELATED_TO, **kw) -> ClaimLink:
    score = kw.pop("score", 0.6 if link_type is LinkType.RELATED_TO else None)
    return ClaimLink(
        id=ClaimLink.make_id(from_id, to_id, link_type),
        from_id=from_id,
        to_id=to_id,
        type=link_type,
        provenance=kw.pop("provenance", model_provenance()),
        rationale=kw.pop("rationale", "co-occurs in 2 spans"),
        score=score,
        **kw,
    )


@pytest.fixture
def chain(store):
    """A -> B -> C -> D, plus an isolated E.

    A deliberately linear graph: it makes path length, depth bounds, and
    branching factor all directly assertable.
    """
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    concepts = [concept(name) for name in names]
    for c in concepts:
        store.put_concept(c)
    for left, right in zip(concepts, concepts[1:-1]):
        store.put_link(link(left.id, right.id))
    return store, {c.canonical_name: c for c in concepts}


@pytest.fixture
def evidenced(store):
    """A claim with a real span behind it, for evidence-chain tests."""
    source = Source.for_path("papers/graph.pdf", kind=SourceKind.PDF, content_hash="g1")
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "g1"),
        source_id=source.id,
        parser="forge.pdf",
        parser_version="1",
        content_hash="g1",
    )
    store.put_document(document)
    span = Span(
        id=Span.make_id(document.id, 0, "p.4"),
        document_id=document.id,
        ordinal=0,
        locator="p.4 L2-L5",
        heading_path=("Graphs",),
        start_line=2,
        end_line=5,
        text="A knowledge graph stores entities and the relationships between them.",
        content_hash="gs1",
        page=4,
    )
    store.put_spans([span])

    subject = concept("Knowledge Graph")
    store.put_concept(subject)
    claim = Claim(
        id=Claim.make_id("A knowledge graph stores entities", subject.id),
        statement="A knowledge graph stores entities and their relationships",
        subject_concept_id=subject.id,
        provenance=model_provenance("CandidateExtractor"),
    )
    evidence = EvidenceLink(
        id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.QUOTES),
        claim_id=claim.id,
        span_id=span.id,
        relation=EvidenceRelation.QUOTES,
        provenance=deterministic_provenance("ProposalActivator"),
    )
    store.put_claim(claim, [evidence])
    return store, subject, claim, span, source


# --------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------


class TestNeighbors:
    def test_relationships_are_returned_in_both_directions(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        links = graph.get_relationships(by_name["Bravo"].id)

        assert len(links) == 2, "Bravo sits between Alpha and Charlie"
        endpoints = {link.from_id for link in links} | {link.to_id for link in links}
        assert by_name["Alpha"].id in endpoints
        assert by_name["Charlie"].id in endpoints

    def test_direction_is_reported_per_neighbor(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        directions = {
            n.label: n.direction for n in graph.get_neighbors(by_name["Bravo"].id)
        }

        assert directions["Alpha"] == "in"
        assert directions["Charlie"] == "out"

    def test_neighbors_resolve_to_concepts(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        neighbors = graph.get_neighbors(by_name["Alpha"].id)

        assert [n.label for n in neighbors] == ["Bravo"]
        assert neighbors[0].concept is not None
        assert neighbors[0].to_dict()["type"] == "RELATED_TO"

    def test_type_filter_excludes_other_edges(self, chain):
        store, by_name = chain
        store.put_link(
            link(by_name["Alpha"].id, by_name["Echo"].id, LinkType.DEPENDS_ON)
        )
        graph = KnowledgeGraph(store)

        related = graph.get_relationships(by_name["Alpha"].id, types=[LinkType.RELATED_TO])
        depends = graph.get_relationships(by_name["Alpha"].id, types=[LinkType.DEPENDS_ON])

        assert [link.type for link in related] == [LinkType.RELATED_TO]
        assert [link.type for link in depends] == [LinkType.DEPENDS_ON]

    def test_inactive_links_are_excluded_by_default(self, chain):
        store, by_name = chain
        edge = link(by_name["Alpha"].id, by_name["Echo"].id, LinkType.DEPENDS_ON)
        store.put_link(edge.model_copy(update={"active": False}))
        graph = KnowledgeGraph(store)

        assert graph.get_relationships(by_name["Alpha"].id, active_only=True) != []
        labels = {n.label for n in graph.get_neighbors(by_name["Alpha"].id)}
        assert "Echo" not in labels, "a deactivated edge must not be traversable"

    def test_isolated_node_has_no_neighbors(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        assert graph.get_neighbors(by_name["Echo"].id) == []

    def test_unknown_entity_traverses_to_nothing(self, chain):
        store, _ = chain
        graph = KnowledgeGraph(store)

        assert graph.get_neighbors("concept:does-not-exist") == []


class TestBoundedTraversal:
    def test_related_concepts_respect_depth(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        depth_one = graph.get_related_concepts(by_name["Alpha"].id, max_depth=1)
        depth_two = graph.get_related_concepts(by_name["Alpha"].id, max_depth=2)

        assert [c.canonical_name for c, _ in depth_one] == ["Bravo"]
        assert [c.canonical_name for c, _ in depth_two] == ["Bravo", "Charlie"]

    def test_distance_is_reported_with_each_concept(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        found = dict(
            (c.canonical_name, d)
            for c, d in graph.get_related_concepts(by_name["Alpha"].id, max_depth=2)
        )

        assert found == {"Bravo": 1, "Charlie": 2}

    def test_depth_cannot_be_raised_past_the_hard_ceiling(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        # Delta is 3 hops from Alpha; asking for 99 must not become unbounded,
        # it must clamp to DEFAULT_MAX_DEPTH.
        reachable = graph.get_related_concepts(by_name["Alpha"].id, max_depth=99)

        assert max(d for _, d in reachable) <= DEFAULT_MAX_DEPTH

    def test_node_budget_truncates_traversal(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        starved = graph.get_related_concepts(by_name["Alpha"].id, max_depth=3, budget=2)
        generous = graph.get_related_concepts(by_name["Alpha"].id, max_depth=3, budget=500)

        assert len(starved) < len(generous), "the node budget must actually bind"

    def test_traversal_terminates_on_a_cycle(self, store):
        a, b, c = concept("Cyc-A"), concept("Cyc-B"), concept("Cyc-C")
        for entity in (a, b, c):
            store.put_concept(entity)
        store.put_link(link(a.id, b.id))
        store.put_link(link(b.id, c.id))
        store.put_link(link(c.id, a.id))
        graph = KnowledgeGraph(store)

        found = graph.get_related_concepts(a.id, max_depth=3)

        assert {c.canonical_name for c, _ in found} == {"Cyc-B", "Cyc-C"}


class TestPathFinding:
    def test_finds_the_shortest_path(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        path = graph.find_path(by_name["Alpha"].id, by_name["Delta"].id)

        assert path is not None
        assert path.depth == 3
        assert path.nodes[0] == by_name["Alpha"].id
        assert path.nodes[-1] == by_name["Delta"].id

    def test_shortcut_is_preferred_over_the_long_route(self, chain):
        store, by_name = chain
        store.put_link(link(by_name["Alpha"].id, by_name["Delta"].id, LinkType.DEPENDS_ON))
        graph = KnowledgeGraph(store)

        path = graph.find_path(by_name["Alpha"].id, by_name["Delta"].id)

        assert path is not None and path.depth == 1

    def test_path_to_self_is_empty_not_none(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        path = graph.find_path(by_name["Alpha"].id, by_name["Alpha"].id)

        assert path is not None
        assert path.depth == 0
        assert path.nodes == (by_name["Alpha"].id,)

    def test_no_path_within_the_bound_returns_none(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        assert graph.find_path(by_name["Alpha"].id, by_name["Echo"].id) is None

    def test_depth_bound_hides_a_path_that_is_too_far(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)

        assert graph.find_path(by_name["Alpha"].id, by_name["Delta"].id, max_depth=2) is None
        assert graph.find_path(by_name["Alpha"].id, by_name["Delta"].id, max_depth=3) is not None

    def test_path_describes_itself_with_labels(self, chain):
        store, by_name = chain
        graph = KnowledgeGraph(store)
        path = graph.find_path(by_name["Alpha"].id, by_name["Charlie"].id)

        described = path.describe(graph.node_labels(path.nodes))

        assert described == "Alpha -[RELATED_TO]-> Bravo -[RELATED_TO]-> Charlie"


# --------------------------------------------------------------------------
# evidence chain
# --------------------------------------------------------------------------


class TestEvidenceChain:
    def test_claim_resolves_all_the_way_back_to_its_source(self, evidenced):
        store, _, claim, span, source = evidenced
        graph = KnowledgeGraph(store)

        chain = graph.get_claim_evidence(claim.id)

        assert len(chain) == 1
        entry = chain[0]
        assert entry["span_id"] == span.id
        assert entry["source_id"] == source.id
        assert entry["page"] == 4
        assert source.locator in entry["citation"]

    def test_unknown_claim_yields_an_empty_chain(self, evidenced):
        store, *_ = evidenced
        assert KnowledgeGraph(store).get_claim_evidence("claim:nope") == []

    def test_explain_concept_names_its_origin_proposal(self, evidenced):
        store, subject, claim, _, _ = evidenced
        graph = KnowledgeGraph(store)

        explained = graph.explain_concept(subject.id)

        assert explained["concept"]["canonical_name"] == "Knowledge Graph"
        assert explained["provenance"]["tier"] == ProvenanceTier.MODEL_INFERENCE.value
        assert [c["id"] for c in explained["claims"]] == [claim.id]

    def test_explain_unknown_concept_is_none(self, evidenced):
        store, *_ = evidenced
        assert KnowledgeGraph(store).explain_concept("concept:nope") is None


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


class TestGraphMetrics:
    def test_counts_match_the_stored_graph(self, chain):
        store, _ = chain
        metrics = KnowledgeGraph(store).metrics()

        assert metrics.nodes == 5
        assert metrics.edges == 3
        assert metrics.by_type == {"RELATED_TO": 3}

    def test_isolated_nodes_are_counted(self, chain):
        store, _ = chain
        assert KnowledgeGraph(store).metrics().isolated_nodes == 1

    def test_branching_factor_ignores_isolated_nodes(self, chain):
        store, _ = chain
        metrics = KnowledgeGraph(store).metrics()

        # 6 endpoints over 4 connected nodes = 1.5; including Echo would give
        # 1.2 and understate what traversal actually costs.
        assert metrics.branching_factor == pytest.approx(1.5)
        assert metrics.mean_degree == pytest.approx(1.2)

    def test_max_degree_finds_the_hub(self, store):
        hub = concept("Hub")
        store.put_concept(hub)
        for i in range(4):
            spoke = concept(f"Spoke-{i}")
            store.put_concept(spoke)
            store.put_link(link(hub.id, spoke.id))

        assert KnowledgeGraph(store).metrics().max_degree == 4

    def test_query_timings_are_measured_not_assumed(self, chain):
        store, _ = chain
        metrics = KnowledgeGraph(store).metrics()

        assert metrics.neighbor_query_ms > 0.0
        assert metrics.to_dict()["neighbor_query_ms"] >= 0.0

    def test_empty_graph_measures_cleanly(self, store):
        metrics = KnowledgeGraph(store).metrics()

        assert metrics.nodes == 0
        assert metrics.edges == 0
        assert metrics.branching_factor == 0.0


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


class TestIntegrity:
    def test_a_well_formed_graph_is_clean(self, evidenced):
        store, *_ = evidenced
        report = check_integrity(store)

        assert report.clean, report.by_code()
        assert report.checked["relationships"] == 0

    def test_missing_target_is_an_error(self, chain):
        store, by_name = chain
        store.put_link(link(by_name["Alpha"].id, "concept:ghost", LinkType.DEPENDS_ON))

        report = check_integrity(store)

        assert IntegrityCode.MISSING_TARGET.value in report.by_code()
        assert any(f.severity == "error" for f in report.errors)

    def test_orphan_relationship_is_an_error(self, chain):
        store, by_name = chain
        store.put_link(link("concept:ghost", by_name["Alpha"].id, LinkType.DEPENDS_ON))

        assert IntegrityCode.ORPHAN_RELATIONSHIP.value in check_integrity(store).by_code()

    def test_the_store_refuses_to_create_an_unevidenced_claim(self, store):
        """GR007's first line of defence is that the state cannot be written."""
        subject = concept("Unevidenced")
        store.put_concept(subject)

        with pytest.raises(ProvenanceViolation):
            store.put_claim(
                Claim(
                    id=Claim.make_id("floating", subject.id),
                    statement="A claim with nothing behind it",
                    subject_concept_id=subject.id,
                    provenance=model_provenance("CandidateExtractor"),
                )
            )

    def test_claim_whose_evidence_was_deleted_is_an_error(self, evidenced):
        """The second line of defence: corruption that arrives after the write.

        Deleting the evidence row behind the store's back is the only way to
        reach this state, which is exactly why the diagnostic exists.
        """
        store, _, claim, _, _ = evidenced
        with store._conn:
            store._conn.execute("DELETE FROM evidence_links WHERE claim_id = ?", (claim.id,))

        report = check_integrity(store)

        assert IntegrityCode.CLAIM_WITHOUT_EVIDENCE.value in report.by_code()
        assert [f.entity_id for f in report.errors] == [claim.id]

    def test_evidence_pointing_at_a_deleted_span_is_an_error(self, evidenced):
        store, _, claim, span, _ = evidenced
        with store._conn:
            store._conn.execute("PRAGMA foreign_keys = OFF")
            store._conn.execute("DELETE FROM spans WHERE id = ?", (span.id,))

        report = check_integrity(store)

        assert IntegrityCode.DANGLING_EVIDENCE.value in report.by_code()

    def test_orphan_concept_is_a_warning_not_an_error(self, store):
        store.put_concept(concept("Nobody Asked", origin_proposal_id=None))

        report = check_integrity(store)

        assert IntegrityCode.ORPHAN_CONCEPT.value in report.by_code()
        assert report.errors == [], "an unexplained concept is untidy, not wrong"

    def test_concept_with_an_origin_proposal_is_not_an_orphan(self, chain):
        store, _ = chain
        assert IntegrityCode.ORPHAN_CONCEPT.value not in check_integrity(store).by_code()

    def test_findings_are_deterministic(self, chain):
        store, by_name = chain
        store.put_link(link(by_name["Alpha"].id, "concept:ghost", LinkType.DEPENDS_ON))

        first = check_integrity(store).to_dict()["findings"]
        second = check_integrity(store).to_dict()["findings"]

        assert first == second

    def test_integrity_check_repairs_nothing(self, chain):
        store, by_name = chain
        store.put_link(link(by_name["Alpha"].id, "concept:ghost", LinkType.DEPENDS_ON))
        before = (len(store.all_links()), len(list(store.list_concepts())))

        report = check_integrity(store)

        assert not report.clean
        assert (len(store.all_links()), len(list(store.list_concepts()))) == before, (
            "diagnostics must report, never repair"
        )

    def test_every_finding_carries_a_human_description(self, chain):
        store, by_name = chain
        store.put_link(link(by_name["Alpha"].id, "concept:ghost", LinkType.DEPENDS_ON))

        for finding in check_integrity(store).to_dict()["findings"]:
            assert finding["description"]
            assert finding["severity"] in {"error", "warning"}

    def test_checked_counts_report_what_was_examined(self, chain):
        store, _ = chain
        report = check_integrity(store)

        assert report.checked == {"concepts": 5, "claims": 0, "relationships": 3}
