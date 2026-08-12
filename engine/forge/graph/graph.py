"""Knowledge graph over SQLite.

**No graph database.** The measured graph is small (see
``GraphMetrics``), and recursive traversal over an indexed adjacency table is
milliseconds at this size. The point of this module is not to avoid Neo4j
forever — it is to *measure* the graph so the decision to adopt one can be
made on evidence rather than anticipation.

Every traversal is **bounded**. There is no unbounded exploration anywhere: a
depth limit and a node budget are required parameters with conservative
defaults, because an unbounded traversal on a densely-connected graph is a way
to hang the CLI.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..domain import Claim, ClaimLink, Concept, EntityType, EvidenceLink, LinkType, Span
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

GRAPH_VERSION = "graph/0.3.0"

#: Conservative traversal defaults. Both are overridable per call; neither may
#: be disabled.
DEFAULT_MAX_DEPTH = 3
DEFAULT_NODE_BUDGET = 500

#: Types the Phase 3 graph is allowed to contain. Mirrors the activator's
#: vocabulary; integrity checks flag anything outside it.
SUPPORTED_GRAPH_TYPES: frozenset[LinkType] = frozenset(
    {
        LinkType.RELATED_TO,
        LinkType.PART_OF,
        LinkType.DEPENDS_ON,
        LinkType.IMPLEMENTS,
        LinkType.EXPLAINS,
    }
)


@dataclass(frozen=True)
class Neighbor:
    """One step out from a node."""

    entity_id: str
    link: ClaimLink
    direction: str  # "out" | "in"
    concept: Concept | None = None

    @property
    def label(self) -> str:
        return self.concept.qualified_name if self.concept else self.entity_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "type": self.link.type.value,
            "direction": self.direction,
            "score": self.link.score,
            "rationale": self.link.rationale,
        }


@dataclass(frozen=True)
class Path:
    """A route between two entities."""

    nodes: tuple[str, ...]
    links: tuple[ClaimLink, ...]

    @property
    def depth(self) -> int:
        return len(self.links)

    def describe(self, labels: dict[str, str] | None = None) -> str:
        labels = labels or {}
        parts = [labels.get(self.nodes[0], self.nodes[0])]
        for node, link in zip(self.nodes[1:], self.links):
            parts.append(f"-[{link.type.value}]->")
            parts.append(labels.get(node, node))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "types": [link.type.value for link in self.links],
            "depth": self.depth,
        }


@dataclass
class GraphMetrics:
    """Measurements that decide whether a graph database is ever justified."""

    nodes: int = 0
    edges: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    max_degree: int = 0
    mean_degree: float = 0.0
    branching_factor: float = 0.0
    isolated_nodes: int = 0
    max_observed_depth: int = 0
    #: Milliseconds, measured on this store.
    neighbor_query_ms: float = 0.0
    path_query_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "by_type": self.by_type,
            "max_degree": self.max_degree,
            "mean_degree": round(self.mean_degree, 3),
            "branching_factor": round(self.branching_factor, 3),
            "isolated_nodes": self.isolated_nodes,
            "max_observed_depth": self.max_observed_depth,
            "neighbor_query_ms": round(self.neighbor_query_ms, 3),
            "path_query_ms": round(self.path_query_ms, 3),
        }


class KnowledgeGraph:
    """Bounded graph operations over the existing SQLite store."""

    version = GRAPH_VERSION

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    # -- neighbourhood -----------------------------------------------------

    def get_relationships(
        self, entity_id: str, *, types: Sequence[LinkType] | None = None, active_only: bool = True
    ) -> list[ClaimLink]:
        """Every edge touching this entity, in either direction."""
        links = [*self.store.links_from(entity_id), *self.store.links_to(entity_id)]
        if active_only:
            links = [link for link in links if link.active]
        if types:
            wanted = set(types)
            links = [link for link in links if link.type in wanted]
        return sorted(links, key=lambda link: (link.type.value, link.to_id, link.from_id))

    def get_neighbors(
        self, entity_id: str, *, types: Sequence[LinkType] | None = None, limit: int = 50
    ) -> list[Neighbor]:
        """Immediate neighbours, with the concept resolved where possible."""
        out: list[Neighbor] = []
        for link in self.get_relationships(entity_id, types=types):
            other = link.to_id if link.from_id == entity_id else link.from_id
            direction = "out" if link.from_id == entity_id else "in"
            out.append(
                Neighbor(
                    entity_id=other,
                    link=link,
                    direction=direction,
                    concept=self.store.get_concept(other),
                )
            )
        return out[:limit]

    def get_related_concepts(
        self, concept_id: str, *, max_depth: int = 2, budget: int = DEFAULT_NODE_BUDGET
    ) -> list[tuple[Concept, int]]:
        """Concepts reachable within ``max_depth``, with their distance.

        Bounded by both depth and a node budget — a densely connected graph
        must not be able to turn this into a full scan.
        """
        max_depth = max(1, min(max_depth, DEFAULT_MAX_DEPTH))
        seen = {concept_id}
        frontier = deque([(concept_id, 0)])
        found: list[tuple[Concept, int]] = []

        while frontier and len(seen) < budget:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.get_neighbors(current):
                if neighbor.entity_id in seen:
                    continue
                seen.add(neighbor.entity_id)
                if neighbor.concept is not None:
                    found.append((neighbor.concept, depth + 1))
                frontier.append((neighbor.entity_id, depth + 1))

        return sorted(found, key=lambda pair: (pair[1], pair[0].canonical_name))

    def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        budget: int = DEFAULT_NODE_BUDGET,
    ) -> Path | None:
        """Shortest path between two entities, or ``None`` within the bound.

        Breadth-first, so the first path found is shortest. ``None`` means "no
        path within ``max_depth``" — deliberately not "no path", which this
        bounded search cannot establish.
        """
        if source_id == target_id:
            return Path(nodes=(source_id,), links=())

        max_depth = max(1, min(max_depth, 6))
        visited = {source_id}
        frontier: deque[tuple[str, tuple[str, ...], tuple[ClaimLink, ...]]] = deque(
            [(source_id, (source_id,), ())]
        )

        while frontier and len(visited) < budget:
            current, nodes, links = frontier.popleft()
            if len(links) >= max_depth:
                continue
            for neighbor in self.get_neighbors(current):
                if neighbor.entity_id in visited:
                    continue
                path = Path(
                    nodes=(*nodes, neighbor.entity_id), links=(*links, neighbor.link)
                )
                if neighbor.entity_id == target_id:
                    return path
                visited.add(neighbor.entity_id)
                frontier.append((neighbor.entity_id, path.nodes, path.links))
        return None

    # -- evidence ----------------------------------------------------------

    def get_claim_evidence(self, claim_id: str) -> list[dict[str, Any]]:
        """The full chain from a claim back to source material.

        Claim -> EvidenceLink -> Span -> Document -> Source, resolved in one
        call, because that chain is the product.
        """
        claim = self.store.get_claim(claim_id)
        if claim is None:
            return []

        chain: list[dict[str, Any]] = []
        for link in self.store.evidence_for_claim(claim_id):
            span = self.store.get_span(link.span_id)
            document = self.store.get_document(span.document_id) if span else None
            source = self.store.get_source(document.source_id) if document else None
            chain.append(
                {
                    "relation": link.relation.value,
                    "span_id": link.span_id,
                    "citation": (
                        f"{source.locator} :: {span.citation()}" if source and span else None
                    ),
                    "page": span.page if span else None,
                    "heading_path": list(span.heading_path) if span else [],
                    "text": span.text if span else None,
                    "document_id": document.id if document else None,
                    "source_id": source.id if source else None,
                    "source_kind": source.kind.value if source else None,
                    "trust_tier": source.trust_tier.value if source else None,
                }
            )
        return chain

    def get_concept_claims(self, concept_id: str) -> list[Claim]:
        return [c for c in self.store.list_claims() if c.subject_concept_id == concept_id]

    def explain_concept(self, concept_id: str) -> dict[str, Any] | None:
        """Everything that justifies a concept's existence.

        Answers "which proposal created this?" and "which spans evidenced it?"
        in one place, so no canonical entity is an unexplained orphan.
        """
        concept = self.store.get_concept(concept_id)
        if concept is None:
            return None

        proposal = (
            self.store.get_proposal(concept.origin_proposal_id)
            if concept.origin_proposal_id
            else None
        )
        spans = [s for s in (self.store.get_span(i) for i in concept.origin_span_ids) if s]

        return {
            "concept": {
                "id": concept.id,
                "canonical_name": concept.canonical_name,
                "qualified_name": concept.qualified_name,
                "namespace": concept.namespace,
                "kind": concept.kind.value,
                "aliases": list(concept.aliases),
                "vault_path": concept.vault_path,
            },
            "origin_proposal": (
                {
                    "id": proposal.id,
                    "type": proposal.type.value,
                    "status": proposal.status.value,
                    "reason": proposal.reason,
                    "decided_by": proposal.decided_by,
                    "safety": proposal.safety.value,
                }
                if proposal
                else None
            ),
            "provenance": {
                "tier": concept.provenance.tier.value,
                "derivation": concept.provenance.derivation.value,
                "model_id": concept.provenance.model_id,
                "agent": concept.provenance.agent,
            },
            "origin_spans": [
                {"span_id": s.id, "citation": s.citation(), "text": s.text[:200]} for s in spans
            ],
            "claims": [
                {"id": c.id, "statement": c.statement, "tier": c.provenance.tier.value}
                for c in self.get_concept_claims(concept_id)
            ],
            "relationships": [n.to_dict() for n in self.get_neighbors(concept_id)],
        }

    # -- measurement -------------------------------------------------------

    def metrics(self, *, sample: int = 25) -> GraphMetrics:
        """Measure the graph. These numbers decide the Neo4j question."""
        concepts = list(self.store.list_concepts())
        links = self.store.all_links()

        degree: dict[str, int] = {c.id: 0 for c in concepts}
        by_type: dict[str, int] = {}
        for link in links:
            by_type[link.type.value] = by_type.get(link.type.value, 0) + 1
            for endpoint in (link.from_id, link.to_id):
                degree[endpoint] = degree.get(endpoint, 0) + 1

        degrees = list(degree.values())
        metrics = GraphMetrics(
            nodes=len(concepts),
            edges=len(links),
            by_type=dict(sorted(by_type.items())),
            max_degree=max(degrees, default=0),
            mean_degree=(sum(degrees) / len(degrees)) if degrees else 0.0,
            isolated_nodes=sum(1 for d in degrees if d == 0),
        )
        # Branching factor over connected nodes only: including isolated nodes
        # would understate what traversal actually costs.
        connected = [d for d in degrees if d > 0]
        metrics.branching_factor = (sum(connected) / len(connected)) if connected else 0.0

        if concepts:
            sampled = concepts[:sample]
            started = time.perf_counter()
            for concept in sampled:
                self.get_neighbors(concept.id)
            metrics.neighbor_query_ms = (time.perf_counter() - started) * 1000 / len(sampled)

            if len(concepts) >= 2:
                started = time.perf_counter()
                path = self.find_path(concepts[0].id, concepts[-1].id)
                metrics.path_query_ms = (time.perf_counter() - started) * 1000
                metrics.max_observed_depth = path.depth if path else 0

        return metrics

    def node_labels(self, entity_ids: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for entity_id in entity_ids:
            concept = self.store.get_concept(entity_id)
            if concept is not None:
                out[entity_id] = concept.qualified_name
                continue
            claim = self.store.get_claim(entity_id)
            if claim is not None:
                out[entity_id] = f"claim: {claim.statement[:40]}"
        return out
