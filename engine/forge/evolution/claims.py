"""Retrieve the existing claims that candidate concepts carry.

Small module, one job, and one rule: **bounded**. Traversal reuses the Phase 3
graph, which cannot be asked for an unbounded walk, and the result is capped
before it reaches a prompt. The failure this prevents is not slowness — it is
sending forty claims to a model and getting forty shallow judgements back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..domain import Claim, ClaimStatus
from ..graph import KnowledgeGraph
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

RETRIEVAL_VERSION = "claim-retrieval/0.1.0"

#: Claims per concept, and overall. Both caps exist: one prevents a single
#: heavily-documented concept from crowding out every other candidate, the
#: other bounds total cost.
DEFAULT_PER_CONCEPT = 5
DEFAULT_TOTAL = 20

#: Claim statuses that new evidence may still bear on.
LIVE_STATUSES: frozenset[ClaimStatus] = frozenset(
    {ClaimStatus.ACTIVE, ClaimStatus.DISPUTED}
)


@dataclass
class ClaimRetrieval:
    claims: list[Claim] = field(default_factory=list)
    by_concept: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def ids(self) -> list[str]:
        return [c.id for c in self.claims]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": len(self.claims),
            "by_concept": dict(sorted(self.by_concept.items())),
            "truncated": self.truncated,
        }


class ClaimRetriever:
    """Finds live claims attached to the candidate concepts."""

    version = RETRIEVAL_VERSION

    def __init__(
        self,
        store: SqliteStore,
        *,
        per_concept: int = DEFAULT_PER_CONCEPT,
        total: int = DEFAULT_TOTAL,
    ) -> None:
        self.store = store
        self.graph = KnowledgeGraph(store)
        self.per_concept = per_concept
        self.total = total

    def retrieve(self, concept_ids: Sequence[str]) -> ClaimRetrieval:
        result = ClaimRetrieval()
        seen: set[str] = set()

        for concept_id in concept_ids:
            concept = self.store.get_concept(concept_id)
            if concept is None:
                continue
            # Superseded and retracted claims are excluded: new evidence
            # bearing on a statement Forge has already replaced is not a
            # knowledge change, and assessing them would spend model calls
            # re-litigating history.
            #
            # DISPUTED claims are deliberately **kept**. A claim flagged as
            # doubtful is still live knowledge, and later evidence may support
            # it, sharpen it, or add a second reason to doubt it. Excluding
            # them would make "disputed" a terminal state that no amount of new
            # evidence could ever revisit — and would also have let this
            # module fake idempotency, which belongs to the assessment cache.
            claims = [
                c
                for c in self.graph.get_concept_claims(concept_id)
                if c.status in LIVE_STATUSES and c.id not in seen
            ][: self.per_concept]

            for claim in claims:
                if len(result.claims) >= self.total:
                    result.truncated = True
                    break
                seen.add(claim.id)
                result.claims.append(claim)
                result.by_concept[concept.qualified_name] = (
                    result.by_concept.get(concept.qualified_name, 0) + 1
                )
            if result.truncated:
                break

        log.info(
            "claims_retrieved",
            concepts=len(concept_ids),
            claims=len(result.claims),
            truncated=result.truncated,
        )
        return result
