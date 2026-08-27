"""Graph integrity diagnostics.

Detects structural problems and **reports them**. Nothing is repaired
automatically — the same discipline the frontmatter diagnostics follow, for the
same reason: an automatic repair to a knowledge graph is an unreviewed change
to what the user believes.

Every check is deterministic. Given the same store, the same findings appear in
the same order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..domain import Derivation, LinkType
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore
from .graph import SUPPORTED_GRAPH_TYPES

log = get_logger(__name__)


class IntegrityCode(str, Enum):
    ORPHAN_RELATIONSHIP = "GR001"
    MISSING_TARGET = "GR002"
    DUPLICATE_RELATIONSHIP = "GR003"
    INVALID_TYPE = "GR004"
    NO_PROVENANCE = "GR005"
    DANGLING_EVIDENCE = "GR006"
    CLAIM_WITHOUT_EVIDENCE = "GR007"
    ORPHAN_CONCEPT = "GR008"
    SELF_RELATIONSHIP = "GR009"


CODE_DESCRIPTIONS: dict[IntegrityCode, str] = {
    IntegrityCode.ORPHAN_RELATIONSHIP: "Relationship whose source entity does not exist.",
    IntegrityCode.MISSING_TARGET: "Relationship whose target entity does not exist.",
    IntegrityCode.DUPLICATE_RELATIONSHIP: (
        "Two relationships connect the same pair with the same type."
    ),
    IntegrityCode.INVALID_TYPE: "Relationship type is outside the supported vocabulary.",
    IntegrityCode.NO_PROVENANCE: "Relationship carries no provenance agent.",
    IntegrityCode.DANGLING_EVIDENCE: "EvidenceLink points at a span that no longer exists.",
    IntegrityCode.CLAIM_WITHOUT_EVIDENCE: (
        "Claim requires evidence by its provenance tier but has none."
    ),
    IntegrityCode.ORPHAN_CONCEPT: (
        "Concept has no origin proposal, no claims, and no canonical vault page "
        "— nothing explains why it exists."
    ),
    IntegrityCode.SELF_RELATIONSHIP: "Relationship connects an entity to itself.",
}

#: Findings that mean the graph is *wrong*, as opposed to merely untidy.
ERROR_CODES: frozenset[IntegrityCode] = frozenset(
    {
        IntegrityCode.ORPHAN_RELATIONSHIP,
        IntegrityCode.MISSING_TARGET,
        IntegrityCode.DANGLING_EVIDENCE,
        IntegrityCode.CLAIM_WITHOUT_EVIDENCE,
        IntegrityCode.SELF_RELATIONSHIP,
    }
)


@dataclass(frozen=True)
class Finding:
    code: IntegrityCode
    entity_id: str
    detail: str

    @property
    def severity(self) -> str:
        return "error" if self.code in ERROR_CODES else "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity,
            "entity_id": self.entity_id,
            "detail": self.detail,
            "description": CODE_DESCRIPTIONS[self.code],
        }


@dataclass
class IntegrityReport:
    findings: list[Finding] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.code.value] = counts.get(finding.code.value, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "checked": self.checked,
            "by_code": self.by_code(),
            "errors": len(self.errors),
            "code_descriptions": {c.value: d for c, d in CODE_DESCRIPTIONS.items()},
            "findings": [f.to_dict() for f in self.findings],
        }


def check_integrity(store: SqliteStore) -> IntegrityReport:
    """Run every integrity check. Reports only; repairs nothing."""
    report = IntegrityReport()

    concepts = {c.id: c for c in store.list_concepts()}
    claims = {c.id: c for c in store.list_claims()}
    links = store.all_links()
    known = set(concepts) | set(claims)

    report.checked = {
        "concepts": len(concepts),
        "claims": len(claims),
        "relationships": len(links),
    }

    seen_pairs: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for link in sorted(links, key=lambda x: x.id):
        if link.from_id == link.to_id:
            report.findings.append(
                Finding(IntegrityCode.SELF_RELATIONSHIP, link.id, f"{link.from_id} -> itself")
            )
        if link.from_id not in known:
            report.findings.append(
                Finding(
                    IntegrityCode.ORPHAN_RELATIONSHIP,
                    link.id,
                    f"source {link.from_id} does not exist",
                )
            )
        if link.to_id not in known:
            report.findings.append(
                Finding(
                    IntegrityCode.MISSING_TARGET, link.id, f"target {link.to_id} does not exist"
                )
            )
        if link.type not in SUPPORTED_GRAPH_TYPES:
            report.findings.append(
                Finding(
                    IntegrityCode.INVALID_TYPE,
                    link.id,
                    f"type {link.type.value} is outside the supported vocabulary",
                )
            )
        if not link.provenance.agent:
            report.findings.append(
                Finding(IntegrityCode.NO_PROVENANCE, link.id, "no provenance agent recorded")
            )
        seen_pairs[(link.from_id, link.to_id, link.type.value)].append(link.id)

    for (from_id, to_id, link_type), ids in sorted(seen_pairs.items()):
        if len(ids) > 1:
            report.findings.append(
                Finding(
                    IntegrityCode.DUPLICATE_RELATIONSHIP,
                    ids[0],
                    f"{len(ids)} relationships of type {link_type} between {from_id} and {to_id}",
                )
            )

    from ..domain import TIERS_WITHOUT_EVIDENCE

    for claim_id, claim in sorted(claims.items()):
        evidence = store.evidence_for_claim(claim_id)
        if claim.provenance.tier not in TIERS_WITHOUT_EVIDENCE and not evidence:
            report.findings.append(
                Finding(
                    IntegrityCode.CLAIM_WITHOUT_EVIDENCE,
                    claim_id,
                    f"tier {claim.provenance.tier.value} requires evidence but none is stored",
                )
            )
        for link in evidence:
            if store.get_span(link.span_id) is None:
                report.findings.append(
                    Finding(
                        IntegrityCode.DANGLING_EVIDENCE,
                        link.id,
                        f"span {link.span_id} no longer exists",
                    )
                )

    claimed_concepts = {c.subject_concept_id for c in claims.values() if c.subject_concept_id}
    for concept_id, concept in sorted(concepts.items()):
        # A canonical vault page explains a concept at least as well as a
        # proposal does. A proposal is a model's suggestion that a human
        # approved; a page is the human's own act of deciding this idea
        # deserves one canonical home. The rule being enforced is "no
        # unexplained orphan", not "no concept without a proposal", so a
        # deterministically-derived concept that names its page is explained.
        explained_by_vault = bool(concept.vault_path) and (
            concept.provenance.derivation is Derivation.DETERMINISTIC
        )
        if (
            not concept.origin_proposal_id
            and concept_id not in claimed_concepts
            and not explained_by_vault
        ):
            report.findings.append(
                Finding(
                    IntegrityCode.ORPHAN_CONCEPT,
                    concept_id,
                    f"{concept.qualified_name!r} has no origin proposal and no claims",
                )
            )

    log.info("graph_integrity_checked", **report.by_code())
    return report
