"""Bridge Phase 1's frontmatter repair proposals into the proposal system.

Phase 1 found 283 files with mechanically repairable frontmatter and
deliberately applied none of them. Phase 2 exposes them for review.

Safety classification is derived, never asserted:

* A repair that was applied in memory and **re-parsed successfully** is
  ``DETERMINISTIC_VERIFIED`` — software computed it and software checked it.
* A repair that was computed but could not be verified is
  ``DETERMINISTIC_UNVERIFIED``.
* An LLM-generated repair would be ``MODEL_GENERATED``, and the domain layer
  refuses to let it claim otherwise. Phase 2 generates no such repairs.

Only ``DETERMINISTIC_VERIFIED`` proposals are ever eligible for automated
application, and even then only once approved and only behind ``--apply``.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from ..corpus.model import IndexedFile
from ..domain import (
    EntityType,
    Proposal,
    ProposalType,
    ProvenanceTier,
    SafetyClass,
    deterministic_provenance,
)
from ..ids import text_hash
from ..parsing.frontmatter import RepairProposal

REPAIR_AGENT = "frontmatter_repair"


def build_repair_proposals(files: Iterable[IndexedFile]) -> Iterator[Proposal]:
    """Yield one proposal per repairable frontmatter line.

    One proposal per *line*, not per file: a file with two malformed fields
    presents two independently reviewable decisions, and bundling them would
    force an all-or-nothing choice on unrelated changes.
    """
    for indexed in files:
        for repair in indexed.repairs:
            yield _proposal(indexed, repair)


def _proposal(indexed: IndexedFile, repair: RepairProposal) -> Proposal:
    safety = (
        SafetyClass.DETERMINISTIC_VERIFIED
        if repair.verified
        else SafetyClass.DETERMINISTIC_UNVERIFIED
    )

    # Affected links: the wikilinks this frontmatter field carries. Shown so a
    # reviewer can see what relationships the repair would make machine-readable.
    affected = list(indexed.related) if repair.key == "related" else []

    diagnostics = [
        d.to_dict() for d in indexed.diagnostics if d.key in (None, repair.key)
    ]

    return Proposal(
        id=Proposal.make_id(
            ProposalType.METADATA_REPAIR,
            indexed.path,
            text_hash(f"{repair.line}:{repair.original}:{repair.proposed}"),
        ),
        type=ProposalType.METADATA_REPAIR,
        safety=safety,
        target_entity_type=EntityType.SOURCE,
        target_entity_id=None,
        operation=_operation(indexed, repair, affected, diagnostics),
        reason=repair.reason,
        # No evidence spans: this is a deterministic repair of a file's own
        # metadata, not a knowledge claim drawn from source text.
        evidence_span_ids=(),
        provenance=deterministic_provenance(
            REPAIR_AGENT, ProvenanceTier.USER_ASSERTION
        ),
    )


def _operation(
    indexed: IndexedFile,
    repair: RepairProposal,
    affected: list[str],
    diagnostics: list[dict[str, object]],
):
    from ..domain import ProposedOperation

    return ProposedOperation(
        action="replace_frontmatter_line",
        target=indexed.path,
        before=repair.original,
        after=repair.proposed,
        details={
            "line": repair.line,
            "key": repair.key,
            "verified": repair.verified,
            "affected_links": affected,
            "diagnostics": diagnostics,
            "file_has_valid_frontmatter": indexed.frontmatter_valid,
        },
    )


def summarize(proposals: list[Proposal]) -> dict[str, int]:
    """Counts by safety class, for reporting."""
    counts: dict[str, int] = {}
    for proposal in proposals:
        counts[proposal.safety.value] = counts.get(proposal.safety.value, 0) + 1
    return dict(sorted(counts.items()))
