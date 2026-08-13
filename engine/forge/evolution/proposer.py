"""Assessments -> proposals.

The hard boundary of Phase 4: **model reasoning never mutates canonical
knowledge.** An assessment is an opinion with citations. It becomes a change
only by passing through the existing proposal system — the same one Phases 2
and 3 use, deliberately not a parallel one, so there is exactly one review
queue and exactly one approval path.

Three proposal types, matching what the evidence actually justifies:

======================  ====================================================
``CLAIM_EVIDENCE``      Corroboration. Adds an EvidenceLink; the claim's text
                        does not change.
``CLAIM_REFINEMENT``    A sharper statement. On activation the old claim is
                        superseded, not deleted.
``CLAIM_CONFLICT``      Evidence that appears to disagree. On activation the
                        claim is marked DISPUTED and the evidence attached.
                        The claim is never retracted automatically.
======================  ====================================================

Safety classification follows Phase 2's rule — derived from provenance, never
asserted. All three are model-generated, so none can be
``DETERMINISTIC_VERIFIED``; conflicts are additionally marked ``AMBIGUOUS``,
which makes Phase 3's batch-approval guard refuse to bulk-approve them without
an explicit flag. That guard was built for a different reason and turns out to
be exactly right here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..domain import (
    AssessmentClass,
    AssessmentRecord,
    Claim,
    Derivation,
    EntityType,
    Proposal,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceInput,
    ProvenanceTier,
    SafetyClass,
)
from ..logging import get_logger
from ..proposals.service import ProposalService
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)

PROPOSER_VERSION = "proposer/0.1.0"

#: Assessment class -> (proposal type, operation verb, safety).
_MAPPING: dict[AssessmentClass, tuple[ProposalType, str, SafetyClass]] = {
    AssessmentClass.SUPPORTS: (
        ProposalType.CLAIM_EVIDENCE,
        "attach_evidence",
        SafetyClass.MODEL_GENERATED,
    ),
    AssessmentClass.REFINES: (
        ProposalType.CLAIM_REFINEMENT,
        "refine_claim",
        SafetyClass.MODEL_GENERATED,
    ),
    AssessmentClass.POTENTIAL_CONFLICT: (
        ProposalType.CLAIM_CONFLICT,
        "flag_conflict",
        SafetyClass.AMBIGUOUS,
    ),
}


@dataclass
class ProposalBatch:
    created: list[Proposal] = field(default_factory=list)
    existing: list[Proposal] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def all_ids(self) -> list[str]:
        return [p.id for p in (*self.created, *self.existing)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": len(self.created),
            "already_present": len(self.existing),
            "skipped": self.skipped,
            "proposal_ids": self.all_ids,
            "by_type": _counts(p.type.value for p in (*self.created, *self.existing)),
        }


class EvolutionProposer:
    """Turns assessments into reviewable proposals. Creates no knowledge."""

    version = PROPOSER_VERSION

    def __init__(self, store: SqliteStore, *, workflow_id: str, source_id: str) -> None:
        self.store = store
        self.service = ProposalService(store)
        self.workflow_id = workflow_id
        self.source_id = source_id

    def propose(
        self,
        assessments: Sequence[AssessmentRecord],
        *,
        refined_statements: dict[str, str] | None = None,
    ) -> ProposalBatch:
        """Build one proposal per actionable assessment.

        Idempotent: proposal identity is derived from the claim, the
        classification, and the evidence, so re-running the same workflow
        returns the existing proposal — including its decision. A rejected
        proposal is never resurrected as PENDING.
        """
        batch = ProposalBatch()
        refined = refined_statements or {}

        for assessment in assessments:
            mapping = _MAPPING.get(assessment.classification)
            if mapping is None:
                # IRRELEVANT and INSUFFICIENT_EVIDENCE are correct outcomes
                # that produce nothing. Recorded so the run can show it
                # considered them.
                batch.skipped.append(
                    {
                        "claim_id": assessment.claim_id,
                        "classification": assessment.classification.value,
                        "reason": "not actionable by design",
                    }
                )
                continue

            claim = self.store.get_claim(assessment.claim_id)
            if claim is None:
                batch.skipped.append(
                    {
                        "claim_id": assessment.claim_id,
                        "classification": assessment.classification.value,
                        "reason": "claim no longer exists",
                    }
                )
                continue

            proposal = self._build(assessment, claim, mapping, refined.get(assessment.claim_id, ""))
            if proposal is None:
                batch.skipped.append(
                    {
                        "claim_id": assessment.claim_id,
                        "classification": assessment.classification.value,
                        "reason": "refinement had no refined statement",
                    }
                )
                continue

            stored, created = self.service.create(proposal)
            (batch.created if created else batch.existing).append(stored)

        log.info(
            "evolution_proposals",
            workflow=self.workflow_id[:12],
            created=len(batch.created),
            existing=len(batch.existing),
            skipped=len(batch.skipped),
        )
        return batch

    # -- construction ------------------------------------------------------

    def _build(
        self,
        assessment: AssessmentRecord,
        claim: Claim,
        mapping: tuple[ProposalType, str, SafetyClass],
        refined_statement: str,
    ) -> Proposal | None:
        proposal_type, action, safety = mapping

        after: str | None = None
        if proposal_type is ProposalType.CLAIM_REFINEMENT:
            if not refined_statement.strip():
                return None
            after = refined_statement.strip()

        details: dict[str, Any] = {
            "classification": assessment.classification.value,
            "workflow_id": self.workflow_id,
            "provider_id": assessment.provider_id,
            "model_id": assessment.model_id,
            "prompt_version": assessment.prompt_version,
            "schema_version": assessment.schema_version,
            "derivation_key": assessment.derivation_key,
        }
        if assessment.classification is AssessmentClass.POTENTIAL_CONFLICT:
            # Named explicitly so a reviewer reading the proposal sees the
            # limit of what Forge is claiming.
            details["review_required"] = (
                "Forge does not assert a contradiction. It reports that the new "
                "evidence may disagree, and asks you to decide."
            )

        return Proposal(
            id=Proposal.make_id(
                proposal_type,
                claim.id,
                # Fingerprint over what makes this proposal *this* proposal:
                # the judgement and the evidence behind it. A different model
                # reaching the same conclusion from the same spans is the same
                # proposal; different evidence is a different one.
                f"{assessment.classification.value}|{'|'.join(sorted(assessment.evidence_span_ids))}",
            ),
            type=proposal_type,
            safety=safety,
            target_entity_type=EntityType.CLAIM,
            target_entity_id=claim.id,
            operation=ProposedOperation(
                action=action,
                target=claim.id,
                before=claim.statement,
                after=after,
                details=details,
            ),
            reason=assessment.rationale,
            evidence_span_ids=assessment.evidence_span_ids,
            source_id=self.source_id,
            provenance=self._provenance(assessment),
        )

    def _provenance(self, assessment: AssessmentRecord) -> Provenance:
        """Model provenance, with the evidence recorded as inputs.

        ``MODEL_INFERENCE`` and no higher: the model concluded this, which is
        never the same as the source asserting it. The floor rule in the domain
        layer would reject anything stronger, and this module never asks.
        """
        return Provenance(
            tier=ProvenanceTier.MODEL_INFERENCE,
            derivation=Derivation.MODEL,
            agent="EvidenceAssessor",
            model_id=f"{assessment.provider_id}|{assessment.model_id}",
            prompt_version=assessment.prompt_version,
            schema_version=assessment.schema_version,
            derivation_key=assessment.derivation_key,
            inputs=tuple(
                # Spans are SOURCE_FACT: the document really does contain that
                # text. The floor rule then permits MODEL_INFERENCE above them
                # and would reject anything stronger.
                ProvenanceInput(
                    entity_type=EntityType.SPAN,
                    entity_id=span_id,
                    tier=ProvenanceTier.SOURCE_FACT,
                )
                for span_id in assessment.evidence_span_ids
            ),
        )


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))
