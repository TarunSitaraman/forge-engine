"""Proposal system — the engine's only route to changing what a human owns."""

from .apply import ApplyOutcome, ApplyReport, ProposalApplier
from .metadata_repair import build_repair_proposals, summarize
from .service import ProposalService, claim_proposal, concept_proposal

__all__ = [
    "ProposalService",
    "concept_proposal",
    "claim_proposal",
    "build_repair_proposals",
    "summarize",
    "ProposalApplier",
    "ApplyReport",
    "ApplyOutcome",
]
