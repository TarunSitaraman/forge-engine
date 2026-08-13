"""Knowledge evolution — evaluating how new evidence affects what Forge knows.

The services here are ordinary Python and are individually testable with no
orchestration framework installed. LangGraph lives in :mod:`.workflow` and is
imported lazily, so importing this package never requires it.

That split is the point: the workflow is orchestration, the services are the
behaviour. Replacing the orchestrator would not change a single rule about
grounding, provenance, or approval.
"""

from __future__ import annotations

from .activation import EvolutionActivator
from .assessor import AssessmentBatch, AssessmentOutcome, EvidenceAssessor
from .candidates import CandidateNarrower, NarrowingResult
from .claims import ClaimRetriever, ClaimRetrieval
from .impact import actionable, classify_impact, impact_of, requires_human_review
from .proposer import EvolutionProposer, ProposalBatch

__all__ = [
    "AssessmentBatch",
    "AssessmentOutcome",
    "CandidateNarrower",
    "ClaimRetrieval",
    "ClaimRetriever",
    "EvidenceAssessor",
    "EvolutionActivator",
    "EvolutionProposer",
    "NarrowingResult",
    "ProposalBatch",
    "actionable",
    "classify_impact",
    "impact_of",
    "requires_human_review",
]
