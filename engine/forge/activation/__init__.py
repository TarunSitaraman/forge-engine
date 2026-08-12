"""Proposal activation: approved decisions become canonical knowledge."""

from .activator import (
    ACTIVATOR_VERSION,
    ActivationOutcome,
    ActivationReport,
    ActivationResult,
    ProposalActivator,
)
from .relationships import (
    MIN_COOCCURRENCE,
    RELATIONSHIP_VERSION,
    SUPPORTED_TYPES,
    RelationshipActivator,
    RelationshipCandidate,
    RelationshipReport,
)

__all__ = [
    "ProposalActivator",
    "ActivationResult",
    "ActivationReport",
    "ActivationOutcome",
    "RelationshipActivator",
    "RelationshipCandidate",
    "RelationshipReport",
    "SUPPORTED_TYPES",
    "MIN_COOCCURRENCE",
    "ACTIVATOR_VERSION",
    "RELATIONSHIP_VERSION",
]
