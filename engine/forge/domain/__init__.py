"""Forge canonical knowledge model (Phase 1).

Pure domain layer: no storage, no HTTP, no LLM. Import graph points inward
only, so the model can be reused unchanged when storage or orchestration
choices change.
"""

from .entities import Claim, ClaimLink, Concept, Document, EvidenceLink, Source, Span
from .enums import (
    DETERMINISTIC_LINK_TYPES,
    TIER_STRENGTH,
    TIERS_WITHOUT_EVIDENCE,
    ChangeStatus,
    ClaimStatus,
    ConceptKind,
    Derivation,
    EntityType,
    EvidenceRelation,
    ExtractionStatus,
    IngestionStatus,
    LinkType,
    MatchKind,
    ProposalStatus,
    ProposalType,
    ProvenanceTier,
    RevisionOp,
    SafetyClass,
    SourceKind,
    TrustTier,
)
from .proposal import (
    Proposal,
    ProposalTransitionError,
    ProposedOperation,
)
from .provenance import (
    Provenance,
    ProvenanceInput,
    ProvenanceViolation,
    deterministic_provenance,
    floor_tier,
    utc_now,
    violates_floor,
)
from .revision import (
    Revision,
    record_change,
    record_create,
    record_invalidate,
    record_supersede,
)
from .validation import ValidationError, validate_claim, validate_claim_link, validate_supersession

__all__ = [
    # entities
    "Source",
    "Document",
    "Span",
    "Concept",
    "Claim",
    "EvidenceLink",
    "ClaimLink",
    # proposals (Phase 2)
    "Proposal",
    "ProposedOperation",
    "ProposalTransitionError",
    # provenance
    "Provenance",
    "ProvenanceInput",
    "ProvenanceViolation",
    "deterministic_provenance",
    "floor_tier",
    "violates_floor",
    "utc_now",
    # revision
    "Revision",
    "record_create",
    "record_change",
    "record_supersede",
    "record_invalidate",
    # enums
    "ProvenanceTier",
    "Derivation",
    "SourceKind",
    "TrustTier",
    "ConceptKind",
    "ClaimStatus",
    "EvidenceRelation",
    "LinkType",
    "RevisionOp",
    "EntityType",
    "ChangeStatus",
    "IngestionStatus",
    "ExtractionStatus",
    "MatchKind",
    "ProposalType",
    "ProposalStatus",
    "SafetyClass",
    "TIER_STRENGTH",
    "TIERS_WITHOUT_EVIDENCE",
    "DETERMINISTIC_LINK_TYPES",
    # validation
    "ValidationError",
    "validate_claim",
    "validate_supersession",
    "validate_claim_link",
]
