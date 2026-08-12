"""Controlled vocabularies for the canonical knowledge model.

These are closed enums on purpose. An open string field would let the graph
accumulate ad-hoc types, which is exactly how a knowledge graph becomes an
untyped mesh.
"""

from __future__ import annotations

from enum import Enum


class ProvenanceTier(str, Enum):
    """How strongly warranted an assertable object is.

    Ordering is defined by :data:`TIER_STRENGTH`, not by declaration order.
    ``USER_ASSERTION`` sits deliberately outside the evidential ordering: it is
    not evidence, but it is authoritative for this user's model, and it is the
    only tier permitted to exist without supporting evidence.
    """

    SOURCE_FACT = "SOURCE_FACT"
    EXTRACTED_CLAIM = "EXTRACTED_CLAIM"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    SYNTHESIS = "SYNTHESIS"
    USER_ASSERTION = "USER_ASSERTION"


#: Evidential strength. Higher is stronger. Used by the provenance floor rule.
#:
#: ``USER_ASSERTION`` is given the same rank as ``EXTRACTED_CLAIM``. Rationale:
#: a user assertion is a first-hand statement of belief — stronger than a model
#: inference drawn from it, weaker than a verbatim source fact. Ranking it
#: below inference would let a model's guess outrank the user; ranking it at the
#: top would let unsourced belief launder itself into quotable evidence.
TIER_STRENGTH: dict[ProvenanceTier, int] = {
    ProvenanceTier.SOURCE_FACT: 4,
    ProvenanceTier.EXTRACTED_CLAIM: 3,
    ProvenanceTier.USER_ASSERTION: 3,
    ProvenanceTier.MODEL_INFERENCE: 2,
    ProvenanceTier.SYNTHESIS: 1,
}

#: Tiers that may exist with no supporting evidence link.
TIERS_WITHOUT_EVIDENCE: frozenset[ProvenanceTier] = frozenset(
    {ProvenanceTier.USER_ASSERTION}
)


class Derivation(str, Enum):
    """Whether an object was produced by ordinary software or by a model.

    Recorded on every assertable object so that Principle 7 ("deterministic
    software does deterministic work") is *measurable* rather than aspirational.
    """

    DETERMINISTIC = "deterministic"
    MODEL = "model"
    HUMAN = "human"


class SourceKind(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    REPO = "repo"
    WEB = "web"
    CODE = "code"
    DATASET = "dataset"
    MANUAL = "manual"


class TrustTier(str, Enum):
    """Trust in the *source*, independent of provenance tier.

    A faithful extraction from a weak source is still a faithful extraction.
    """

    PEER_REVIEWED = "peer_reviewed"
    OFFICIAL_DOCS = "official_docs"
    REPUTABLE_SECONDARY = "reputable_secondary"
    COMMUNITY = "community"
    UNVERIFIED = "unverified"
    USER_AUTHORED = "user_authored"


class ConceptKind(str, Enum):
    """Discriminator that keeps Technology/Project/Person from needing tables."""

    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PATTERN = "pattern"
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    PROJECT = "project"
    PERSON = "person"
    EXPERIMENT = "experiment"
    DECISION = "decision"
    TOPIC = "topic"
    PLAYBOOK = "playbook"
    TEMPLATE = "template"
    PROMPT = "prompt"
    PROBLEM = "problem"
    MISTAKE = "mistake"
    INTERVIEW_GUIDE = "interview_guide"
    CHEAT_SHEET = "cheat_sheet"
    COURSE = "course"


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    DISPUTED = "disputed"


class EvidenceRelation(str, Enum):
    """How a claim relates to the span that evidences it.

    Finer-grained than the claim's own tier: this is what keeps "the source
    says this" permanently distinguishable from "a model concluded this".
    """

    QUOTES = "quotes"
    PARAPHRASES = "paraphrases"
    INFERS_FROM = "infers_from"


class LinkType(str, Enum):
    """Typed relationships. Domain/range constraints live in :mod:`.relations`."""

    # Concept <-> Concept (structural)
    PART_OF = "PART_OF"
    DEPENDS_ON = "DEPENDS_ON"
    REQUIRES = "REQUIRES"
    IMPLEMENTS = "IMPLEMENTS"
    PRECEDES = "PRECEDES"
    EXPLAINS = "EXPLAINS"
    RELATED_TO = "RELATED_TO"
    # Claim <-> Claim (epistemic)
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    SUPERSEDES = "SUPERSEDES"
    DERIVED_FROM = "DERIVED_FROM"
    # Cross-type
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"


#: Link types a deterministic process is allowed to create. Anything else
#: requires ``Derivation.MODEL`` (or an explicit human assertion).
DETERMINISTIC_LINK_TYPES: frozenset[LinkType] = frozenset(
    {
        LinkType.MENTIONS,
        LinkType.DERIVED_FROM,
        LinkType.PRECEDES,
        LinkType.RELATED_TO,
        LinkType.ABOUT,
    }
)


class RevisionOp(str, Enum):
    CREATE = "create"
    CHANGE = "change"
    SUPERSEDE = "supersede"
    INVALIDATE = "invalidate"
    MERGE = "merge"
    SPLIT = "split"


class EntityType(str, Enum):
    SOURCE = "Source"
    DOCUMENT = "Document"
    SPAN = "Span"
    CONCEPT = "Concept"
    CLAIM = "Claim"
    EVIDENCE_LINK = "EvidenceLink"
    CLAIM_LINK = "ClaimLink"


class ChangeStatus(str, Enum):
    """Result of hash-based change detection for one source."""

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
