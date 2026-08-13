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
    #: Phase 2. A proposal is a *recorded intention* to change something, and
    #: has its own revision history, so it is an entity rather than a note.
    PROPOSAL = "Proposal"
    #: Phase 4. A workflow run is the durable record of *why* knowledge
    #: changed — which evidence, which candidates, which model, which
    #: decision. Without it, an approved change is unexplainable after the
    #: fact, which defeats the point of provenance.
    WORKFLOW = "Workflow"


# --------------------------------------------------------------------------
# Phase 2 — ingestion, extraction, and proposals
# --------------------------------------------------------------------------


class IngestionStatus(str, Enum):
    """Outcome of ingesting one source.

    ``OCR_REQUIRED`` is a first-class outcome, not a failure: the PDF is valid
    and was read successfully, it simply carries no extractable text. Reporting
    it as success with zero spans would be a lie; reporting it as a parse error
    would send the user looking for the wrong problem.
    """

    INGESTED = "ingested"
    UNCHANGED = "unchanged"
    OCR_REQUIRED = "ocr_required"
    UNSUPPORTED = "unsupported"
    PARSE_FAILED = "parse_failed"
    NOT_FOUND = "not_found"


class ExtractionStatus(str, Enum):
    """Outcome of an LLM extraction pass.

    ``PARTIAL`` exists so a run that produced some valid objects and some
    invalid ones is not rounded up to success or down to failure. Fabricating
    the missing pieces is never an option.
    """

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED_NO_PROVIDER = "skipped_no_provider"
    SKIPPED_CACHED = "skipped_cached"


class IdentityState(str, Enum):
    """How an extracted name relates to canonical concept identity.

    Finer-grained than :class:`MatchKind`, which describes the matcher's
    *action*; this describes the *evidence* for identity, so a reviewer can
    see whether a match rests on an exact name, a registered alias, or a
    user's explicit decision.
    """

    EXACT_MATCH = "exact_match"
    ALIAS_MATCH = "alias_match"
    RESOLVED_BY_USER = "resolved_by_user"
    NEW = "new"
    AMBIGUOUS = "ambiguous"


class MatchKind(str, Enum):
    """Result of matching an extracted concept against existing concepts.

    There is deliberately no ``MERGED``. Deciding that two concepts are the
    same is a human judgement in Phase 2, so the matcher's strongest possible
    output is a candidate.
    """

    NEW_CONCEPT = "new_concept"
    MATCH_CANDIDATE = "match_candidate"
    AMBIGUOUS = "ambiguous"


class ProposalType(str, Enum):
    METADATA_REPAIR = "metadata_repair"
    NEW_CONCEPT = "new_concept"
    CONCEPT_MATCH = "concept_match"
    NEW_CLAIM = "new_claim"
    # Phase 4 — knowledge *evolution*. These target knowledge that already
    # exists, which is what distinguishes them from everything above: the
    # types before this line only ever add.
    CLAIM_EVIDENCE = "claim_evidence"  # corroborating evidence for a live claim
    CLAIM_REFINEMENT = "claim_refinement"  # a more precise statement, superseding
    CLAIM_CONFLICT = "claim_conflict"  # evidence that appears to disagree


class AssessmentClass(str, Enum):
    """How new evidence relates to one existing claim.

    Deliberately **no ``CONTRADICTS``**. The strongest thing the model may say
    is ``POTENTIAL_CONFLICT``, which routes to a human. A false contradiction
    costs more trust than a missed one: the first makes the user distrust
    everything Forge asserts, the second only leaves them where they were.

    ``INSUFFICIENT_EVIDENCE`` is a first-class outcome, not a failure. A model
    forced to choose between five substantive options will pick one; giving it
    an honest way to decline is what keeps the other four meaningful.
    """

    SUPPORTS = "SUPPORTS"
    REFINES = "REFINES"
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"
    IRRELEVANT = "IRRELEVANT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ImpactClass(str, Enum):
    """What an assessment means for the knowledge base as a whole.

    Distinct from :class:`AssessmentClass` because they answer different
    questions. An assessment is about one claim; an impact is about what Forge
    should now *do*. The mapping between them is deterministic code, not a
    second model call — see :mod:`forge.evolution.impact`.
    """

    NO_MATERIAL_CHANGE = "NO_MATERIAL_CHANGE"
    SUPPORTS = "SUPPORTS"
    REFINES = "REFINES"
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"
    NEW_KNOWLEDGE = "NEW_KNOWLEDGE"


class WorkflowStatus(str, Enum):
    """Lifecycle of one knowledge-evolution run.

    ``WAITING_FOR_REVIEW`` is the state that makes the workflow agentic in the
    way that matters: the run stops itself, persists, and waits for a human,
    rather than proceeding on its own judgement.
    """

    RUNNING = "running"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    FAILED = "failed"
    #: The configured provider could not serve a semantic step. Resumable —
    #: never silently downgraded to a weaker model.
    SEMANTIC_ANALYSIS_UNAVAILABLE = "semantic_analysis_unavailable"


class ProposalStatus(str, Enum):
    """Proposal lifecycle.

    ``ACTIVATED`` is distinct from ``APPROVED`` on purpose. Approval is a
    human decision; activation is the persistence of canonical knowledge that
    followed from it. Collapsing them would make "approved" ambiguous about
    whether anything actually exists in the model — and would let a failed
    write masquerade as success.
    """

    PENDING = "pending"
    APPROVED = "approved"
    ACTIVATED = "activated"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class SafetyClass(str, Enum):
    """How much trust a proposal's content warrants.

    ``DETERMINISTIC_VERIFIED`` is reserved for changes computed by ordinary
    software *and* verified by re-parsing the result — the Phase 1 frontmatter
    repairs. An LLM-generated proposal can never carry it, no matter how
    confident the model was.
    """

    DETERMINISTIC_VERIFIED = "deterministic_verified"
    DETERMINISTIC_UNVERIFIED = "deterministic_unverified"
    MODEL_GENERATED = "model_generated"
    AMBIGUOUS = "ambiguous"


class ChangeStatus(str, Enum):
    """Result of hash-based change detection for one source."""

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
