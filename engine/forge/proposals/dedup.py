"""Cross-span deduplication of extracted proposals. Zero model calls.

**Why this cannot be a prompt rule.** The extractor sends one span per call, so
the model physically cannot know that another span already produced
"Evaluating retrieval quality involves calculating recall at k" when it emits
"Evaluating a RAG system requires a labelled evaluation set with recall@k
metrics". Asking the prompt to deduplicate was a category error — measured
2026-08-19, three such clusters appeared in a 20-claim sample, and the same
sample carried three concept alias pairs in 14: `RAG` / `Retrieval Augmented
Generation`, `Reranking` / `Reranker`, `Hybrid search` / `hybrid (keyword +
vector) search`.

Deduplication is deterministic, so it applies retroactively to proposals that
already exist — the same property that let the grounding audit re-check a
corpus without re-running extraction.

**Nothing is merged automatically.** This reports clusters and a suggested
survivor; deciding that two statements say the same thing is a judgement, and
the engine's rule is that judgement routes to a human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

from ..domain import Proposal, ProposalType
from ..parsing.links import normalize

DEDUP_VERSION = "dedup/0.1.0"

#: Similarity at which two claim statements are treated as near-duplicates.
#:
#: Measured against the duplicate clusters actually observed in the 2026-08-19
#: sample, not chosen by feel — a first guess of 0.60 was wrong by a factor of
#: three, and the real clusters would all have been missed.
#:
#:     known duplicates   0.262  0.302  0.338  0.492
#:     distinct claims    0.076  0.132  0.142  0.170
#:
#: 0.22 sits in the gap. **That gap is 0.09 wide on eight hand-picked pairs**,
#: which is narrow and a small sample, so this will produce false positives.
#: That is acceptable only because the output is a cluster for a human to
#: review, never an automatic merge — a threshold this soft must not be wired
#: to anything that decides.
CLAIM_SIMILARITY = 0.22

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Suffixes that mark the same concept under a different part of speech.
#: `Reranking` / `Reranker` is one concept; the vault's one-canonical-home rule
#: says so, and no model call is needed to see it.
_STEM_SUFFIXES = ("ing", "er", "ers", "s", "es", "ed")


@dataclass
class Cluster:
    """A group of proposals that appear to say the same thing."""

    kind: str
    key: str
    proposal_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    #: The label to keep, if a human agrees. Longest wins: the fuller phrasing
    #: is more likely to be the canonical name a reader would search for.
    suggested: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "size": len(self.proposal_ids),
            "labels": self.labels,
            "suggested": self.suggested,
            "proposal_ids": self.proposal_ids,
        }


def _stem(name: str) -> str:
    """Normalized name with a common inflection removed."""
    base = normalize(name)
    for suffix in _STEM_SUFFIXES:
        if len(base) > len(suffix) + 3 and base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}


def similarity(a: str, b: str) -> float:
    """Blend token overlap with sequence similarity.

    Token overlap alone treats a reordered restatement as identical; sequence
    similarity alone punishes the same fact said in a different order. Two
    phrasings of one claim usually score well on both.
    """
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return (jaccard + ratio) / 2


def cluster_concepts(proposals: Iterable[Proposal]) -> list[Cluster]:
    """Group concept proposals whose names are the same concept."""
    by_stem: dict[str, Cluster] = {}
    for proposal in proposals:
        if proposal.type is not ProposalType.NEW_CONCEPT:
            continue
        name = proposal.operation.target
        cluster = by_stem.setdefault(_stem(name), Cluster(kind="concept", key=_stem(name)))
        cluster.proposal_ids.append(proposal.id)
        if name not in cluster.labels:
            cluster.labels.append(name)

    out = [c for c in by_stem.values() if len(c.labels) > 1 or len(c.proposal_ids) > 1]
    for cluster in out:
        cluster.suggested = max(cluster.labels, key=len)
    return sorted(out, key=lambda c: (-len(c.proposal_ids), c.key))


def cluster_claims(
    proposals: Iterable[Proposal], *, threshold: float = CLAIM_SIMILARITY
) -> list[Cluster]:
    """Group claim proposals whose statements are near-duplicates.

    Quadratic in the number of claims, which is fine at this corpus size: 1,170
    claims is ~680k comparisons of short strings, well under a second, and the
    alternative — an index — would add a dependency to save time nobody is
    waiting on.
    """
    claims = [p for p in proposals if p.type is ProposalType.NEW_CLAIM]
    statements = [p.operation.after or p.operation.target for p in claims]

    assigned: dict[int, Cluster] = {}
    clusters: list[Cluster] = []
    for i, statement in enumerate(statements):
        if i in assigned:
            continue
        cluster = Cluster(kind="claim", key=claims[i].id[:12])
        cluster.proposal_ids.append(claims[i].id)
        cluster.labels.append(statement)
        assigned[i] = cluster
        for j in range(i + 1, len(statements)):
            if j in assigned:
                continue
            if similarity(statement, statements[j]) >= threshold:
                cluster.proposal_ids.append(claims[j].id)
                cluster.labels.append(statements[j])
                assigned[j] = cluster
        if len(cluster.proposal_ids) > 1:
            cluster.suggested = max(cluster.labels, key=len)
            clusters.append(cluster)

    return sorted(clusters, key=lambda c: -len(c.proposal_ids))


def find_duplicates(proposals: Iterable[Proposal], *, threshold: float = CLAIM_SIMILARITY):
    """All duplicate clusters, concepts first."""
    items = list(proposals)
    return cluster_concepts(items) + cluster_claims(items, threshold=threshold)
