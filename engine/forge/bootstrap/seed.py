"""Deterministic graph seeding from vault structure.

**Zero model calls.** The vault already states what its concepts are: a human
decided `Binary Search` deserves one canonical home and created the file. That
decision is the exact judgement LLM extraction was struggling to reproduce, and
it is sitting in the directory listing — measured 2026-08-20, extraction over
prose returned `RAM`, `Answer`, `Fluency` and `VARCHAR(n)` as concepts.

Two derivations, both structural:

* **Filenames -> Concepts.** Provenance is ``USER_ASSERTION`` /
  ``DETERMINISTIC``: the user asserted this concept by creating a page for it,
  and reading the filename involves no inference. ``USER_ASSERTION`` is the one
  tier permitted to stand without supporting evidence, which is correct here —
  the evidence *is* the file.
* **Links -> edges.** Every edge is ``RELATED_TO``. That is not a style
  choice: the concept graph accepts ``{RELATED_TO, PART_OF, DEPENDS_ON,
  IMPLEMENTS, EXPLAINS}`` and deterministic code may assert ``{MENTIONS,
  DERIVED_FROM, PRECEDES, RELATED_TO, ABOUT}``, and ``RELATED_TO`` is the
  single member of both. Promoting one to ``DEPENDS_ON`` or ``PART_OF`` needs
  judgement and is deliberately out of scope.

  ``RELATED_TO`` requires an explicit score, so that the edge which would
  otherwise turn the graph into an untyped mesh has to justify itself. These
  score ``1.0`` — not a computed similarity, but a human-authored link, which
  is the strongest evidence of relatedness the vault contains. The rationale
  on every edge says so, so a later reader cannot mistake it for a
  measurement.

Nothing is written to the vault. Seeding populates the derived store only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..corpus.model import CorpusIndex
from ..domain import (
    ClaimLink,
    Concept,
    ConceptKind,
    Derivation,
    LinkType,
    Provenance,
    ProvenanceTier,
)
from ..parsing.links import LinkStatus, normalize
from .kinds import KIND_NAMESPACES, kind_for

BOOTSTRAP_VERSION = "bootstrap/0.1.0"

#: Stems that name navigation, not concepts. An index page is a table of
#: contents; making it a concept puts "_index" in the graph 20 times over.
EXCLUDED_STEMS: frozenset[str] = frozenset(
    {
        "_index",
        "index",
        "readme",
        "start_here",
        "claude",
        "roadmap",
        "conventions",
        "workflow",
        "license",
        "contributing",
    }
)

#: Numbered section files inside a knowledge pack — `01-overview.md`,
#: `10-roadmap.md`. These are chapters of a project's documentation, not
#: concepts, and their stems collide across every pack that uses the pattern.
_NUMBERED_SECTION_RE = re.compile(r"^\d{2}-[a-z0-9-]+$")

#: Status and session artifacts. Point-in-time records, not durable ideas.
_ARTIFACT_RE = re.compile(r"(SUMMARY|STATUS|PLAN|CHECKLIST|_SESSION_)", re.IGNORECASE)

#: Folders that hold navigation only. `DSA/00_Index` is a set of hub pages —
#: `Pattern Index`, `Data Structure Index`, `DSA Home` — that route the reader
#: to the real pages. Seeding them makes "Data Structure Index" a concept
#: sitting alongside the data structures it lists, which is the same
#: table-of-contents mistake extraction made.
EXCLUDED_FOLDERS: tuple[str, ...] = ("DSA/00_Index/", "Archive/", "Inbox/")


def is_concept_page(rel_path: str) -> bool:
    """Does this page name a concept, as opposed to navigation or a chapter?"""
    if rel_path.startswith(EXCLUDED_FOLDERS):
        return False
    stem = rel_path.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".md") else stem
    if stem.casefold() in EXCLUDED_STEMS:
        return False
    if _NUMBERED_SECTION_RE.match(stem.casefold()):
        return False
    if _ARTIFACT_RE.search(stem):
        return False
    return True


@dataclass
class SeedPlan:
    """What seeding would create. Inspectable before anything is stored."""

    concepts: list[Concept] = field(default_factory=list)
    links: list[ClaimLink] = field(default_factory=list)
    #: Bare names that map to more than one page and have no recorded decision.
    #: Left out of the graph entirely — the engine must not pick one.
    undecided_collisions: dict[str, list[str]] = field(default_factory=dict)
    skipped_pages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "concepts": len(self.concepts),
            "links": len(self.links),
            "undecided_collisions": self.undecided_collisions,
            "skipped_pages": len(self.skipped_pages),
            "by_kind": self.by_kind(),
        }

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.concepts:
            counts[c.kind.value] = counts.get(c.kind.value, 0) + 1
        return dict(sorted(counts.items()))


def _provenance() -> Provenance:
    return Provenance(
        tier=ProvenanceTier.USER_ASSERTION,
        derivation=Derivation.DETERMINISTIC,
        confidence=1.0,
        agent=BOOTSTRAP_VERSION,
    )


def build_plan(index: CorpusIndex, decided: dict[str, str] | None = None) -> SeedPlan:
    """Derive concepts and edges from an indexed corpus. Makes no model calls."""
    decided = decided or {}
    plan = SeedPlan()

    pages = [f for f in index.files if is_concept_page(f.path)]
    plan.skipped_pages = [f.path for f in index.files if not is_concept_page(f.path)]

    by_stem: dict[str, list[str]] = {}
    for f in pages:
        stem = f.path.rsplit("/", 1)[-1][:-3]
        by_stem.setdefault(stem, []).append(f.path)

    path_to_concept: dict[str, Concept] = {}
    for stem, paths in sorted(by_stem.items()):
        namespaced = len(paths) > 1
        if namespaced and normalize(stem) not in decided:
            # A collision nobody has ruled on. Creating both under invented
            # namespaces would fabricate a distinction the user never drew.
            plan.undecided_collisions[stem] = sorted(paths)
            continue
        for path in sorted(paths):
            kind = kind_for(path)
            namespace = KIND_NAMESPACES.get(kind, "concept") if namespaced else None
            concept = Concept(
                id=Concept.make_id(stem, namespace),
                canonical_name=stem,
                kind=kind,
                vault_path=path,
                namespace=namespace,
                provenance=_provenance(),
            )
            plan.concepts.append(concept)
            path_to_concept[path] = concept

    plan.links = list(_links(pages, path_to_concept))
    return plan


def _links(pages, path_to_concept: dict[str, Concept]) -> Iterable[ClaimLink]:
    """One MENTIONS edge per resolved link between two concept pages.

    Deduplicated: a page linking to the same target five times is one edge, not
    five. Self-links are dropped — the domain rejects them, and a page
    mentioning itself carries no information.
    """
    seen: set[tuple[str, str]] = set()
    for f in pages:
        source = path_to_concept.get(f.path)
        if source is None:
            continue
        for link in f.links:
            if link.status not in (LinkStatus.RESOLVED, LinkStatus.CASE_MISMATCH):
                continue
            target = path_to_concept.get(link.resolved_path or "")
            if target is None or target.id == source.id:
                continue
            key = (source.id, target.id)
            if key in seen:
                continue
            seen.add(key)
            where = "related: field" if link.in_frontmatter else "body"
            yield ClaimLink(
                id=ClaimLink.make_id(source.id, target.id, LinkType.RELATED_TO),
                from_id=source.id,
                to_id=target.id,
                type=LinkType.RELATED_TO,
                provenance=_provenance(),
                score=1.0,
                rationale=(
                    f"human-authored link in {f.path} ({where}) -> "
                    f"{target.vault_path}; not a computed similarity"
                ),
            )
