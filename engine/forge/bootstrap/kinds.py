"""Folder -> ConceptKind mapping for filename-derived concepts.

The vault's directory layout already encodes what kind of thing a page is —
`DSA/01_Patterns/Heap.md` is a pattern, `Technologies/Docs/redis.md` is a
technology. Reading that is deterministic and needs no model; guessing kinds
from prose is exactly the job extraction did badly.

Unmapped folders fall back to ``CONCEPT`` rather than raising. A new folder
should not break a bootstrap run, and ``CONCEPT`` is the honest default: the
page is a concept whose finer kind nobody has stated.
"""

from __future__ import annotations

from ..domain import ConceptKind

#: Longest prefix wins, so `DSA/01_Patterns` beats a bare `DSA`.
FOLDER_KINDS: dict[str, ConceptKind] = {
    "DSA/01_Patterns": ConceptKind.PATTERN,
    "DSA/02_Algorithms": ConceptKind.ALGORITHM,
    "DSA/03_DataStructures": ConceptKind.DATA_STRUCTURE,
    "DSA/04_Problems": ConceptKind.TOPIC,
    "DSA/05_Templates": ConceptKind.TEMPLATE,
    "DSA/06_Complexity": ConceptKind.CONCEPT,
    "DSA/07_Interview": ConceptKind.TOPIC,
    "DSA/08_Mistakes": ConceptKind.TOPIC,
    "DSA/09_CheatSheets": ConceptKind.TOPIC,
    "Technologies/Docs": ConceptKind.TECHNOLOGY,
    "Technologies/Playbooks": ConceptKind.PLAYBOOK,
    "Technologies/Templates": ConceptKind.TEMPLATE,
    "Technologies/Prompt-Library": ConceptKind.TEMPLATE,
    "Technologies/Project-System": ConceptKind.TEMPLATE,
    "Projects": ConceptKind.PROJECT,
    "Courses": ConceptKind.TOPIC,
    "Career": ConceptKind.TOPIC,
    "Resources": ConceptKind.TOPIC,
    "Reference": ConceptKind.CONCEPT,
}


def kind_for(rel_path: str) -> ConceptKind:
    """Kind implied by the folder a page lives in."""
    best: tuple[int, ConceptKind] = (0, ConceptKind.CONCEPT)
    for prefix, kind in FOLDER_KINDS.items():
        if rel_path.startswith(prefix + "/") and len(prefix) > best[0]:
            best = (len(prefix), kind)
    return best[1]


#: Namespace recorded on a concept whose bare name collides with another page.
#: Mirrors the namespaces used in `config/concept-identity.yaml`.
KIND_NAMESPACES: dict[ConceptKind, str] = {
    ConceptKind.PATTERN: "pattern",
    ConceptKind.ALGORITHM: "algorithm",
    ConceptKind.DATA_STRUCTURE: "data-structure",
    ConceptKind.TECHNOLOGY: "technology",
    ConceptKind.TEMPLATE: "templates",
    ConceptKind.PLAYBOOK: "playbook",
    ConceptKind.PROJECT: "project",
    ConceptKind.TOPIC: "topic",
    ConceptKind.CONCEPT: "concept",
}
