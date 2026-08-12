"""Machine-readable representation of Forge's competing convention systems.

The Phase 0 audit found two convention systems in active use that contradict
each other on filenames, tags, and frontmatter:

* ``CONVENTIONS.md``            — repo-wide: kebab-case, namespaced ``#type/``
  tags, minimal frontmatter "only when it carries real metadata".
* ``DSA/Documentation Standards.md`` — DSA-local: Title Case filenames,
  ``dsa/pattern`` style tags, frontmatter mandatory on every page.

**This module does not choose between them.** Choosing is an architectural
decision that belongs to a human (ADR-001 D3). What it does is make both
systems explicit and measure conformance, so the decision can be made against
evidence instead of preference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .model import CorpusIndex

_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*\.md$")
_TITLE_RE = re.compile(r"^[A-Z0-9][^/]*\.md$")
_NAMESPACED_TAG_RE = re.compile(r"^[a-z]+/[a-z0-9-]+$")
_DSA_TAG_RE = re.compile(r"^(dsa|forge)/[a-z0-9-]+$")


@dataclass(frozen=True)
class ConventionRule:
    id: str
    description: str
    #: Rule kind, so callers can evaluate without parsing prose.
    kind: str  # "filename" | "tags" | "frontmatter"
    expected: str


@dataclass(frozen=True)
class ConventionSystem:
    id: str
    name: str
    defined_in: str
    #: Path prefixes this system claims authority over ("" = whole repo).
    scope: tuple[str, ...]
    rules: tuple[ConventionRule, ...]

    def claims(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.scope) if self.scope else True


REPO_WIDE = ConventionSystem(
    id="repo-wide",
    name="Forge repository conventions",
    defined_in="CONVENTIONS.md",
    scope=(),
    rules=(
        ConventionRule(
            id="repo.filename.kebab",
            description="Files use kebab-case.md",
            kind="filename",
            expected="kebab-case",
        ),
        ConventionRule(
            id="repo.tags.namespaced",
            description="Tags are namespaced (#status/, #type/, #stack/), max 3 per file",
            kind="tags",
            expected="namespaced",
        ),
        ConventionRule(
            id="repo.frontmatter.minimal",
            description="Frontmatter only when it carries real metadata",
            kind="frontmatter",
            expected="optional",
        ),
    ),
)

DSA_LOCAL = ConventionSystem(
    id="dsa-local",
    name="DSA documentation standards",
    defined_in="DSA/Documentation Standards.md",
    scope=("DSA/",),
    rules=(
        ConventionRule(
            id="dsa.filename.title",
            description="Files use readable Title Case names, e.g. 'Binary Search.md'",
            kind="filename",
            expected="title-case",
        ),
        ConventionRule(
            id="dsa.tags.scoped",
            description="Tags grouped by scope: dsa/pattern, dsa/algorithm, ...",
            kind="tags",
            expected="dsa-scoped",
        ),
        ConventionRule(
            id="dsa.frontmatter.required",
            description="Frontmatter required on every knowledge page with type/status/tags/canonical",
            kind="frontmatter",
            expected="required",
        ),
    ),
)

SYSTEMS: tuple[ConventionSystem, ...] = (REPO_WIDE, DSA_LOCAL)

#: The rule pairs that directly contradict each other.
KNOWN_CONFLICTS: tuple[dict[str, str], ...] = (
    {
        "kind": "filename",
        "repo_wide": "kebab-case.md",
        "dsa_local": "Title Case.md",
        "note": "Both are in active use; neither is documented as deliberate.",
    },
    {
        "kind": "tags",
        "repo_wide": "#status/, #type/, #stack/ namespaces, max 3",
        "dsa_local": "dsa/pattern, dsa/algorithm, ... scopes",
        "note": "Different vocabularies for the same field.",
    },
    {
        "kind": "frontmatter",
        "repo_wide": "optional, only when meaningful",
        "dsa_local": "mandatory on every page",
        "note": "Directly opposed obligations for files under DSA/.",
    },
)


@dataclass
class ConventionReport:
    systems: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    conformance: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolution_status: str = "UNRESOLVED — requires human decision (ADR-001 D3)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution_status": self.resolution_status,
            "systems": self.systems,
            "conflicts": self.conflicts,
            "conformance": self.conformance,
        }


def analyze_conventions(index: CorpusIndex) -> ConventionReport:
    """Measure how the corpus actually conforms to each system.

    Reports both. Chooses neither.
    """
    conformance: dict[str, dict[str, Any]] = {}

    for system in SYSTEMS:
        scoped = [f for f in index.files if system.claims(f.path)]
        filename_ok = tags_ok = fm_ok = 0

        for f in scoped:
            name = f.path.rsplit("/", 1)[-1]
            if system.id == "repo-wide":
                if _KEBAB_RE.match(name):
                    filename_ok += 1
                if f.tags and all(_NAMESPACED_TAG_RE.match(t) for t in f.tags):
                    tags_ok += 1
                fm_ok += 1  # optional: every file trivially conforms
            else:
                if _TITLE_RE.match(name) and not _KEBAB_RE.match(name):
                    filename_ok += 1
                if f.tags and all(_DSA_TAG_RE.match(t) for t in f.tags):
                    tags_ok += 1
                if f.frontmatter_present:
                    fm_ok += 1

        total = len(scoped) or 1
        conformance[system.id] = {
            "scope": list(system.scope) or ["<entire repository>"],
            "files_in_scope": len(scoped),
            "filename_conforming": filename_ok,
            "filename_pct": round(100 * filename_ok / total, 1),
            "tags_conforming": tags_ok,
            "tags_pct": round(100 * tags_ok / total, 1),
            "frontmatter_conforming": fm_ok,
            "frontmatter_pct": round(100 * fm_ok / total, 1),
        }

    return ConventionReport(
        systems=[
            {
                "id": s.id,
                "name": s.name,
                "defined_in": s.defined_in,
                "scope": list(s.scope) or ["<entire repository>"],
                "rules": [
                    {
                        "id": r.id,
                        "kind": r.kind,
                        "expected": r.expected,
                        "description": r.description,
                    }
                    for r in s.rules
                ],
            }
            for s in SYSTEMS
        ],
        conflicts=[dict(c) for c in KNOWN_CONFLICTS],
        conformance=conformance,
    )
