"""The handful of numbers a reader actually looks at first.

Both renderers derive from this rather than each reaching into the payload,
so an HTML report and a Markdown one can never disagree about how many links
are dead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Headline:
    files: int
    links: int
    dead_links: int
    dead_link_targets: int
    without_frontmatter: int
    invalid_frontmatter: int
    convention_conflicts: int
    graph_errors: int
    #: None when the graph has never been populated, which is not an error --
    #: `forge index` alone does not create concepts.
    graph_checked: int | None

    def rows(self) -> list[tuple[str, str, bool]]:
        """(label, value, is_ok) — ordered by what a reader cares about."""
        out = [
            (
                "Dead links",
                f"{self.dead_links:,} occurrence(s) across "
                f"{self.dead_link_targets:,} target(s)",
                self.dead_links == 0,
            ),
            (
                "Files without frontmatter",
                f"{self.without_frontmatter:,} of {self.files:,}",
                self.without_frontmatter == 0,
            ),
            (
                "Invalid frontmatter",
                f"{self.invalid_frontmatter:,}",
                self.invalid_frontmatter == 0,
            ),
            (
                "Convention conflicts",
                f"{self.convention_conflicts:,}",
                self.convention_conflicts == 0,
            ),
        ]
        if self.graph_checked:
            out.append(
                ("Graph integrity", f"{self.graph_errors:,} error(s)", self.graph_errors == 0)
            )
        return out

    @property
    def clean(self) -> bool:
        return all(ok for _, _, ok in self.rows())


def headline(payload: dict[str, Any]) -> Headline:
    links = payload.get("links", {})
    link_summary = links.get("summary", {})
    fm_summary = payload.get("frontmatter", {}).get("summary", {})
    graph = payload.get("graph", {})
    checked = graph.get("checked") or {}

    return Headline(
        files=int(fm_summary.get("total_files", 0)),
        links=int(link_summary.get("total_links", 0)),
        dead_links=int(link_summary.get("unresolved_occurrences", 0)),
        dead_link_targets=int(link_summary.get("unresolved_distinct_targets", 0)),
        without_frontmatter=int(fm_summary.get("without_frontmatter", 0)),
        invalid_frontmatter=int(fm_summary.get("invalid", 0)),
        convention_conflicts=len(payload.get("conventions", {}).get("conflicts", [])),
        graph_errors=int(graph.get("errors", 0)),
        graph_checked=sum(int(v) for v in checked.values()) if checked else None,
    )
