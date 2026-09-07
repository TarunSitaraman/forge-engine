"""What changed in my understanding, over a window.

The vision asks for "temporal versioning of the model itself", and the
`revisions` table already is that: every change to a derived object was
recorded as it happened, with what caused it. This module reads that log rather
than diffing snapshots, which matters because a diff can only show the
endpoints. A claim created, disputed, and superseded inside the window is three
facts about how understanding moved, and one diff would show it as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..domain import EntityType, RevisionOp, utc_now
from ..storage import SqliteStore

#: Revisions to read when scanning a window. The log is append-only and this
#: is a local SQLite table, so the cost of a generous ceiling is small; the
#: ceiling exists so a corpus-scale backfill cannot make one query unbounded.
MAX_REVISIONS_SCANNED = 20_000


@dataclass
class ChangeReport:
    """What moved between two instants, and what caused it."""

    since: datetime
    until: datetime
    #: Revision counts by entity type then operation.
    by_entity: dict[str, dict[str, int]] = field(default_factory=dict)
    claims_created: list[str] = field(default_factory=list)
    claims_superseded: list[str] = field(default_factory=list)
    claims_disputed: list[str] = field(default_factory=list)
    concepts_created: list[str] = field(default_factory=list)
    syntheses_staled: list[str] = field(default_factory=list)
    #: Distinct `cause` values seen, which is what answers "why did it change".
    causes: list[str] = field(default_factory=list)
    total: int = 0
    #: True when the window hit `MAX_REVISIONS_SCANNED`, so the counts below
    #: are a floor rather than a total. Reported, never silently truncated.
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "total": self.total,
            "truncated": self.truncated,
            "by_entity": self.by_entity,
            "claims_created": self.claims_created,
            "claims_superseded": self.claims_superseded,
            "claims_disputed": self.claims_disputed,
            "concepts_created": self.concepts_created,
            "syntheses_staled": self.syntheses_staled,
            "causes": self.causes,
        }


def changes_since(
    store: SqliteStore,
    since: datetime | None = None,
    *,
    until: datetime | None = None,
    days: int = 30,
) -> ChangeReport:
    """Everything the model recorded between `since` and `until`.

    Defaults to the last `days` days, so "what changed this month" is the
    no-argument call.
    """
    until = until or utc_now()
    since = since or (until - timedelta(days=days))
    report = ChangeReport(since=since, until=until)

    revisions = store.recent_revisions(limit=MAX_REVISIONS_SCANNED)
    report.truncated = len(revisions) >= MAX_REVISIONS_SCANNED

    causes: dict[str, None] = {}
    for revision in revisions:
        at = revision.created_at
        if at < since or at > until:
            continue
        report.total += 1
        by_op = report.by_entity.setdefault(revision.entity_type.value, {})
        by_op[revision.op.value] = by_op.get(revision.op.value, 0) + 1
        if revision.cause:
            causes[revision.cause] = None

        if revision.entity_type is EntityType.CLAIM:
            if revision.op is RevisionOp.CREATE:
                report.claims_created.append(revision.entity_id)
            elif revision.op is RevisionOp.SUPERSEDE:
                report.claims_superseded.append(revision.entity_id)
            elif revision.op is RevisionOp.CHANGE and _became_disputed(revision):
                report.claims_disputed.append(revision.entity_id)
        elif revision.entity_type is EntityType.CONCEPT and revision.op is RevisionOp.CREATE:
            report.concepts_created.append(revision.entity_id)
        elif revision.entity_type is EntityType.SYNTHESIS and _became_stale(revision):
            report.syntheses_staled.append(revision.entity_id)

    report.causes = sorted(causes)
    return report


def _became_disputed(revision: Any) -> bool:
    before = (revision.before or {}).get("status")
    after = (revision.after or {}).get("status")
    return after == "disputed" and before != "disputed"


def _became_stale(revision: Any) -> bool:
    before = (revision.before or {}).get("stale")
    after = (revision.after or {}).get("stale")
    return bool(after) and not bool(before)
