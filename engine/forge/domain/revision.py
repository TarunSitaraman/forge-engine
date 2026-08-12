"""Revision tracking — append-only history of derived state.

Two things this is *not*:

* It is not Git. Git tracks the Markdown corpus. This tracks the engine's
  derived model, which Git never sees.
* It is not an audit log bolted on afterwards. It exists from the first write,
  because history cannot be reconstructed retroactively — a system that starts
  logging revisions in Phase 9 has no history for Phases 1-8.

Storage-agnostic by construction: a Revision is a plain record of
(entity, op, before, after, cause). Nothing about it presumes a graph database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..ids import new_id
from .enums import EntityType, RevisionOp
from .provenance import utc_now


class Revision(BaseModel):
    """One recorded change to one derived object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=new_id)
    entity_type: EntityType
    entity_id: str
    op: RevisionOp

    #: Serialized state before/after. ``before`` is None for CREATE;
    #: ``after`` is None for INVALIDATE.
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None

    #: What caused this change — usually a source_id or claim_id. This is what
    #: makes "why did my understanding change?" answerable.
    cause: str | None = None
    workflow_run_id: str | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _shape(self) -> Revision:
        if self.op is RevisionOp.CREATE and self.before is not None:
            raise ValueError("CREATE revision must not carry a `before` state")
        if self.op is RevisionOp.CREATE and self.after is None:
            raise ValueError("CREATE revision must carry an `after` state")
        if self.op in (RevisionOp.CHANGE, RevisionOp.SUPERSEDE) and (
            self.before is None or self.after is None
        ):
            raise ValueError(f"{self.op.value} revision requires both `before` and `after`")
        if self.op is RevisionOp.INVALIDATE and self.before is None:
            raise ValueError("INVALIDATE revision must carry the `before` state it invalidated")
        return self


def record_create(
    entity_type: EntityType, entity_id: str, after: dict[str, Any], **kw: Any
) -> Revision:
    return Revision(
        entity_type=entity_type, entity_id=entity_id, op=RevisionOp.CREATE, after=after, **kw
    )


def record_change(
    entity_type: EntityType,
    entity_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    **kw: Any,
) -> Revision:
    return Revision(
        entity_type=entity_type,
        entity_id=entity_id,
        op=RevisionOp.CHANGE,
        before=before,
        after=after,
        **kw,
    )


def record_supersede(
    entity_type: EntityType,
    entity_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    superseded_by: str,
    **kw: Any,
) -> Revision:
    """Supersession retains both states. This is Principle 11's enforcement point."""
    return Revision(
        entity_type=entity_type,
        entity_id=entity_id,
        op=RevisionOp.SUPERSEDE,
        before=before,
        after=after,
        cause=kw.pop("cause", superseded_by),
        note=kw.pop("note", f"superseded by {superseded_by}"),
        **kw,
    )


def record_invalidate(
    entity_type: EntityType, entity_id: str, before: dict[str, Any], **kw: Any
) -> Revision:
    return Revision(
        entity_type=entity_type, entity_id=entity_id, op=RevisionOp.INVALIDATE, before=before, **kw
    )
