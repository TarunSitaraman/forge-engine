"""Applying approved proposals to the vault.

**Disabled by default.** Every entry point requires an explicit ``apply=True``,
which the CLI only sets when the user passes ``--apply``. Without it this module
computes and reports the exact changes and writes nothing.

Guarantees, all enforced here rather than documented and hoped for:

1. **Backup first.** The original file is copied into ``.forge/backups/`` before
   a byte is written, so every application is reversible.
2. **A revision is recorded** for the affected source, holding before and after.
3. **Exactly which files change is shown** before anything happens.
4. **Ambiguous changes are refused.** Only ``DETERMINISTIC_VERIFIED`` proposals
   are applicable, and only when approved.
5. **Unrelated files are never touched.** Each proposal names one file, one
   line, and the current content of that line must still match what the
   proposal recorded — otherwise the file changed underneath us and the
   application is refused.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..domain import (
    EntityType,
    Proposal,
    ProposalStatus,
    SafetyClass,
    record_change,
)
from ..logging import get_logger
from ..storage.sqlite_store import SqliteStore

log = get_logger(__name__)


@dataclass
class ApplyOutcome:
    """What happened, or would happen, for one proposal."""

    proposal_id: str
    path: str
    applied: bool
    refused_reason: str | None = None
    backup_path: str | None = None
    line: int | None = None
    before: str | None = None
    after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "path": self.path,
            "applied": self.applied,
            "refused_reason": self.refused_reason,
            "backup_path": self.backup_path,
            "line": self.line,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class ApplyReport:
    dry_run: bool
    outcomes: list[ApplyOutcome] = field(default_factory=list)

    @property
    def applied(self) -> list[ApplyOutcome]:
        return [o for o in self.outcomes if o.applied]

    @property
    def refused(self) -> list[ApplyOutcome]:
        return [o for o in self.outcomes if o.refused_reason]

    def files_changed(self) -> list[str]:
        return sorted({o.path for o in self.applied})

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "applied": len(self.applied),
            "refused": len(self.refused),
            "files_changed": self.files_changed(),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


class ProposalApplier:
    """Applies approved, verified metadata repairs to the vault."""

    def __init__(self, vault_path: Path, store: SqliteStore, backup_dir: Path) -> None:
        self.vault_path = vault_path
        self.store = store
        self.backup_dir = backup_dir

    def apply(self, proposals: Sequence[Proposal], *, apply: bool = False) -> ApplyReport:
        """Apply proposals, or report what would happen when ``apply`` is False."""
        report = ApplyReport(dry_run=not apply)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        for proposal in proposals:
            outcome = self._one(proposal, apply=apply, stamp=stamp)
            report.outcomes.append(outcome)

        log.info(
            "proposals_apply",
            dry_run=report.dry_run,
            applied=len(report.applied),
            refused=len(report.refused),
        )
        return report

    def _one(self, proposal: Proposal, *, apply: bool, stamp: str) -> ApplyOutcome:
        path_str = proposal.operation.target
        outcome = ApplyOutcome(
            proposal_id=proposal.id,
            path=path_str,
            applied=False,
            line=int(proposal.operation.details.get("line", 0)) or None,
            before=proposal.operation.before,
            after=proposal.operation.after,
        )

        if (refusal := self._refusal(proposal)) is not None:
            outcome.refused_reason = refusal
            return outcome

        target = self.vault_path / path_str
        if not target.is_file():
            outcome.refused_reason = f"file no longer exists: {path_str}"
            return outcome

        original = target.read_text(encoding="utf-8")
        lines = original.split("\n")
        index = int(proposal.operation.details["line"]) - 1

        if not (0 <= index < len(lines)):
            outcome.refused_reason = (
                f"line {index + 1} is out of range for a {len(lines)}-line file"
            )
            return outcome

        # The file must still look exactly as the proposal recorded. If it has
        # been edited since, the proposal is stale and applying it could
        # clobber the user's own change.
        if lines[index] != proposal.operation.before:
            outcome.refused_reason = (
                f"line {index + 1} has changed since the proposal was generated; "
                f"expected {proposal.operation.before!r}, found {lines[index]!r}"
            )
            return outcome

        if not apply:
            return outcome  # dry run: computed and verified, nothing written

        backup = self._backup(target, path_str, stamp)
        outcome.backup_path = str(backup)

        lines[index] = proposal.operation.after or ""
        updated = "\n".join(lines)
        target.write_text(updated, encoding="utf-8")

        self._record(proposal, path_str, original, updated)
        outcome.applied = True
        log.info("proposal_applied", proposal_id=proposal.id, path=path_str)
        return outcome

    def _refusal(self, proposal: Proposal) -> str | None:
        """Reasons a proposal may not be applied. Conservative by design."""
        if proposal.status is not ProposalStatus.APPROVED:
            return f"not approved (status: {proposal.status.value})"
        if proposal.safety is not SafetyClass.DETERMINISTIC_VERIFIED:
            return (
                f"safety class {proposal.safety.value} is not automatically applicable; "
                f"only deterministic, verified repairs are"
            )
        if proposal.operation.action != "replace_frontmatter_line":
            return f"unsupported operation for write-back: {proposal.operation.action}"
        if proposal.operation.before is None or proposal.operation.after is None:
            return "proposal does not record both before and after content"
        if "line" not in proposal.operation.details:
            return "proposal does not record a line number"
        return None

    def _backup(self, target: Path, rel_path: str, stamp: str) -> Path:
        backup = self.backup_dir / stamp / rel_path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        return backup

    def _record(self, proposal: Proposal, rel_path: str, before: str, after: str) -> None:
        """Record the vault mutation against its Source, with both states."""
        source = self.store.get_source_by_locator(rel_path)
        if source is None:
            return
        self.store.append_revision(
            record_change(
                EntityType.SOURCE,
                source.id,
                {"path": rel_path, "content": before},
                {"path": rel_path, "content": after},
                cause=proposal.id,
                note=f"applied proposal {proposal.id[:10]} ({proposal.type.value})",
            )
        )
