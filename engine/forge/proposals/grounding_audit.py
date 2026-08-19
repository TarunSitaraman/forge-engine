"""Re-check stored evidence quotes against the current grounding rule.

Grounding is a deterministic string check, so it can be applied retroactively
to an existing store at **zero model calls**. That is what makes this useful:
when the rule tightens, the corpus already extracted under the looser rule can
be held to the new one without re-running extraction.

That matters because re-extracting is not cheap and not free of consequence.
``EXTRACTOR_VERSION`` is part of the derivation key, so bumping it to force a
re-run discards every cached extraction result. Auditing costs seconds.

Used by ``forge proposals audit-grounding``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..extraction.extractor import _grounded, _ordered_overlap, _tokens
from ..storage.sqlite_store import SqliteStore


@dataclass(frozen=True)
class QuoteCheck:
    """One stored quote, re-checked against the span it cites."""

    proposal_id: str
    status: str
    span_id: str
    quote: str
    grounded: bool
    overlap: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "status": self.status,
            "span_id": self.span_id,
            "quote": self.quote,
            "grounded": self.grounded,
            "overlap": self.overlap,
            "note": self.note,
        }


def audit(store: SqliteStore, *, limit: int = 100_000) -> list[QuoteCheck]:
    """Every proposal carrying an evidence quote, re-checked against its span."""
    checks: list[QuoteCheck] = []
    for proposal in store.list_proposals(limit=limit):
        quote = (proposal.operation.details or {}).get("evidence_quote")
        if not quote:
            continue
        for span_id in proposal.evidence_span_ids:
            span = store.get_span(span_id)
            if span is None:
                # A cited span that no longer exists is not a pass. Silently
                # treating a missing span as "fine" would be the same mistake
                # the grounding rule itself exists to prevent.
                checks.append(
                    QuoteCheck(
                        proposal_id=proposal.id,
                        status=proposal.status.value,
                        span_id=span_id,
                        quote=quote,
                        grounded=False,
                        overlap=0.0,
                        note="span missing from store",
                    )
                )
                continue
            checks.append(
                QuoteCheck(
                    proposal_id=proposal.id,
                    status=proposal.status.value,
                    span_id=span_id,
                    quote=quote,
                    grounded=_grounded(quote, span.text),
                    overlap=round(_ordered_overlap(_tokens(quote), _tokens(span.text)), 3),
                )
            )
    return checks
