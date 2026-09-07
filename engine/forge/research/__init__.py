"""Phase 9: the research-intelligence queries.

The vision names six questions as the product's real acceptance criteria, and
says they are "deliberately not answerable by a search box". This package is
where they become answerable:

1. *What do I currently believe about X?* -> :func:`belief.belief_for_concept`
2. *Which sources support that belief? Which disagree?* -> the same call
3. *What changed in my understanding this month?* -> :func:`changes.changes_since`
4. *What questions remain unanswered?* -> :func:`gaps.open_questions`
5. *What concepts am I missing?* -> :func:`gaps.detect_gaps`
6. *Which papers are most relevant to this unresolved question?* ->
   :func:`gaps.evidence_for_question`

**Every one is deterministic.** No model is called anywhere in this package.
That is not an implementation shortcut: a gap report or a belief summary that a
model generated would be exactly the kind of plausible, unfalsifiable output
this system exists to avoid. What these return are graph facts, with the
provenance of everything they rest on.
"""

from __future__ import annotations

from .belief import Belief, Dissent, belief_for_concept
from .changes import ChangeReport, changes_since
from .gaps import GapReport, detect_gaps, evidence_for_question, gap_report, open_questions

__all__ = [
    "Belief",
    "ChangeReport",
    "Dissent",
    "GapReport",
    "belief_for_concept",
    "changes_since",
    "detect_gaps",
    "evidence_for_question",
    "gap_report",
    "open_questions",
]
