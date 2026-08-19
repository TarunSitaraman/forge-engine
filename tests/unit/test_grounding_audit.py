"""The offline re-check of stored evidence quotes.

Grounding is deterministic, so a rule change can be applied retroactively to
an existing store without re-calling a model. These tests pin that: the audit
must flag a quote the current rule rejects even though an earlier, looser rule
admitted it — which is precisely the corpus state the 2026-08-19 fix leaves
behind, since bumping the extractor version to force re-extraction would
discard every cached result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from audit_grounding import audit  # noqa: E402

from forge.domain import (  # noqa: E402
    Derivation,
    Document,
    EntityType,
    Proposal,
    ProposalType,
    ProposedOperation,
    Provenance,
    ProvenanceTier,
    SafetyClass,
    Source,
    SourceKind,
    Span,
)
from forge.storage import SqliteStore  # noqa: E402

SPAN_TEXT = (
    "Retrieval Augmented Generation grounds generation in retrieved passages, "
    "which reduces hallucination on open-domain questions considerably."
)


@pytest.fixture
def store(tmp_path):
    store = SqliteStore(tmp_path / "forge.db")
    store.initialize()
    source = Source.for_path("Technologies/Docs/rag.md", kind=SourceKind.MARKDOWN, content_hash="h")
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "h"),
        source_id=source.id,
        parser="p",
        parser_version="1",
        content_hash="h",
    )
    store.put_document(document)
    store.put_spans(
        [
            Span(
                id="sp1",
                document_id=document.id,
                ordinal=0,
                locator="p.1",
                start_line=1,
                end_line=3,
                text=SPAN_TEXT,
                content_hash="h",
            )
        ]
    )
    yield store
    store.close()


def _proposal(quote: str, ident: str) -> Proposal:
    return Proposal(
        id=ident,
        type=ProposalType.NEW_CLAIM,
        safety=SafetyClass.MODEL_GENERATED,
        target_entity_type=EntityType.CLAIM,
        operation=ProposedOperation(
            action="create_claim",
            target="a claim",
            after="a claim",
            details={"evidence_quote": quote, "concept": "RAG"},
        ),
        reason="test",
        evidence_span_ids=("sp1",),
        provenance=Provenance(
            tier=ProvenanceTier.EXTRACTED_CLAIM,
            derivation=Derivation.MODEL,
            confidence=0.5,
            agent="test/1",
            model_id="test-model",
        ),
    )


def test_a_genuinely_grounded_quote_passes(store):
    store.put_proposal(_proposal("grounds generation in retrieved passages", "p-good"))
    rows = audit(store)
    assert len(rows) == 1
    assert rows[0]["grounded"] is True


def test_a_quote_reordered_from_span_vocabulary_is_flagged(store):
    """The exact case the pre-fix bag-of-words rule let through."""
    store.put_proposal(
        _proposal("retrieved passages reduces generation grounds hallucination", "p-bad")
    )
    rows = audit(store)
    assert len(rows) == 1
    assert rows[0]["grounded"] is False
    assert rows[0]["overlap"] < 0.9


def test_audit_makes_no_model_calls(store):
    from forge.llm.base import CALLS

    store.put_proposal(_proposal("grounds generation in retrieved passages", "p-good"))
    CALLS.reset()
    audit(store)
    assert CALLS.count == 0


def test_a_missing_span_is_reported_not_silently_passed(store):
    proposal = _proposal("grounds generation in retrieved passages", "p-orphan")
    store.put_proposal(proposal.model_copy(update={"evidence_span_ids": ("sp-nope",)}))
    rows = audit(store)
    assert rows[0]["grounded"] is False
    assert "span missing" in rows[0]["note"]


def test_proposals_without_a_quote_are_skipped(store):
    proposal = _proposal("grounds generation in retrieved passages", "p-noquote")
    store.put_proposal(
        proposal.model_copy(
            update={"operation": proposal.operation.model_copy(update={"details": {}})}
        )
    )
    assert audit(store) == []
