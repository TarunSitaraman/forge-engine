"""The Phase 2 ingestion pipeline.

    EXTERNAL SOURCE -> SOURCE REGISTRATION -> CONTENT ACQUISITION
      -> DETERMINISTIC EXTRACTION -> DOCUMENT + SPAN CREATION
      -> HASH / CHANGE DETECTION -> PROVENANCE -> PERSISTED KNOWLEDGE INPUT

Everything up to and including span persistence is deterministic and works with
no model installed. Semantic extraction is a strictly optional stage bolted on
after the deterministic pipeline has already succeeded — so a failure, or the
absence, of an LLM can never cost you the ingest.

Cost control is structural rather than best-effort:

* an unchanged source short-circuits before acquisition, costing nothing;
* extraction results are cached under a derivation key covering content,
  processor version, model, prompt version, and schema version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import Settings
from ..domain import (
    ChangeStatus,
    Document,
    ProposalStatus,
    ProposalType,
    EvidenceRelation,
    ExtractionStatus,
    IngestionStatus,
    MatchKind,
    Source,
    SourceKind,
    TrustTier,
    deterministic_provenance,
    ProvenanceTier,
)
from ..extraction.extractor import CandidateExtractor, ExtractionResult, extraction_provenance
from ..llm.base import CALLS
from ..logging import get_logger
from ..matching.matcher import ConceptMatcher, build_ambiguity_index
from ..proposals.service import ProposalService, claim_proposal, concept_proposal
from ..sources.base import AcquisitionResult
from ..sources.registry import AdapterRegistry
from ..storage.sqlite_store import SqliteStore
from .chunking import CHUNK_STRATEGY, build_spans
from .derivation import CacheStats, extraction_key
from .report import IngestionReport, SourceReport

log = get_logger(__name__)

PIPELINE_VERSION = "ingest/0.2.0"


@dataclass
class IngestOptions:
    """Per-run knobs. Defaults are the safe, cheap, offline path."""

    #: Run LLM extraction. Off unless asked for: ingestion's job is evidence.
    extract: bool = False
    #: Re-process even when the content hash is unchanged.
    force: bool = False
    #: Persist spans and documents.
    persist: bool = True
    #: Generate concept/claim proposals from extraction results.
    propose: bool = True
    #: Cap on spans sent to the model per document.
    max_spans: int = 12


class IngestionPipeline:
    """Ingests local files into the canonical knowledge model."""

    version = PIPELINE_VERSION

    def __init__(
        self,
        settings: Settings,
        store: SqliteStore,
        *,
        registry: AdapterRegistry | None = None,
        extractor: CandidateExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry or AdapterRegistry()
        self.extractor = extractor
        self.proposals = ProposalService(store)
        self._vault_path_cache: list[str] | None = None

    # -- entry points ------------------------------------------------------

    def ingest_path(self, path: Path, options: IngestOptions | None = None) -> IngestionReport:
        """Ingest a file, or every supported file under a directory."""
        opts = options or IngestOptions()
        started = time.perf_counter()
        report = IngestionReport(cache=CacheStats())

        targets = self._discover(path) if path.is_dir() else [path]
        if not targets:
            report.add(
                SourceReport(
                    locator=path.as_posix(),
                    status=IngestionStatus.UNSUPPORTED,
                    detail=(
                        f"nothing ingestible at {path}; supported: "
                        f"{', '.join(self.registry.supported_extensions())}"
                    ),
                )
            )
            report.duration_seconds = time.perf_counter() - started
            return report

        for position, target in enumerate(targets, start=1):
            # Rebuilt per source so each file sees what earlier files proposed.
            report.add(self._ingest_one(target, opts, self._matcher(), report.cache))
            if opts.extract:
                # Extraction runs for hours; a run with no visible progress
                # cannot be distinguished from a stalled one. The estimate is
                # measured-so-far, not a prediction: it is simply the mean
                # per-source time of this run projected over what is left, and
                # it will swing early on and while cached sources fly past.
                elapsed = time.perf_counter() - started
                remaining = (elapsed / position) * (len(targets) - position)
                log.info(
                    "ingest_progress",
                    source=position,
                    of=len(targets),
                    elapsed_minutes=round(elapsed / 60, 1),
                    estimated_remaining_minutes=round(remaining / 60, 1),
                    llm_calls=CALLS.count,
                )

        report.duration_seconds = time.perf_counter() - started
        log.info("ingestion_run_complete", **report.totals())
        return report

    def _discover(self, root: Path) -> list[Path]:
        """Find ingestible files under a directory, honouring the exclude list.

        The adapter registry globs everything it can parse; it has no notion of
        which directories belong to the vault. Without this filter, ingesting
        the vault root also sweeps in `tests/fixtures/`, `engine/`, and
        `.forge/` — turning the engine's own test data into user knowledge.
        """
        excludes = set(self.settings.exclude_dirs)
        out: list[Path] = []
        for candidate in self.registry.discover(root):
            try:
                relative = candidate.resolve().relative_to(self.settings.vault_path)
            except ValueError:
                out.append(candidate)  # outside the vault: caller asked for it explicitly
                continue
            parts = relative.parts[:-1]
            if any(part in excludes or part.startswith(".") for part in parts):
                continue
            out.append(candidate)
        return out

    # -- per-source --------------------------------------------------------

    def _ingest_one(
        self,
        path: Path,
        opts: IngestOptions,
        matcher: ConceptMatcher,
        cache: CacheStats,
    ) -> SourceReport:
        started = time.perf_counter()
        locator = self._locator(path)
        report = SourceReport(locator=locator, status=IngestionStatus.INGESTED)

        # --- change detection, before any parsing work -------------------
        existing = self.store.get_source_by_locator(locator)
        if existing is not None and not opts.force:
            probe = self.registry.acquire(path)
            if not probe.ok:
                return self._failed(report, probe, started)
            stored_spans = self._spans_for_source(existing.id)
            if probe.content_hash == existing.content_hash and stored_spans:
                report.status = IngestionStatus.UNCHANGED
                report.source_id = existing.id
                report.content_hash = existing.content_hash
                report.spans = len(stored_spans)

                # Unchanged content does not imply extracted content. A vault
                # ingested deterministically first — the normal order, since
                # extraction is opt-in and expensive — has every source stored
                # with a matching hash and nothing extracted. Short-circuiting
                # here on that basis made `--extract` a silent no-op, and broke
                # resumability: an interrupted run, restarted, skipped every
                # source it had already *ingested* rather than every source it
                # had already *extracted*.
                #
                # Re-deriving the document would bump its version and duplicate
                # spans for no reason, so extraction runs over the stored spans
                # instead. Repeat runs still cost zero calls — that guarantee
                # comes from the derivation cache inside `_extract`, which is
                # where it belongs.
                if opts.extract and self.extractor is not None:
                    return self._extract_only(report, existing, opts, matcher, cache, started)

                report.extraction_status = ExtractionStatus.SKIPPED_CACHED
                report.duration_seconds = time.perf_counter() - started
                log.info("source_unchanged", locator=locator)
                return report
            if probe.content_hash == existing.content_hash:
                # Content unchanged, but this source has no spans *this*
                # chunker produced — so there is real work to do. See
                # `_spans_for_source` for why that is not the same question as
                # "does this source have spans".
                log.info("rechunking_for_ingestion", locator=locator)
            acquisition = probe
        else:
            acquisition = self.registry.acquire(path)
            if not acquisition.ok:
                return self._failed(report, acquisition, started)

        # --- source registration -----------------------------------------
        source = Source.for_path(
            locator,
            kind=acquisition.kind,
            content_hash=acquisition.content_hash,
            trust_tier=self._trust_tier(acquisition.kind),
            title=acquisition.title,
            byte_size=acquisition.byte_size,
            line_count=acquisition.line_count,
        )
        report.source_id = source.id
        report.content_hash = source.content_hash
        report.pages = acquisition.page_count
        report.chars = len(acquisition.text)
        report.warnings = list(acquisition.warnings)

        # --- document + spans (deterministic) ----------------------------
        document = Document(
            id=Document.make_id(source.id, acquisition.content_hash),
            source_id=source.id,
            version=self._next_version(source.id),
            parser=f"forge.{acquisition.kind.value}",
            parser_version=self.registry.processor_version(path),
            content_hash=acquisition.content_hash,
            headings=tuple(
                (b.heading_level or 1, b.text, b.start_line)
                for b in acquisition.blocks
                if b.is_heading
            ),
            frontmatter_present=bool(acquisition.metadata.get("frontmatter_present")),
            frontmatter_valid=bool(acquisition.metadata.get("frontmatter_valid")),
        )
        spans = build_spans(document.id, acquisition.blocks)
        report.document_id = document.id
        report.spans = len(spans)

        if opts.persist:
            # A modified source keeps its prior document rows; put_source
            # records a CHANGE revision holding both hashes, so history is
            # preserved rather than overwritten.
            self.store.put_source(source)
            self.store.put_document(document)
            if spans:
                self.store.put_spans(spans)
            if existing is not None:
                self.store.invalidate_derivations(source.id)

        # --- optional semantic extraction --------------------------------
        if opts.extract and spans:
            result = self._extract(source, spans, opts, cache)
            report.extraction_status = result.status
            report.llm_calls = result.llm_calls
            report.concepts_proposed = len(result.concepts)
            report.claims_proposed = len(result.claims)
            for failure in result.failures:
                report.errors.append(f"{failure.get('kind')}: {failure.get('error', '')[:160]}")

            if opts.propose:
                created = self._propose(result, source, matcher)
                report.proposals_created = created
                report.evidence_links = sum(
                    1 for c in result.claims if c.span_id
                )
        elif opts.extract:
            report.extraction_status = ExtractionStatus.SUCCEEDED

        report.duration_seconds = time.perf_counter() - started
        log.info(
            "source_ingested",
            locator=locator,
            spans=report.spans,
            pages=report.pages,
            llm_calls=report.llm_calls,
        )
        return report

    # -- stages ------------------------------------------------------------

    def _extract_only(
        self,
        report: SourceReport,
        source: Source,
        opts: IngestOptions,
        matcher: ConceptMatcher,
        cache: CacheStats,
        started: float,
    ) -> SourceReport:
        """Extract from an already-stored source without re-deriving it.

        Used when content is unchanged but extraction has not run against it.
        The spans come from the store, so no document version is created and no
        span is duplicated; the source stays exactly as it was on disk and in
        the database, and only proposals are added.
        """
        spans = self._spans_for_source(source.id)
        if not spans:
            report.extraction_status = ExtractionStatus.SKIPPED_CACHED
            report.duration_seconds = time.perf_counter() - started
            return report

        result = self._extract(source, spans, opts, cache)
        report.extraction_status = result.status
        report.llm_calls = result.llm_calls
        report.concepts_proposed = len(result.concepts)
        report.claims_proposed = len(result.claims)
        for failure in result.failures:
            report.errors.append(f"{failure.get('kind')}: {failure.get('error', '')[:160]}")

        if opts.propose:
            report.proposals_created = self._propose(result, source, matcher)
            report.evidence_links = sum(1 for c in result.claims if c.span_id)

        report.duration_seconds = time.perf_counter() - started
        log.info(
            "source_extracted_in_place",
            locator=report.locator,
            spans=len(spans),
            llm_calls=report.llm_calls,
            status=result.status.value,
        )
        return report

    def _extract(
        self,
        source: Source,
        spans: Sequence,
        opts: IngestOptions,
        cache: CacheStats,
    ) -> ExtractionResult:
        """Run extraction, reusing a cached result when nothing relevant changed."""
        if self.extractor is None:
            return ExtractionResult(status=ExtractionStatus.SKIPPED_NO_PROVIDER)

        key = extraction_key(
            content_hash=source.content_hash,
            processor_version=self.extractor.version,
            model_id=self.extractor.model_id(),
            prompt_version=self.extractor.prompt_version,
            schema_version=self.extractor.schema_version,
        )

        if (cached := self.store.get_derivation(key.value())) is not None:
            cache.hit()
            log.info("extraction_cache_hit", **key.describe())
            return ExtractionResult.from_dict(cached)

        cache.miss()
        before = CALLS.count
        result = self.extractor.extract(spans)
        result.llm_calls = max(result.llm_calls, CALLS.count - before)

        # Only cache outcomes worth reusing.
        #
        # SUCCEEDED only, deliberately. PARTIAL means some calls failed, and on
        # this hardware that is overwhelmingly a timeout — a transient failure,
        # not a property of the input. Caching it writes the transient failure
        # into the derivation key permanently: a span whose concept call timed
        # out returns `concepts=0` forever, and re-running reports a cache hit
        # rather than retrying. Observed 2026-08-20, when `_index.md` lost its
        # concepts to three consecutive timeouts and cached the empty result.
        #
        # This is the same reasoning that already excluded provider-unavailable
        # results, applied one step further. The cost is that a span which fails
        # deterministically re-spends its calls on every run; the benefit is
        # that a transient failure never becomes permanent. Given extraction is
        # resumable and re-running a *successful* scope is still free, that
        # trade is the right way round.
        if result.status is ExtractionStatus.SUCCEEDED:
            self.store.put_derivation(
                key.value(),
                "extraction",
                source.content_hash,
                result.to_dict(),
                source_id=source.id,
            )
            cache.write()
        return result

    def _propose(
        self, result: ExtractionResult, source: Source, matcher: ConceptMatcher
    ) -> int:
        """Turn extraction candidates into reviewable proposals."""
        if result.model_id is None:
            return 0

        spans_by_id = {s.id: s for s in self._spans_for_source(source.id)}
        created = 0

        seen: set[str] = set()
        for candidate in result.concepts:
            if candidate.name.casefold() in seen:
                continue
            seen.add(candidate.name.casefold())
            span = spans_by_id.get(candidate.span_id)
            if span is None:
                continue
            match = matcher.match(candidate.name)
            provenance = extraction_provenance(
                result.model_id, span, tier=ProvenanceTier.MODEL_INFERENCE
            )
            _, was_created = self.proposals.create(
                concept_proposal(candidate, match, provenance, source_id=source.id)
            )
            created += int(was_created)

        for candidate in result.claims:
            span = spans_by_id.get(candidate.span_id)
            if span is None:
                continue
            provenance = extraction_provenance(
                result.model_id, span, tier=ProvenanceTier.EXTRACTED_CLAIM
            )
            _, was_created = self.proposals.create(
                claim_proposal(candidate, provenance, source_id=source.id)
            )
            created += int(was_created)

        return created

    # -- helpers -----------------------------------------------------------

    def _vault_paths(self) -> list[str]:
        """Every Markdown path in the vault, scanned once per pipeline instance.

        The ambiguity index must be built from the **vault filesystem**, not
        from previously-ingested sources. A collision like `Heap` — a pattern
        and a data structure sharing a name — exists in the corpus whether or
        not either file has been ingested, and building the index only from
        stored sources would miss it entirely and report the concept as new.
        """
        if self._vault_path_cache is None:
            from ..corpus.indexer import CorpusIndexer

            try:
                self._vault_path_cache = CorpusIndexer(self.settings).discover()
            except Exception as exc:  # a missing vault must not break ingestion
                log.warning("vault_scan_failed", error=str(exc)[:120])
                self._vault_path_cache = []
        return self._vault_path_cache

    def _matcher(self) -> ConceptMatcher:
        """Build a matcher over stored concepts, prior proposals, and vault collisions.

        Rebuilt per source rather than once per run, so ingesting a directory
        lets the second file see what the first proposed. That costs a couple of
        cheap queries and is what makes overlap detection actually work.
        """
        vault_paths = self._vault_paths() + [s.locator for s in self.store.list_sources()]
        proposed = [
            (p.operation.target, p.id)
            for p in self.store.list_proposals(type=ProposalType.NEW_CONCEPT, limit=1000)
            if p.status is not ProposalStatus.REJECTED
        ]
        return ConceptMatcher(
            self.store.list_concepts(),
            ambiguity_index=build_ambiguity_index(vault_paths),
            proposed_concepts=proposed,
        )

    def _locator(self, path: Path) -> str:
        """Vault-relative when inside the vault, absolute otherwise.

        Matching the Phase 1 indexer's convention means a file ingested here
        and indexed there are recognised as the same source, not two.
        """
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.settings.vault_path).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _trust_tier(self, kind: SourceKind) -> TrustTier:
        """External material is UNVERIFIED; the user's own vault is USER_AUTHORED.

        A PDF someone downloaded is not evidence of the same standing as a note
        they wrote, and collapsing the two would poison the tiering of
        everything built on top.
        """
        return TrustTier.USER_AUTHORED if kind is SourceKind.MARKDOWN else TrustTier.UNVERIFIED

    def _next_version(self, source_id: str) -> int:
        return len(self.store.documents_for_source(source_id)) + 1

    def _spans_for_source(self, source_id: str) -> list:
        """Spans this pipeline owns — never another chunker's.

        `forge index` and `forge ingest` both write spans, for different jobs:
        Phase 1 produces heading-delimited spans for retrieval, ingestion
        produces structurally grouped, sentence-split ones for evidence. They
        share a table and a document, so an indexed-then-ingested vault holds
        both, and reading them all back conflates two chunkings.

        That is not hypothetical. Until 2026-08-19 the unchanged-source
        short-circuit reported and extracted over whatever spans existed, so a
        vault indexed first — the documented order — extracted over Phase 1's
        boundaries: 208 spans instead of 98 on `Technologies/Docs`, hence 416
        model calls instead of 196.

        Filtering rather than deleting is deliberate. The Phase 1 spans are not
        stale; retrieval is still using them.
        """
        spans: list = []
        for document in self.store.documents_for_source(source_id):
            spans.extend(
                span
                for span in self.store.spans_for_document(document.id)
                if span.chunk_strategy == CHUNK_STRATEGY
            )
        return spans

    def _failed(
        self, report: SourceReport, acquisition: AcquisitionResult, started: float
    ) -> SourceReport:
        report.status = acquisition.status
        report.detail = acquisition.detail
        report.pages = acquisition.page_count
        report.warnings = list(acquisition.warnings)
        report.duration_seconds = time.perf_counter() - started
        log.warning(
            "source_not_ingested",
            locator=report.locator,
            status=report.status.value,
            detail=report.detail,
        )
        return report
