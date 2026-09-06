"""Extraction quality measured against the vault's own structure.

**Why a second extraction eval exists.** `extraction.py` scores a labelled set
of six hand-written cases. That set is small, it is hand-built, and every
expected concept in it was chosen by the person who also wrote the prompt.
This module scores extraction against 545 concepts nobody labelled for the
purpose: the vault's own page names, which `forge bootstrap` derives from the
directory listing with zero model calls. A human decided `Binary Search`
deserved one canonical page and created it. That decision is the exact
judgement extraction is trying to reproduce, and it is already written down.

**The headline is not recall, for the same reason as in `extraction.py`.** An
extractor emitting every noun phrase on the page scores perfectly on recall and
is still the failure mode this corpus actually had. Three numbers are reported
and none of them is recall over the whole vocabulary:

* **Self-recovery.** For one page, does extraction over that page's own text
  recover that page's own concept? This is the only recall-shaped question the
  vault can answer honestly. Asking "how many of the 545 did it find" is
  meaningless per page — a page about B-trees is not supposed to mention 544
  other concepts, so a miss would not be a miss.
* **Junk rate.** Share of emitted names that appear on the forbidden list the
  labelled set already carries (`RAM`, `Answer`, `maxmemory`, `VARCHAR(n)` and
  twenty-one others actually observed coming out of this corpus). Same
  definition as `ExtractionReport.junk_rate`, so the two evals' junk numbers
  are comparable.
* **Off-vocabulary rate.** Share of emitted names matching no page in the
  vault. Reported, and deliberately **not** called junk: a genuinely new
  concept the vault has no page for lands here, and that is extraction working,
  not failing. It is a shape measurement, useful next to junk rate, quotable as
  neither quality nor its absence.

**A page whose calls did not all return is not scored**, on the rule
`extraction.py` learned the hard way: a timeout returns a truncated result
rather than raising, and a page that emitted nothing emits no junk either, so
folding it in makes a broken run look clean.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..extraction.extractor import _grounded
from ..parsing.links import normalize


@dataclass(frozen=True)
class VaultPage:
    """One page of the vault, and the concept its existence asserts.

    ``targets`` is normally the filename stem, because that is what
    `forge bootstrap` turns into a concept. The page's H1 title is accepted
    too: `Technologies/Docs/rag.md` is titled "RAG (Retrieval-Augmented
    Generation)", and an extractor emitting `RAG` has recovered that page's
    concept by any reading. Both are recorded so a hit says which one matched.
    """

    path: str
    canonical_name: str
    title: str = ""
    text: str = ""

    @property
    def targets(self) -> tuple[str, ...]:
        names = [self.canonical_name]
        if self.title and normalize(self.title) != normalize(self.canonical_name):
            names.append(self.title)
        return tuple(names)


@dataclass
class Vocabulary:
    """The vault's concept names, normalized for comparison.

    Built from a `SeedPlan`, so it is exactly the set `forge bootstrap`
    would write to the graph — including its exclusions. Navigation pages,
    numbered knowledge-pack chapters and status artifacts are already gone,
    which matters here: were `_index` in the vocabulary, an extractor emitting
    it would score as on-vocabulary rather than as the table-of-contents
    mistake it is.

    It is smaller than the concept count, and that is not a bug. A decided
    collision produces two namespaced concepts sharing one bare name — this
    vault has four (`Heap`, `Binary Search`, `Trie`, `weekly-review`), so 545
    concepts give 541 names. The namespace is disambiguation a human recorded
    in `concept-identity.yaml`; the extractor is never asked to produce it and
    must not be scored on it. Emitting `Heap` therefore counts as recovering
    either Heap page, which is the right answer to the question being asked.
    """

    names: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return normalize(name) in self.names

    @classmethod
    def from_concepts(cls, concepts: Iterable[Any]) -> Vocabulary:
        return cls({normalize(c.canonical_name): c.canonical_name for c in concepts})


@dataclass
class PageScore:
    """One page, scored. Names every string, so a number can be checked."""

    path: str
    #: The page's own concept, as bootstrap derived it.
    canonical_name: str
    #: Which of the page's target names extraction emitted, if any.
    recovered_as: str | None = None
    emitted: list[str] = field(default_factory=list)
    junk: list[str] = field(default_factory=list)
    #: Emitted names that match some *other* vault page. Not junk and not
    #: recovery: a page about RAG mentioning `Vector Databases` is correct.
    in_vocabulary: list[str] = field(default_factory=list)
    off_vocabulary: list[str] = field(default_factory=list)
    claims: int = 0
    grounded_claims: int = 0
    #: Claims dropped before return for failing the same grounding check this
    #: module applies. Without them the denominator is survivors only and the
    #: rate is 1.000 for every model ever run.
    dropped_claims: int = 0
    llm_calls: int = 0
    spans: int = 0
    error: str | None = None
    status: str = "succeeded"
    failures: list[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return self.recovered_as is not None

    @property
    def complete(self) -> bool:
        """Did every call for this page return? Only then is the score real."""
        return self.status == "succeeded" and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "canonical_name": self.canonical_name,
            "recovered": self.recovered,
            "recovered_as": self.recovered_as,
            "emitted": self.emitted,
            "junk": self.junk,
            "in_vocabulary": self.in_vocabulary,
            "off_vocabulary": self.off_vocabulary,
            "claims": self.claims,
            "grounded_claims": self.grounded_claims,
            "dropped_claims": self.dropped_claims,
            "llm_calls": self.llm_calls,
            "spans": self.spans,
            "error": self.error,
            "status": self.status,
            "complete": self.complete,
            "failures": self.failures,
        }


@dataclass
class CorpusExtractionReport:
    model_id: str
    prompt_version: str
    vocabulary_size: int = 0
    sampled: int = 0
    population: int = 0
    seed: int | None = None
    scores: list[PageScore] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def complete(self) -> list[PageScore]:
        return [s for s in self.scores if s.complete]

    @property
    def failed(self) -> list[PageScore]:
        return [s for s in self.scores if not s.complete]

    @property
    def trustworthy(self) -> bool:
        """No rate below is quotable as a model property unless this holds."""
        return not self.failed and bool(self.scores)

    @property
    def self_recovery(self) -> float:
        """Share of pages whose own concept extraction found in their own text."""
        scored = self.complete
        return sum(s.recovered for s in scored) / len(scored) if scored else 0.0

    @property
    def emitted_total(self) -> int:
        return sum(len(s.emitted) for s in self.complete)

    @property
    def junk_rate(self) -> float:
        """Share of emitted names on the observed-junk list. The headline."""
        return sum(len(s.junk) for s in self.complete) / self.emitted_total if self.emitted_total else 0.0

    @property
    def off_vocabulary_rate(self) -> float:
        """Share of emitted names matching no vault page. Shape, not quality."""
        off = sum(len(s.off_vocabulary) for s in self.complete)
        return off / self.emitted_total if self.emitted_total else 0.0

    @property
    def concepts_per_page(self) -> float:
        scored = self.complete
        return self.emitted_total / len(scored) if scored else 0.0

    @property
    def grounding_rate(self) -> float:
        kept = sum(s.claims for s in self.complete)
        dropped = sum(s.dropped_claims for s in self.complete)
        grounded = sum(s.grounded_claims for s in self.complete)
        total = kept + dropped
        return grounded / total if total else 1.0

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "vocabulary_size": self.vocabulary_size,
            "population": self.population,
            "sampled": self.sampled,
            "seed": self.seed,
            "pages": len(self.scores),
            "scored": len(self.complete),
            "failed": len(self.failed),
            "trustworthy": self.trustworthy,
            "self_recovery": round(self.self_recovery, 3),
            "junk_rate": round(self.junk_rate, 3),
            "off_vocabulary_rate": round(self.off_vocabulary_rate, 3),
            "concepts_per_page": round(self.concepts_per_page, 2),
            "grounding_rate": round(self.grounding_rate, 3),
            "emitted_total": self.emitted_total,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 2),
            "scores": [s.to_dict() for s in self.scores],
        }

    def summary_line(self) -> str:
        scope = (
            f"{len(self.complete)}/{len(self.scores)} pages"
            if not self.trustworthy
            else f"{len(self.scores)} pages"
        )
        return (
            f"{self.model_id}  prompt={self.prompt_version}  "
            f"self-recovery={self.self_recovery:.2f}  junk={self.junk_rate:.2f}  "
            f"off-vocab={self.off_vocabulary_rate:.2f}  "
            f"grounded={self.grounding_rate:.2f}  calls={self.llm_calls}  {scope}"
        )


def score_page(
    page: VaultPage,
    concepts: Sequence[str],
    vocabulary: Vocabulary,
    forbidden: Sequence[str] = (),
    claims: Sequence[tuple[str, str]] = (),
    *,
    llm_calls: int = 0,
    spans: int = 0,
    error: str | None = None,
    status: str = "succeeded",
    failures: Sequence[str] = (),
    dropped_claims: int = 0,
) -> PageScore:
    """Score one page's emitted concepts against the vault vocabulary.

    Matching is through `normalize()` — the same comparison the link resolver,
    the identity config and `extraction.score_case` use — so `B-tree index`,
    `B Tree Index` and `b-tree-index` are one concept rather than a miss plus
    two false positives.

    The three buckets are exclusive and ordered: junk first, because a string
    on the forbidden list stays junk even if the vault happens to have a page
    with that name; then vocabulary; then everything else.
    """
    emitted = {normalize(c): c.strip() for c in concepts if c and c.strip()}
    forbidden_keys = {normalize(f) for f in forbidden}
    targets = {normalize(t): t for t in page.targets}

    recovered_as = next((targets[k] for k in targets if k in emitted), None)
    junk = sorted(emitted[k] for k in emitted if k in forbidden_keys)
    known = sorted(
        emitted[k] for k in emitted if k not in forbidden_keys and k in vocabulary.names
    )
    off = sorted(
        emitted[k] for k in emitted if k not in forbidden_keys and k not in vocabulary.names
    )

    grounded = sum(1 for _, quote in claims if _grounded(quote, page.text))
    return PageScore(
        path=page.path,
        canonical_name=page.canonical_name,
        recovered_as=recovered_as,
        emitted=sorted(emitted.values()),
        junk=junk,
        in_vocabulary=known,
        off_vocabulary=off,
        claims=len(claims),
        grounded_claims=grounded,
        dropped_claims=dropped_claims,
        llm_calls=llm_calls,
        spans=spans,
        error=error,
        status=status,
        failures=list(failures),
    )


def _failure_lines(failures: Sequence[dict]) -> list[str]:
    """Deduplicate failures to `kind: message`, keeping the message.

    Twelve spans failing on one rejected model name is one fact, not twelve,
    and dropping the message leaves `llm_error` — the layer that caught the
    error, and nothing about the error.
    """
    seen: dict[str, None] = {}
    for failure in failures:
        kind = str(failure.get("kind", "unknown"))
        message = str(failure.get("error", "")).strip()
        seen.setdefault(f"{kind}: {message}" if message else kind, None)
    return list(seen)


def run(
    pages: Sequence[VaultPage],
    extractor: Any,
    vocabulary: Vocabulary,
    spans_for: Any,
    *,
    forbidden: Sequence[str] = (),
    on_page: Any = None,
) -> CorpusExtractionReport:
    """Drive the production extractor over real vault pages, one page at a time.

    Uses `CandidateExtractor` rather than re-implementing the prompt path, so
    what is measured is what ships — the same reason `extraction.run` and the
    assessment eval drive the real objects.

    ``spans_for`` maps a page to its spans. It is injected rather than built
    here because span construction is the corpus indexer's job and needs the
    vault on disk, which the scoring logic must not require in order to be
    testable.
    """
    import time

    report = CorpusExtractionReport(
        model_id=extractor.model_id(),
        prompt_version=extractor.prompt_version,
        vocabulary_size=len(vocabulary),
        sampled=len(pages),
    )
    started = time.perf_counter()

    for page in pages:
        try:
            spans = list(spans_for(page))
            result = extractor.extract(spans)
        except Exception as exc:  # noqa: BLE001 - a provider failure is a result, not a crash
            score = score_page(
                page,
                [],
                vocabulary,
                forbidden,
                error=f"{type(exc).__name__}: {exc}"[:200],
                status="failed",
            )
            report.scores.append(score)
            if on_page is not None:
                on_page(score, len(report.scores))
            continue

        # A timeout does not raise: `extract` catches it per call and returns a
        # PARTIAL result with fewer concepts. Reading only the exception path
        # scores a truncated page as a clean one.
        score = score_page(
            page,
            [c.name for c in result.concepts],
            vocabulary,
            forbidden,
            [(c.statement, c.evidence_quote) for c in result.claims],
            llm_calls=result.llm_calls,
            spans=len(spans),
            status=result.status.value,
            failures=_failure_lines(result.failures),
            dropped_claims=sum(
                1 for f in result.failures if f.get("kind") == "ungrounded_quote"
            ),
        )
        report.scores.append(score)
        if on_page is not None:
            on_page(score, len(report.scores))

    report.duration_seconds = time.perf_counter() - started
    return report
