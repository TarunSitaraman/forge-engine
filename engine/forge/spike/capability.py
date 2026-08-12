"""Local-model capability spike.

Purpose, stated plainly: find out whether the local model available on the
development machine can produce **reliable structured output**, how slow it is,
and how it fails. It is not a benchmark and not a prompt-tuning exercise.

Contradiction detection is deliberately **not** included — it is not a required
Phase 1 capability, and testing the hardest task on an unvalidated setup would
produce a discouraging number that means nothing.

Inputs come from the real corpus, not synthetic text, so results reflect the
material Forge will actually process.

Results are written to ``docs/research/local-model-capability-spike.md``. If the
model performs poorly, that is what gets recorded. Nothing here retries until
it looks good, and nothing falls back to heuristics to disguise a failure.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel

from ..llm.base import (
    CompletionRequest,
    LLMError,
    LLMProvider,
    Message,
    ProviderUnavailable,
    StructuredOutputError,
)
from ..logging import get_logger
from .schemas import ClaimExtraction, ConceptExtraction, RelationshipExtraction, SmallSynthesis

log = get_logger(__name__)

SYSTEM = (
    "You are a precise information-extraction component in a knowledge system. "
    "Return only JSON. Do not add commentary."
)


@dataclass
class TaskResult:
    task: str
    schema: str
    attempts: int = 0
    successes: int = 0
    latencies: list[float] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def median_latency(self) -> float | None:
        return round(statistics.median(self.latencies), 2) if self.latencies else None

    @property
    def max_latency(self) -> float | None:
        return round(max(self.latencies), 2) if self.latencies else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "schema": self.schema,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 3),
            "median_latency_s": self.median_latency,
            "max_latency_s": self.max_latency,
            "failures": self.failures,
            "samples": self.samples,
        }


@dataclass
class SpikeReport:
    provider: str
    model: str
    reachable: bool
    detail: str
    tasks: list[TaskResult] = field(default_factory=list)
    started_at: str = ""
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reachable": self.reachable,
            "detail": self.detail,
            "started_at": self.started_at,
            "total_seconds": round(self.total_seconds, 2),
            "tasks": [t.to_dict() for t in self.tasks],
            "overall_success_rate": round(self.overall_success_rate, 3),
        }

    @property
    def overall_success_rate(self) -> float:
        attempts = sum(t.attempts for t in self.tasks)
        successes = sum(t.successes for t in self.tasks)
        return successes / attempts if attempts else 0.0


# --------------------------------------------------------------------------
# Fixtures drawn from the real corpus
# --------------------------------------------------------------------------

DEFAULT_SAMPLES: dict[str, str] = {
    "concepts": (
        "Sliding Window maintains a contiguous range over a sequence and moves its "
        "boundaries instead of recomputing from scratch. It applies when a problem "
        "asks for the best or valid subarray under a constraint that changes "
        "monotonically as the window grows or shrinks. Compare with Two Pointers, "
        "which need not maintain a contiguous range."
    ),
    "claims": (
        "Retrieval-Augmented Generation reduces hallucination by grounding generation "
        "in retrieved passages. Chunk size materially affects retrieval quality: chunks "
        "that are too small lose context, and chunks that are too large dilute the "
        "embedding. Hybrid search combining BM25 with dense vectors outperforms either "
        "method alone on most benchmarks."
    ),
    "relationships": (
        "Dijkstra's algorithm requires a priority queue to select the next closest "
        "vertex efficiently. A priority queue is typically implemented as a binary heap. "
        "Dijkstra is a graph traversal algorithm and does not work with negative edge "
        "weights; Bellman-Ford handles those instead."
    ),
    "synthesis": (
        "Note A: Vector databases index embeddings for approximate nearest-neighbour "
        "search. Note B: At small corpus sizes, brute-force cosine similarity is fast "
        "enough and avoids operating a separate service. Note C: HNSW indexes trade "
        "recall for latency and are worth adopting once vector counts reach the "
        "hundreds of thousands."
    ),
}


def _prompt(task: str, text: str) -> list[Message]:
    instructions = {
        "concepts": "Extract the distinct technical concepts named in this text.",
        "claims": "Extract the individual factual assertions this text makes. Quote the supporting text verbatim.",
        "relationships": "Extract typed relationships between the technical entities in this text.",
        "synthesis": "Synthesize these notes into a short summary with key points.",
    }[task]
    return [
        Message(role="system", content=SYSTEM),
        Message(role="user", content=f"{instructions}\n\n---\n{text}\n---"),
    ]


TASKS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("structured_concept_extraction", "concepts", ConceptExtraction),
    ("simple_claim_extraction", "claims", ClaimExtraction),
    ("relationship_extraction", "relationships", RelationshipExtraction),
    ("small_synthesis", "synthesis", SmallSynthesis),
)


def run_spike(
    provider: LLMProvider,
    *,
    model_role: str = "extraction",
    repetitions: int = 3,
    samples: dict[str, str] | None = None,
) -> SpikeReport:
    """Run the four capability tasks and report honestly.

    ``repetitions`` matters: a single success proves nothing about reliability,
    and structured-output reliability is precisely what is being measured.
    """
    texts = samples or DEFAULT_SAMPLES
    reachable, detail = provider.health()
    model = _model_name(provider, model_role)

    report = SpikeReport(
        provider=provider.capabilities.name,
        model=model,
        reachable=reachable,
        detail=detail,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if not reachable:
        log.warning("spike_provider_unreachable", detail=detail)
        return report

    started = time.perf_counter()
    for task_name, sample_key, schema in TASKS:
        result = TaskResult(task=task_name, schema=schema.__name__)
        request = CompletionRequest(
            messages=_prompt(sample_key, texts[sample_key]),
            model_role=model_role,
            temperature=0.0,
        )

        for attempt in range(repetitions):
            result.attempts += 1
            call_started = time.perf_counter()
            try:
                parsed = provider.structured(request, schema)  # type: ignore[type-var]
                result.latencies.append(time.perf_counter() - call_started)
                result.successes += 1
                if attempt == 0:
                    result.samples.append(parsed.model_dump())
            except StructuredOutputError as exc:
                result.latencies.append(time.perf_counter() - call_started)
                result.failures.append(
                    {
                        "kind": "schema_violation",
                        "error": str(exc)[:300],
                        "raw_excerpt": exc.raw[:300],
                    }
                )
            except ProviderUnavailable as exc:
                result.failures.append({"kind": "provider_unavailable", "error": str(exc)[:300]})
                break
            except LLMError as exc:
                result.failures.append({"kind": "llm_error", "error": str(exc)[:300]})

        report.tasks.append(result)
        log.info(
            "spike_task_complete",
            task=task_name,
            success_rate=round(result.success_rate, 2),
            median_latency=result.median_latency,
        )

    report.total_seconds = time.perf_counter() - started
    return report


def _model_name(provider: LLMProvider, role: str) -> str:
    resolve = getattr(provider, "resolve_model", None)
    if callable(resolve):
        try:
            return str(resolve(role))
        except Exception:  # pragma: no cover - unconfigured role
            pass
    caps = provider.capabilities
    return caps.available_models[0] if caps.available_models else caps.name


def render_markdown(report: SpikeReport, *, notes: Sequence[str] = ()) -> str:
    """Render the spike report as the Markdown document required by Phase 1."""
    lines: list[str] = [
        "# Local Model Capability Spike",
        "",
        "*Generated by `forge model-test`. Records measured behaviour of the local "
        "model available on this machine — including failures.*",
        "",
        f"- **Provider:** {report.provider}",
        f"- **Model:** {report.model}",
        f"- **Reachable:** {'yes' if report.reachable else '**no**'}",
        f"- **Detail:** {report.detail}",
        f"- **Run at:** {report.started_at}",
        f"- **Total wall time:** {report.total_seconds:.2f}s",
        "",
    ]

    if not report.reachable:
        lines += [
            "## Result: NOT RUN — no local model reachable",
            "",
            "No capability results exist. The tasks below were **not executed**, so "
            "nothing about this model's structured-output reliability, latency, or "
            "failure modes has been established.",
            "",
            "This is recorded as an absence of evidence, not as a negative result.",
            "",
        ]
    else:
        lines += [
            "## Results",
            "",
            "| Task | Schema | Attempts | Success | Median latency | Max latency |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for t in report.tasks:
            lines.append(
                f"| {t.task} | `{t.schema}` | {t.attempts} | "
                f"{t.successes}/{t.attempts} ({t.success_rate:.0%}) | "
                f"{t.median_latency if t.median_latency is not None else '—'}s | "
                f"{t.max_latency if t.max_latency is not None else '—'}s |"
            )
        lines += ["", f"**Overall structured-output success rate: {report.overall_success_rate:.0%}**", ""]

        failures = [(t.task, f) for t in report.tasks for f in t.failures]
        lines += ["## Failure modes", ""]
        if not failures:
            lines.append("No failures observed in this run.")
        else:
            for task, f in failures[:20]:
                lines += [
                    f"- **{task}** — `{f.get('kind')}`: {f.get('error', '')}",
                ]
                if f.get("raw_excerpt"):
                    lines += ["", "  ```", f"  {f['raw_excerpt']}", "  ```"]
        lines.append("")

    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]

    lines += [
        "## How to reproduce",
        "",
        "```bash",
        "# 1. Install and start Ollama (https://ollama.com)",
        "ollama serve",
        "# 2. Pull a model",
        "ollama pull llama3.1:8b",
        "# 3. Point Forge at it and run the spike",
        "export FORGE_MODEL_DEFAULT=llama3.1:8b",
        "forge model-test --repetitions 3",
        "```",
        "",
        "The spike writes this file. Re-running it overwrites the results above.",
        "",
    ]
    return "\n".join(lines)
