"""LLM provider abstraction.

Domain and corpus code never import a provider. They ask for a *role*
(``extraction``, ``analysis``, ``resolution``, ``synthesis``) and receive
whatever the configuration bound to it. Ollama is the default; nothing about
Ollama appears above this package.

Design rules:

* **No API keys in code.** Providers read credentials from configuration only.
* **Structured output is schema-validated**, with bounded repair retries and
  then a hard failure. A malformed model response must never become a
  silently-degraded write.
* **Every call is counted.** :class:`CallRecorder` makes "this operation made
  zero LLM calls" an assertable property rather than a claim.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for provider failures."""


class ProviderUnavailable(LLMError):
    """The provider could not be reached. Distinct from a bad response.

    Kept separate so the CLI and spike can report "no local model is running"
    differently from "the model answered badly" — they need different fixes.
    """


class StructuredOutputError(LLMError):
    """The model did not produce output matching the requested schema."""

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    messages: Sequence[Message]
    #: Logical role; the provider resolves it to a concrete model.
    model_role: str = "extraction"
    temperature: float = 0.0  # deterministic by default
    max_tokens: int | None = None
    #: JSON schema the response must satisfy, when structured output is used.
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompletionResponse:
    text: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    name: str
    supports_structured_output: bool
    supports_streaming: bool
    context_length: int | None = None
    available_models: tuple[str, ...] = ()


class CallRecorder:
    """Process-wide counter of LLM calls.

    Used by tests to assert that deterministic paths make no model calls, and
    by the CLI to report call counts per run.
    """

    def __init__(self) -> None:
        self.count = 0
        self.by_model: dict[str, int] = {}

    def record(self, model: str) -> None:
        self.count += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def reset(self) -> None:
        self.count = 0
        self.by_model.clear()


#: Shared recorder. Providers must record every outbound call here.
CALLS = CallRecorder()


@runtime_checkable
class LLMProvider(Protocol):
    """The only interface the rest of Forge may depend on."""

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    def structured(self, request: CompletionRequest, schema: type[T]) -> T: ...

    def health(self) -> tuple[bool, str]:
        """``(reachable, detail)`` — never raises. Used by ``forge status``."""
        ...


# --------------------------------------------------------------------------
# Shared helpers for providers that return JSON as text
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Recover a JSON document from model output.

    Local models very often wrap JSON in prose or code fences even when told
    not to. Recovering deterministically here is cheaper and more reliable than
    spending another model call asking for a correction — and, importantly,
    this is a parsing problem, so software solves it (Principle 7).
    """
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    if (m := _FENCE_RE.search(text)) is not None:
        return m.group(1).strip()
    # Fall back to the outermost balanced braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return stripped


def parse_structured(
    text: str, schema: type[T], *, attempts: int = 1
) -> T:
    """Validate model output against ``schema``, raising a typed error."""
    candidate = extract_json(text)
    try:
        return schema.model_validate_json(candidate)
    except ValidationError as exc:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as jexc:
            raise StructuredOutputError(
                f"response was not valid JSON: {jexc}", raw=text, attempts=attempts
            ) from jexc
        raise StructuredOutputError(
            f"response did not match schema {schema.__name__}: {_first_error(exc)}",
            raw=text,
            attempts=attempts,
        ) from exc
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"response was not valid JSON: {exc}", raw=text, attempts=attempts
        ) from exc


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:  # pragma: no cover
        return str(exc)
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    return f"{loc}: {first.get('msg')}"


def timed() -> float:
    return time.perf_counter()
