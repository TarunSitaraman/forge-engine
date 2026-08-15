"""Ollama provider — the local-first default.

Ollama is reached over plain HTTP on localhost. No API key, no account, no
paid service. This is what satisfies "the engine must run without a paid API".

Structured output uses Ollama's ``format`` parameter when a JSON schema is
supplied, and falls back to prompt-level instruction plus deterministic JSON
recovery when the model ignores it (local models frequently do).
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ..logging import get_logger
from .base import (
    CALLS,
    CompletionRequest,
    CompletionResponse,
    LLMError,
    Message,
    ProviderCapabilities,
    ProviderUnavailable,
    StructuredOutputError,
    parse_structured,
    timed,
)

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaProvider:
    """Local inference via the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        models: dict[str, str] | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        think: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = models or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.think = think
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._capabilities: ProviderCapabilities | None = None

    # -- introspection -----------------------------------------------------

    def list_models(self) -> tuple[str, ...]:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderUnavailable(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        return tuple(sorted(m.get("name", "") for m in data.get("models", []) if m.get("name")))

    @property
    def capabilities(self) -> ProviderCapabilities:
        if self._capabilities is None:
            try:
                available = self.list_models()
            except ProviderUnavailable:
                available = ()
            self._capabilities = ProviderCapabilities(
                name="ollama",
                supports_structured_output=True,
                supports_streaming=True,
                context_length=None,  # model-dependent; not advertised by /api/tags
                available_models=available,
            )
        return self._capabilities

    def health(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except ProviderUnavailable as exc:
            return False, str(exc)
        if not models:
            return False, f"Ollama is running at {self.base_url} but has no models pulled"
        return True, f"Ollama at {self.base_url} with {len(models)} model(s): {', '.join(models)}"

    def resolve_model(self, role: str) -> str:
        model = self.models.get(role)
        if not model:
            raise LLMError(f"no model configured for role {role!r}")
        return model

    @property
    def identity_variant(self) -> str:
        """Suffix distinguishing runs that are not comparable.

        A thinking model answering with reasoning disabled is, for caching
        purposes, a different instrument from the same model reasoning
        normally. Without this, a fast think-off run would silently serve
        cached results to a later think-on run and vice versa, and the two
        would be averaged together in any evaluation.
        """
        if self.think is False:
            return "+nothink"
        if self.think is True:
            return "+think"
        return ""

    # -- inference ---------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = self.resolve_model(request.model_role)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        if request.json_schema is not None:
            payload["format"] = request.json_schema
        if self.think is not None:
            payload["think"] = self.think

        started = timed()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                CALLS.record(model)
                return CompletionResponse(
                    text=data.get("message", {}).get("content", ""),
                    model=model,
                    latency_seconds=timed() - started,
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                    raw=data,
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                # Ollama rejects `think` outright for models that have no
                # reasoning mode. Retrying without the field is not a silent
                # downgrade — there is no reasoning to lose on such a model —
                # but it is a departure from what was asked for, so it is said
                # out loud and done exactly once.
                if (
                    exc.response.status_code < 500
                    and "think" in payload
                    and "think" in exc.response.text.casefold()
                ):
                    log.warning(
                        "ollama_thinking_unsupported",
                        model=model,
                        requested_think=payload.pop("think"),
                        detail=exc.response.text[:200],
                    )
                    continue
                # Other 4xx is a request problem; retrying identical input won't help.
                if exc.response.status_code < 500:
                    raise LLMError(
                        f"Ollama rejected the request ({exc.response.status_code}): "
                        f"{exc.response.text[:200]}"
                    ) from exc
                log.warning("ollama_retry", attempt=attempt, status=exc.response.status_code)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise ProviderUnavailable(
                    f"cannot reach Ollama at {self.base_url}: {exc}. "
                    f"Is `ollama serve` running?"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("ollama_retry", attempt=attempt, error=str(exc))

        raise LLMError(f"Ollama call failed after {self.max_retries + 1} attempts: {last_error}")

    def structured(self, request: CompletionRequest, schema: type[T]) -> T:
        """Request schema-conforming JSON, with bounded repair retries.

        One repair attempt is made by feeding the model its own invalid output
        and the validation error. If that fails, the error is raised — a
        malformed response never becomes a degraded write.
        """
        schema_json = schema.model_json_schema()
        req = CompletionRequest(
            messages=[
                *request.messages,
                Message(
                    role="system",
                    content=(
                        "Respond with a single JSON object matching this schema. "
                        "No prose, no code fences.\n" + json.dumps(schema_json)
                    ),
                ),
            ],
            model_role=request.model_role,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_schema=schema_json,
        )

        response = self.complete(req)
        try:
            return parse_structured(response.text, schema, attempts=1)
        except StructuredOutputError as first:
            log.warning("structured_output_repair", error=str(first))
            repair = CompletionRequest(
                messages=[
                    *req.messages,
                    Message(role="assistant", content=response.text),
                    Message(
                        role="user",
                        content=(
                            f"That response was invalid: {first}. "
                            f"Return only corrected JSON matching the schema."
                        ),
                    ),
                ],
                model_role=request.model_role,
                temperature=request.temperature,
                json_schema=schema_json,
            )
            retry = self.complete(repair)
            return parse_structured(retry.text, schema, attempts=2)

    def close(self) -> None:
        self._client.close()
