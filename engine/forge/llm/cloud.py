"""Cloud provider — portable inference for machines that cannot host a model.

**Why this exists.** Forge's default deployment is local-first: Ollama on a box
with a GPU, no account, no bill. That stays true. But the primary workstation
this is developed on is an 8 GB Intel MacBook, which cannot practically run an
8B model, and the GPU box is not always on. Without a portable provider, Forge's
semantic features would simply be unavailable on the machine it is used from
most.

**What this is not.** It is not a vendor integration that leaks upward. Nothing
above :mod:`forge.llm` imports this module or branches on the vendor: the
knowledge model, the workflow, the proposals, and the activation logic are all
identical whichever provider answered. Swapping providers is configuration.

**Credentials.** The API key is read from an environment variable named in
configuration, at call time. It is never stored in config, written to the
database, attached to provenance, or logged. :meth:`health` reports whether a
key is present without ever revealing it.

Provenance records the *provider and model identity*, because an assessment
made by a 3B local model and one made by a frontier model are not
interchangeable and must never be silently compared.
"""

from __future__ import annotations

import json
import os
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

#: Wire formats this provider can speak. Adding one is a change here and
#: nowhere else.
ANTHROPIC = "anthropic"
OPENAI_COMPATIBLE = "openai"

SUPPORTED_VENDORS = (ANTHROPIC, OPENAI_COMPATIBLE)


class CloudProvider:
    """Hosted inference behind the standard :class:`~forge.llm.base.LLMProvider`.

    Two wire formats are supported because "OpenAI-compatible" is the de-facto
    shape for most hosted and self-hosted gateways, and Anthropic's differs.
    Both are request/response translation only — no vendor-specific behaviour
    reaches the caller.
    """

    def __init__(
        self,
        *,
        vendor: str = ANTHROPIC,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        max_retries: int = 2,
        # Sized for a hosted model that thinks by default. On current Anthropic
        # models `max_tokens` caps thinking *plus* response text together, so
        # the old 2048 could be consumed by reasoning and truncate the JSON
        # mid-object — which Forge would surface as a structured-output failure
        # rather than as the budget problem it actually is.
        max_tokens: int = 16000,
        supports_structured_output: bool = True,
        models: dict[str, str] | None = None,
    ) -> None:
        if vendor not in SUPPORTED_VENDORS:
            raise LLMError(
                f"unsupported cloud vendor {vendor!r}; supported: {list(SUPPORTED_VENDORS)}"
            )
        self.vendor = vendor
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._structured = supports_structured_output
        #: Role -> model. Roles may bind different models; unbound roles use
        #: the default model rather than failing, because a cloud deployment
        #: usually wants one model for everything.
        self.models = models or {}
        self._client: httpx.Client | None = None

    # -- credentials -------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """The key, read fresh from the environment. Never cached to disk."""
        key = os.environ.get(self.api_key_env)
        return key.strip() if key and key.strip() else None

    def _require_key(self) -> str:
        key = self.api_key
        if not key:
            raise ProviderUnavailable(
                f"cloud provider {self.vendor!r} has no credential: environment "
                f"variable {self.api_key_env} is unset. Export it, or select a "
                f"different provider with FORGE_LLM_PROVIDER."
            )
        return key

    # -- introspection -----------------------------------------------------

    def resolve_model(self, role: str) -> str:
        return self.models.get(role) or self.model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=f"cloud:{self.vendor}",
            supports_structured_output=self._structured,
            supports_streaming=False,  # not used by Forge; nothing streams yet
            context_length=None,
            available_models=(self.model,),
        )

    def health(self) -> tuple[bool, str]:
        """Never raises, and never prints the key."""
        if not self.api_key:
            return False, (
                f"cloud provider {self.vendor!r} unavailable: {self.api_key_env} is not set"
            )
        return True, (
            f"cloud provider {self.vendor!r} configured with model {self.model!r} "
            f"(credential present in {self.api_key_env})"
        )

    # -- inference ---------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        key = self._require_key()
        model = self.resolve_model(request.model_role)
        path, payload, headers = self._build(request, model, key)

        started = timed()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http().post(path, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                CALLS.record(model)
                return CompletionResponse(
                    text=self._text_of(data),
                    model=model,
                    latency_seconds=timed() - started,
                    prompt_tokens=self._tokens(data, "input"),
                    completion_tokens=self._tokens(data, "output"),
                    raw=data,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    # A bad credential is not a transient failure and must not
                    # be retried into a rate limit.
                    raise ProviderUnavailable(
                        f"cloud provider rejected the credential in {self.api_key_env} "
                        f"({status})"
                    ) from exc
                if status < 500 and status != 429:
                    raise LLMError(
                        f"cloud provider rejected the request ({status}): "
                        f"{exc.response.text[:200]}"
                    ) from exc
                last_error = exc
                log.warning("cloud_retry", attempt=attempt, status=status, vendor=self.vendor)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise ProviderUnavailable(
                    f"cannot reach cloud provider at {self.base_url}: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("cloud_retry", attempt=attempt, error=str(exc)[:120])

        raise LLMError(
            f"cloud call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def structured(self, request: CompletionRequest, schema: type[T]) -> T:
        """Schema-validated JSON, with one bounded repair attempt.

        Identical contract to the Ollama provider: a response that will not
        validate raises rather than becoming a degraded write. Hosted models
        fail this less often than local ones, which is a reason to keep the
        check, not to drop it.
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
            json_schema=schema_json if self._structured else None,
        )

        response = self.complete(req)
        try:
            return parse_structured(response.text, schema, attempts=1)
        except StructuredOutputError as first:
            log.warning("structured_output_repair", provider=self.vendor, error=str(first)[:160])
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
                json_schema=req.json_schema,
            )
            retry = self.complete(repair)
            return parse_structured(retry.text, schema, attempts=2)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- wire formats ------------------------------------------------------

    def _build(
        self, request: CompletionRequest, model: str, key: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        max_tokens = request.max_tokens or self.max_tokens

        if self.vendor == ANTHROPIC:
            # `temperature` is deliberately **not** sent. Current Anthropic
            # models reject non-default sampling parameters: `temperature`,
            # `top_p`, and `top_k` return 400 on Opus 4.7 and later, and on
            # Sonnet 5 any non-default value does the same. Forge asks for
            # `temperature=0.0` everywhere for determinism, which is exactly the
            # non-default value that fails — so sending it would have made
            # *every* cloud call a 400. It is dropped rather than clamped
            # because 0.0 never guaranteed identical outputs anyway;
            # determinism here comes from the schema and the grounding check,
            # not from a sampling knob. The OpenAI-compatible path below still
            # sends it — those gateways accept it.
            payload: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": turns or [{"role": "user", "content": system}],
            }
            if system and turns:
                payload["system"] = system
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            return "/v1/messages", payload, headers

        # OpenAI-compatible: system stays in the message list.
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.json_schema is not None and self._structured:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        return "/v1/chat/completions", payload, headers

    def _text_of(self, data: dict[str, Any]) -> str:
        if self.vendor == ANTHROPIC:
            blocks = data.get("content") or []
            return "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            )
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""

    def _tokens(self, data: dict[str, Any], which: str) -> int | None:
        usage = data.get("usage") or {}
        if self.vendor == ANTHROPIC:
            return usage.get(f"{which}_tokens")
        return usage.get("prompt_tokens" if which == "input" else "completion_tokens")
