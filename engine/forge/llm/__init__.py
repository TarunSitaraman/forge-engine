"""LLM provider abstraction and implementations.

Import :func:`get_provider`, never a concrete provider class.
"""

from __future__ import annotations

from ..config import Settings
from ..logging import get_logger
from .base import (
    CALLS,
    CallRecorder,
    CompletionRequest,
    CompletionResponse,
    LLMError,
    LLMProvider,
    Message,
    ProviderCapabilities,
    ProviderUnavailable,
    StructuredOutputError,
    extract_json,
    parse_structured,
)
from .cloud import CloudProvider
from .mock import MockProvider, fixture_provider, malformed_provider, unavailable_provider
from .ollama import OllamaProvider

log = get_logger(__name__)


def get_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider, or its configured fallback.

    The only place in the engine that knows which providers exist. Selection is
    **explicit**: an unavailable provider is reported as unavailable, never
    quietly replaced by a different one. See :func:`require_provider`.

    `FORGE_LLM_FALLBACK` relaxes that, but narrowly and on purpose:

    * It is **opt-in**. With nothing set, behaviour is exactly as before.
    * Failover happens **once, before any work**, decided by a health check —
      never mid-run and never per call. Whichever provider is returned is the
      one that answers every call in that run, so the model identity recorded
      in each derivation key is the model that actually produced the result.
      A per-call fallback would break that and is deliberately not offered.
    * It triggers only on *unavailability*. A provider that answers badly is
      not a provider that is down, and swapping on bad output would silently
      mix two models' results under one run.
    * It is **loud**. The switch is logged at warning level and `forge status`
      reports it.
    """
    primary = _build(settings, settings.llm.provider)
    if settings.llm.fallback is None:
        return primary

    try:
        healthy, detail = primary.health()
    except Exception as exc:  # a provider that cannot even be probed is down
        healthy, detail = False, f"{type(exc).__name__}: {exc}"
    if healthy:
        return primary

    log.warning(
        "provider_fallback",
        requested=settings.llm.provider,
        using=settings.llm.fallback,
        reason=detail,
    )
    return _build(settings, settings.llm.fallback)


def _build(settings: Settings, provider: str) -> LLMProvider:
    """Construct one named provider. No fallback logic lives here."""
    if provider == "mock":
        return MockProvider()
    if provider == "ollama":
        # `llm.base_url` remains the Phase 1-3 spelling and stays authoritative
        # when it has been set away from the default; `llm.ollama.base_url` is
        # the preferred one and wins otherwise.
        default = type(settings.llm).model_fields["base_url"].default
        base_url = (
            settings.llm.base_url
            if settings.llm.base_url != default
            else settings.llm.ollama.base_url
        )
        return OllamaProvider(
            base_url,
            models=dict(settings.llm.models),
            timeout=settings.llm.ollama.timeout_seconds,
            max_retries=settings.llm.ollama.max_retries,
            think=settings.llm.ollama.think,
        )
    if provider == "cloud":
        cloud = settings.llm.cloud
        return CloudProvider(
            vendor=cloud.vendor,
            model=cloud.model,
            api_key_env=cloud.api_key_env,
            base_url=cloud.base_url,
            timeout=cloud.timeout_seconds,
            max_retries=cloud.max_retries,
            max_tokens=cloud.max_tokens,
            supports_structured_output=cloud.supports_structured_output,
        )
    raise LLMError(f"unknown provider {settings.llm.provider!r}")


def provider_identity(provider: LLMProvider, role: str = "analysis") -> tuple[str, str]:
    """``(provider_id, model_id)`` for provenance and derivation keys.

    Both halves matter. Two assessments produced by the same *model name* on
    different providers are still two different things, and an assessment must
    never be reused across a provider change just because the model string
    happened to match.
    """
    name = provider.capabilities.name
    resolve = getattr(provider, "resolve_model", None)
    try:
        model = resolve(role) if callable(resolve) else name
    except Exception:  # pragma: no cover - unbound role on a partial config
        model = "unknown"
    return name, model


def require_provider(settings: Settings, *, role: str = "analysis") -> LLMProvider:
    """Return the configured provider, or raise :class:`ProviderUnavailable`.

    This is the gate in front of every high-risk semantic operation. It exists
    to make one behaviour impossible: quietly answering a knowledge-mutation
    question with whatever model happens to be reachable. If the configured
    provider is not usable, the caller gets an explicit unavailability — never
    a substitute.
    """
    provider = get_provider(settings)
    reachable, detail = provider.health()
    if not reachable:
        raise ProviderUnavailable(detail)
    provider_id, model_id = provider_identity(provider, role)
    if not model_id or model_id == "unknown":
        raise ProviderUnavailable(
            f"provider {provider_id!r} has no model bound to role {role!r}"
        )
    return provider


__all__ = [
    "get_provider",
    "provider_identity",
    "require_provider",
    "CloudProvider",
    "LLMProvider",
    "LLMError",
    "ProviderUnavailable",
    "StructuredOutputError",
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "ProviderCapabilities",
    "MockProvider",
    "OllamaProvider",
    "fixture_provider",
    "unavailable_provider",
    "malformed_provider",
    "CALLS",
    "CallRecorder",
    "extract_json",
    "parse_structured",
]
