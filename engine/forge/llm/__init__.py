"""LLM provider abstraction and implementations.

Import :func:`get_provider`, never a concrete provider class.
"""

from __future__ import annotations

from ..config import Settings
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
from .mock import MockProvider, fixture_provider, malformed_provider, unavailable_provider
from .ollama import OllamaProvider


def get_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider.

    The only place in the engine that knows which providers exist.
    """
    if settings.llm.provider == "mock":
        return MockProvider()
    if settings.llm.provider == "ollama":
        return OllamaProvider(
            settings.llm.base_url,
            models=dict(settings.llm.models),
            timeout=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )
    raise LLMError(f"unknown provider {settings.llm.provider!r}")


__all__ = [
    "get_provider",
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
