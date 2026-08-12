"""Deterministic mock provider.

This is the **CI default**. The entire engine must be testable offline with no
model installed; if the test suite needed Ollama it would be skipped in CI and
would rot.

The mock is deliberately *not* clever. It returns fixture responses keyed by a
hash of the request, so tests are reproducible, and it can be told to fail in
specific ways so that error paths are exercised rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel

from .base import (
    CALLS,
    CompletionRequest,
    CompletionResponse,
    ProviderCapabilities,
    ProviderUnavailable,
    StructuredOutputError,
    parse_structured,
)

T = TypeVar("T", bound=BaseModel)


class MockProvider:
    """Deterministic, offline provider."""

    MODEL_NAME = "mock-1"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default_response: str = "{}",
        fail_with: Exception | None = None,
        responder: Callable[[CompletionRequest], str] | None = None,
        latency: float = 0.0,
    ) -> None:
        self.responses = responses or {}
        self.default_response = default_response
        self.fail_with = fail_with
        self.responder = responder
        self.latency = latency
        self.requests: list[CompletionRequest] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="mock",
            supports_structured_output=True,
            supports_streaming=False,
            context_length=8192,
            available_models=(self.MODEL_NAME,),
        )

    def health(self) -> tuple[bool, str]:
        return True, "mock provider always available"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        CALLS.record(self.MODEL_NAME)

        if self.fail_with is not None:
            raise self.fail_with

        if self.responder is not None:
            text = self.responder(request)
        else:
            text = self.responses.get(request_key(request), self.default_response)

        return CompletionResponse(
            text=text,
            model=self.MODEL_NAME,
            latency_seconds=self.latency,
            prompt_tokens=_approx_tokens(request),
            completion_tokens=len(text) // 4,
        )

    def structured(self, request: CompletionRequest, schema: type[T]) -> T:
        response = self.complete(request)
        return parse_structured(response.text, schema)


def request_key(request: CompletionRequest) -> str:
    """Stable key for a request, so fixtures can be looked up reproducibly."""
    payload = json.dumps(
        {
            "role": request.model_role,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fixture_provider(pairs: Sequence[tuple[CompletionRequest, str]]) -> MockProvider:
    """Build a MockProvider from (request, response) pairs."""
    return MockProvider({request_key(req): resp for req, resp in pairs})


def unavailable_provider() -> MockProvider:
    """A provider that behaves as if no local model is running."""
    return MockProvider(fail_with=ProviderUnavailable("mock: no model running"))


def malformed_provider(text: str = "here you go: {oops") -> MockProvider:
    """A provider that returns unparseable output, for error-path tests."""
    return MockProvider(default_response=text)


def _approx_tokens(request: CompletionRequest) -> int:
    return sum(len(m.content) for m in request.messages) // 4


__all__ = [
    "MockProvider",
    "request_key",
    "fixture_provider",
    "unavailable_provider",
    "malformed_provider",
    "StructuredOutputError",
    "Any",
]
