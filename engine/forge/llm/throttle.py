"""A provider wrapper that paces outbound calls.

**Why proactive pacing, when `CloudProvider` already retries on 429.** That
backoff is *reactive*: it starts only after the limit has been hit, and a host
that is already unhappy answers the retry with another 429. Measured on Groq,
2026-09-06: an assessment run with `FORGE_LLM_MAX_RETRIES=5` spent forty
minutes in 60-second backoffs and finished its third repetition scoring 1 of
21, every case a `retryable_failure`. Raising retries made it worse, because
more retries means more requests into a limiter that is counting requests.

Spacing calls out costs a known amount of wall clock and avoids the limit
instead of arguing with it.

**The unit is the call, not the case.** One page of the concept-extraction eval
is up to six calls, so pacing per case would leave five of them unpaced and
back-to-back, which is exactly the burst a limiter sees.

**A minimum interval, not a fixed sleep.** The interval is measured between
call *starts*, so a call that itself took twelve seconds satisfies a
five-second interval on its own and waits not at all. A blanket
`sleep(n)` after every call would add `n` to runs that were never near the
limit.

**What this does not pace.** A provider's own internal retries: `structured`
repairing a malformed response, or `complete` backing off a 429, are made
inside the wrapped provider and never pass back through here. The throttle
governs the calls its caller makes, which is the burst worth flattening.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .base import CompletionRequest, CompletionResponse, ProviderCapabilities

T = TypeVar("T", bound=BaseModel)


class ThrottledProvider:
    """Wraps any provider, enforcing a minimum interval between call starts.

    Everything not overridden here is delegated, so `resolve_model`, which
    `provider_identity` looks for, and any other provider-specific attribute
    still reach the wrapped object. That matters: a wrapper that quietly hid
    `resolve_model` would make every derivation key record the model as
    "unknown".
    """

    def __init__(
        self,
        inner: Any,
        min_interval_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must not be negative")
        self.inner = inner
        self.min_interval_seconds = float(min_interval_seconds)
        self._sleep = sleep
        self._clock = clock
        self._last_start: float | None = None
        #: Total wall clock spent waiting. Callers that report latency should
        #: subtract this, or the throttle shows up as the model being slow.
        self.slept_seconds = 0.0
        self.waits = 0

    # -- the protocol ------------------------------------------------------

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.inner.capabilities

    def health(self) -> tuple[bool, str]:
        """Not throttled. A health probe is one call before any work starts."""
        return self.inner.health()

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._wait()
        return self.inner.complete(request)

    def structured(self, request: CompletionRequest, schema: type[T]) -> T:
        self._wait()
        return self.inner.structured(request, schema)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define, so the
        # methods above are never shadowed by the wrapped provider's.
        return getattr(self.inner, name)

    # -- pacing ------------------------------------------------------------

    def _wait(self) -> None:
        """Sleep only for the part of the interval that has not already passed."""
        now = self._clock()
        if self._last_start is not None:
            remaining = self.min_interval_seconds - (now - self._last_start)
            if remaining > 0:
                self._sleep(remaining)
                self.slept_seconds += remaining
                self.waits += 1
                now = self._clock()
        self._last_start = now


def throttled(provider: Any, min_interval_seconds: float) -> Any:
    """Wrap `provider` if an interval was asked for, else hand it back.

    Returning the provider unchanged at zero keeps the un-throttled path
    exactly what it was, rather than a wrapper that happens to sleep for zero.
    """
    if min_interval_seconds <= 0:
        return provider
    return ThrottledProvider(provider, min_interval_seconds)
