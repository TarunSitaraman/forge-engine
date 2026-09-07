"""Proactive call pacing.

Written because reactive backoff was measured making things worse. On Groq,
2026-09-06, an assessment run with `FORGE_LLM_MAX_RETRIES=5` spent forty
minutes in 60-second backoffs and scored 1 of 21 on its third repetition,
every case a `retryable_failure`. More retries means more requests into a
limiter that is counting requests.

The interesting properties are all about *not* sleeping: a slow call should
satisfy the interval on its own, the first call should not wait, and zero
should leave the provider untouched rather than wrapping it in a no-op.
"""

from __future__ import annotations

import pytest
from forge.llm import MockProvider, provider_identity, throttled
from forge.llm.base import CompletionRequest, Message
from forge.llm.throttle import ThrottledProvider


class FakeClock:
    """A clock the test advances, including when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def request() -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content="hi")])


def build(interval: float = 5.0, *, latency: float = 0.0):
    clock = FakeClock()
    inner = MockProvider(default_response="{}")
    if latency:
        original = inner.complete

        def slow(req):
            clock.advance(latency)
            return original(req)

        inner.complete = slow  # type: ignore[method-assign]
    return ThrottledProvider(inner, interval, sleep=clock.sleep, clock=clock), clock


# -- when it does not sleep -----------------------------------------------


def test_the_first_call_never_waits():
    """There is nothing to pace against yet, and delaying the start is waste."""
    provider, clock = build()
    provider.complete(request())
    assert clock.sleeps == []
    assert provider.slept_seconds == 0.0


def test_a_call_slower_than_the_interval_satisfies_it_on_its_own():
    """The interval is between call *starts*, not a sleep bolted onto each call.

    This is the difference between pacing and just being slower. Groq calls in
    the runs that motivated this took 9 to 12 seconds each; a blanket
    `sleep(5)` would have added 5 to every one of them for nothing.
    """
    provider, clock = build(5.0, latency=12.0)
    provider.complete(request())
    provider.complete(request())
    assert clock.sleeps == []
    assert provider.waits == 0


def test_only_the_unelapsed_part_of_the_interval_is_slept():
    provider, clock = build(5.0, latency=2.0)
    provider.complete(request())
    provider.complete(request())
    assert clock.sleeps == [pytest.approx(3.0)]


def test_zero_hands_back_the_provider_itself():
    """Not a wrapper that sleeps for zero: the un-throttled path stays as it was."""
    inner = MockProvider(default_response="{}")
    assert throttled(inner, 0) is inner
    assert throttled(inner, 0.0) is inner


def test_a_negative_interval_is_refused():
    with pytest.raises(ValueError):
        ThrottledProvider(MockProvider(), -1)


# -- when it does -----------------------------------------------------------


def test_back_to_back_calls_are_spaced_by_the_interval():
    provider, clock = build(5.0)
    for _ in range(4):
        provider.complete(request())
    assert clock.sleeps == [5.0, 5.0, 5.0], "three gaps between four calls"
    assert provider.slept_seconds == pytest.approx(15.0)
    assert provider.waits == 3


def test_structured_calls_are_paced_too():
    """`structured` is the call the evals actually make."""
    from pydantic import BaseModel

    class Shape(BaseModel):
        pass

    provider, clock = build(5.0)
    provider.structured(request(), Shape)
    provider.structured(request(), Shape)
    assert clock.sleeps == [5.0]


def test_a_health_probe_is_not_paced():
    """One call before any work. Pacing it only delays reporting an outage."""
    provider, clock = build(5.0)
    provider.complete(request())
    provider.health()
    provider.health()
    assert clock.sleeps == []


# -- delegation -------------------------------------------------------------


def test_provider_identity_still_resolves_through_the_wrapper():
    """The failure this guards against is silent and poisons every record.

    `provider_identity` looks for `resolve_model`. A wrapper that did not
    delegate would return "unknown", and every derivation key written during a
    throttled run would record an unknown model, making its results
    uncacheable and unattributable.
    """

    class WithModel(MockProvider):
        def resolve_model(self, role: str) -> str:
            return f"model-for-{role}"

    provider = ThrottledProvider(WithModel(), 5.0)
    assert provider_identity(provider, "analysis") == ("mock", "model-for-analysis")


def test_capabilities_and_unknown_attributes_delegate():
    inner = MockProvider(default_response="{}")
    provider = ThrottledProvider(inner, 5.0)
    assert provider.capabilities.name == "mock"
    assert provider.requests is inner.requests  # provider-specific attribute


def test_the_wrappers_own_methods_are_not_shadowed_by_the_inner_provider():
    """`__getattr__` runs only for names this class does not define.

    If `complete` ever resolved through delegation the pacing would silently
    stop happening, and the run would look fine until the 429s came back.
    """
    provider, clock = build(5.0)
    provider.complete(request())
    provider.complete(request())
    assert clock.sleeps == [5.0]
