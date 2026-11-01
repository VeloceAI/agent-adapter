import asyncio

import pytest

from agent_adapter.core.errors import (
    ApprovalPending,
    ConfigError,
    ModelRateLimited,
    ToolArgsInvalid,
    ToolError,
    ToolTimeout,
    ToolUsageError,
)
from agent_adapter.exec.retry import Backoff, _attempt, is_retryable
from tests.fakes.clock import FakeClock


class GraphBubbleUp(Exception):
    pass


def test_backoff_grows_by_multiplier_and_caps():
    backoff = Backoff(base_s=1.0, max_s=5.0, multiplier=2.0, jitter=0.0)
    assert [backoff.delay(n, lambda: 0.5) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]


def test_backoff_jitter_scales_the_delay():
    backoff = Backoff(base_s=2.0, jitter=0.5)
    assert backoff.delay(1, lambda: 0.0) == 2.0
    assert backoff.delay(1, lambda: 1.0) == 3.0


def test_backoff_huge_attempt_does_not_overflow():
    assert (
        Backoff(base_s=1.0, max_s=7.0, multiplier=3.0, jitter=0.0).delay(10_000, lambda: 0) == 7.0
    )


def test_backoff_attempt_starts_at_one():
    with pytest.raises(ToolUsageError, match="attempt"):
        Backoff().delay(0, lambda: 0.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"base_s": -1},
        {"max_s": -1},
        {"multiplier": 0.5},
        {"jitter": -0.1},
        {"base_s": float("nan")},
    ],
)
def test_backoff_rejects_bad_values(fields):
    with pytest.raises(ConfigError):
        Backoff(**fields)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ValueError("x"), True),
        (ToolTimeout("t", 1.0), True),
        (ModelRateLimited("slow", provider="p"), True),
        (ToolError("no"), False),
        (ToolArgsInvalid("t", []), False),
        (ApprovalPending("apr_1"), False),
        (asyncio.CancelledError(), False),
        (GraphBubbleUp(), False),
        (KeyboardInterrupt(), False),
    ],
)
def test_retry_rule(exc, expected):
    assert is_retryable(exc) is expected


def _flaky(failures: list[BaseException], value: str = "ok"):
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return value

    return fn, calls


async def _try(fn, clock, *, retries=2, timeout=None, retried=None):
    return await _attempt(
        fn,
        retries=retries,
        timeout=timeout,
        backoff=Backoff(base_s=1.0, jitter=0.0),
        rng=lambda: 0.0,
        sleep=clock.sleep,
        on_timeout=lambda: ToolTimeout("t", timeout or 0),
        on_retry=(lambda n, delay, exc: retried.append((n, delay, type(exc))))
        if retried is not None
        else None,
    )


async def test_attempt_retries_then_succeeds_with_backoff_delays():
    clock, retried = FakeClock(), []
    fn, calls = _flaky([ValueError("a"), ValueError("b")])
    outcome = await _try(fn, clock, retried=retried)
    assert outcome.value == "ok" and outcome.error is None and outcome.attempts == 3
    assert calls["n"] == 3 and clock.sleeps == [1.0, 2.0]
    assert retried == [(1, 1.0, ValueError), (2, 2.0, ValueError)]


async def test_attempt_gives_up_after_retries():
    clock = FakeClock()
    fn, _ = _flaky([ValueError("a"), ValueError("b"), ValueError("c")])
    outcome = await _try(fn, clock, retries=1)
    assert outcome.attempts == 2 and isinstance(outcome.error, ValueError)
    assert str(outcome.error) == "b"


async def test_attempt_does_not_retry_non_retryable_errors():
    clock = FakeClock()
    fn, calls = _flaky([ToolError("bad input")])
    outcome = await _try(fn, clock, retries=3)
    assert calls["n"] == 1 and outcome.attempts == 1 and isinstance(outcome.error, ToolError)
    assert clock.sleeps == []


@pytest.mark.parametrize("signal", [asyncio.CancelledError(), GraphBubbleUp(), KeyboardInterrupt()])
async def test_attempt_lets_control_flow_and_base_exceptions_through(signal):
    fn, calls = _flaky([signal])
    with pytest.raises(type(signal)):
        await _try(fn, FakeClock(), retries=3)
    assert calls["n"] == 1


async def test_attempt_turns_its_own_timeout_into_on_timeout_error():
    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    outcome = await _try(slow, FakeClock(), retries=0, timeout=0.01)
    assert isinstance(outcome.error, ToolTimeout) and outcome.attempts == 1
    assert isinstance(outcome.error.__cause__, TimeoutError)


async def test_a_timeout_error_raised_by_the_function_is_not_converted():
    fn, _ = _flaky([TimeoutError("upstream")])
    outcome = await _try(fn, FakeClock(), retries=0, timeout=5)
    assert type(outcome.error) is TimeoutError


async def test_timeouts_are_retried():
    clock = FakeClock()
    state = {"n": 0}

    async def first_slow() -> str:
        state["n"] += 1
        if state["n"] == 1:
            await asyncio.sleep(1)
        return "ok"

    outcome = await _try(first_slow, clock, retries=1, timeout=0.01)
    assert outcome.value == "ok" and outcome.attempts == 2 and clock.sleeps == [1.0]
