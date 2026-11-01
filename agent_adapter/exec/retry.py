"""Backoff, the retry rule and the attempt loop shared by tool calls and `parallel`."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import AdapterError, ConfigError, ToolUsageError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Backoff:
    """Exponential backoff with jitter; delays in seconds."""

    base_s: float = 0.5
    max_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25  # delay *= 1 + rng() * jitter

    def __post_init__(self) -> None:
        checks = {
            "base_s": self.base_s >= 0,
            "max_s": self.max_s >= 0,
            "multiplier": self.multiplier >= 1,
            "jitter": self.jitter >= 0,
        }
        for name, ok in checks.items():
            value = getattr(self, name)
            if not ok or not math.isfinite(value):
                raise ConfigError(f"Backoff.{name} is out of range: {value!r}")

    def delay(self, attempt: int, rng: Callable[[], float]) -> float:
        """Seconds to wait after failed attempt `attempt` (1-based).

        Raises:
            ToolUsageError: `attempt` is below 1.
        """
        if attempt < 1:
            raise ToolUsageError(f"Backoff.delay: attempt starts at 1, got {attempt}")
        try:
            raw = self.base_s * self.multiplier ** (attempt - 1)
        except OverflowError:
            raw = math.inf
        return min(self.max_s, raw) * (1 + rng() * self.jitter)


def is_retryable(exc: BaseException) -> bool:
    """The retry rule: not control flow, and foreign or an AdapterError marked retryable."""
    if not isinstance(exc, Exception) or is_control_flow(exc):
        return False
    return not isinstance(exc, AdapterError) or type(exc).retryable


@dataclass(frozen=True)
class _Outcome(Generic[T]):
    """What the attempt loop ended with: a value, or the final error."""

    attempts: int
    value: T | None = None
    error: Exception | None = None


async def _attempt(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    timeout: float | None,
    backoff: Backoff,
    rng: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    on_timeout: Callable[[], Exception],
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> _Outcome[T]:
    """Run `fn` up to `retries + 1` times, each under `timeout`; control flow propagates.

    Args:
        on_timeout: Builds the error reported when `timeout` (not `fn` itself) expires.
        on_retry: Called with (failed attempt, delay, error) before each backoff sleep.
    """
    number = 0
    while True:
        number += 1
        try:
            async with asyncio.timeout(timeout) as scope:
                value = await fn()
            return _Outcome(attempts=number, value=value)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            error = exc
            if isinstance(exc, TimeoutError) and scope.expired():
                error = on_timeout()
                error.__cause__ = exc
        if number > retries or not is_retryable(error):
            return _Outcome(attempts=number, error=error)
        delay = backoff.delay(number, rng)
        if on_retry is not None:
            on_retry(number, delay, error)
        await sleep(delay)
