"""RateLimitHook: a process-wide token bucket per tool kind (and "llm"); waits, never fails."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.base import Hook, LLMInvocation, ToolInvocation

logger = logging.getLogger(__name__)

LLM_KIND = "llm"


@dataclass(frozen=True)
class Rate:
    """At most `calls` calls per `per` seconds, with bursts of up to `calls`."""

    calls: int
    per: float

    def __post_init__(self) -> None:
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) or self.calls < 1:
            raise ConfigError(f"Rate.calls must be an int >= 1, got {self.calls!r}")
        if not isinstance(self.per, int | float) or not math.isfinite(self.per) or self.per <= 0:
            raise ConfigError(f"Rate.per must be a finite number of seconds > 0, got {self.per!r}")


class _Bucket:
    """GCRA state: `tail` is when the latest reservation's slot frees (callers hold the lock)."""

    def __init__(self, rate: Rate) -> None:
        self.interval = rate.per / rate.calls  # seconds between calls at the steady rate
        self.burst = (rate.calls - 1) * self.interval  # how far ahead of `tail` a call may start
        self.tail = -math.inf

    def reserve(self, now: float) -> tuple[float, float]:
        """Book the next slot: (seconds to wait, the new tail that identifies this booking)."""
        tail = max(self.tail, now)
        start = max(now, tail - self.burst)
        self.tail = tail + self.interval
        return start - now, self.tail

    def release(self, booked_tail: float) -> None:
        """Give back a booking that will not run; only the newest one can be undone."""
        if self.tail == booked_tail:
            self.tail -= self.interval


class RateLimitHook(Hook):
    """Delays tool calls by `spec.kind` and model calls by `"llm"`; unlisted kinds never wait.

    One bucket per kind is shared by every event loop and thread: its state sits under a
    `threading.Lock`, and each caller books a slot there (GCRA) and then sleeps in its own
    loop, so no asyncio primitive is shared between loops. A cancelled wait returns its slot
    only when it was the latest booking, so a slot is never handed out twice.
    """

    name = "rate_limit"

    def __init__(
        self,
        per_kind: Mapping[str, Rate],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        for kind, rate in per_kind.items():
            if not isinstance(rate, Rate):
                raise ConfigError(f"per_kind[{kind!r}] must be a Rate, got {type(rate).__name__}")
        self.rates = dict(per_kind)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._buckets = {kind: _Bucket(rate) for kind, rate in per_kind.items()}

    async def before_tool(self, inv: ToolInvocation) -> None:
        await self._acquire(inv.spec.kind)

    async def before_llm(self, inv: LLMInvocation) -> None:
        await self._acquire(LLM_KIND)

    async def _acquire(self, kind: str) -> None:
        bucket = self._buckets.get(kind)
        if bucket is None:
            return
        with self._lock:
            wait, booked = bucket.reserve(self._clock())
        if wait <= 0:
            return
        logger.debug("rate limit for kind %r: waiting %.3f s", kind, wait)
        try:
            await self._sleep(wait)
        except BaseException:
            # Cancelled (or a failing sleep): the call never runs. Its slot is returned only
            # when no later booking depends on it; otherwise later callers keep their slots.
            with self._lock:
                bucket.release(booked)
            raise
