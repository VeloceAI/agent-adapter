"""A controllable clock for tests: call it for the time, await `sleep` to advance it."""

from __future__ import annotations

import asyncio


class FakeClock:
    """Time that moves only when told to; `sleep` advances it instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._now

    __call__ = now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)
