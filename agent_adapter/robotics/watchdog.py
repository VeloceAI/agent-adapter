"""Deadman watchdog: no heartbeat for `timeout` seconds stops the robot."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

__all__ = ["Deadman"]

logger = logging.getLogger(__name__)


class Deadman:
    """Calls `on_lost` once when `beat()` has not been called for `timeout` seconds.

    Heartbeats should come from the operator or the controlling process, never from the
    model: a model that can beat can be told to keep beating.
    """

    def __init__(
        self,
        timeout: float,
        on_lost: Callable[[], Awaitable[None]],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be above 0")
        self.timeout = timeout
        self._on_lost = on_lost
        self._clock = clock
        self._last = clock()
        self._task: asyncio.Task[None] | None = None
        self.tripped = False

    def beat(self) -> None:
        self._last = self._clock()

    def alive(self) -> bool:
        return not self.tripped and self._clock() - self._last <= self.timeout

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.beat()
            self.tripped = False
            self._task = asyncio.get_running_loop().create_task(self._watch())

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _watch(self) -> None:
        interval = self.timeout / 10
        while True:
            late = self._clock() - self._last - self.timeout
            if late > 0:
                self.tripped = True
                logger.warning("deadman: no heartbeat for %.3f s; stopping", self.timeout + late)
                await self._on_lost()
                return
            await asyncio.sleep(min(interval, -late))
