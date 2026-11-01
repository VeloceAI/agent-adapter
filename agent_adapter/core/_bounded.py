"""Size and age bounds shared by the process-wide in-memory registries."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypedDict


class Bounds(TypedDict, total=False):
    """The keyword arguments of `Bounded`."""

    max_entries: int
    ttl: float
    clock: Callable[[], float]


class Bounded:
    """A lock plus at most `max_entries` entries per map, each kept for `ttl` seconds."""

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        ttl: float = 24 * 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()

    def _horizon(self) -> float:
        """The oldest timestamp still alive."""
        return self._clock() - self.ttl
