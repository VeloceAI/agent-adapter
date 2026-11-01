"""A small thread-safe table whose entries expire, bounded by dropping the oldest first."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class ExpiringTable(Generic[K, V]):
    """Entries live until their deadline; a full table drops its oldest entries to make room."""

    def __init__(
        self,
        *,
        ttl: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
        label: str = "entries",
    ) -> None:
        self._ttl = ttl
        self._max = max_entries
        self._clock = clock
        self._label = label  # what the entries are, for DEBUG logs
        self._lock = threading.Lock()  # callers may run on worker threads
        self._entries: OrderedDict[K, tuple[V, float]] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            self._purge()
            return len(self._entries)

    def deadline(self) -> float:
        """When an entry added now expires."""
        return self._clock() + self._ttl

    def add(self, key: K, value: V, *, expires_at: float | None = None) -> list[V]:
        """Store `value` under `key`; returns the entries dropped to stay within the bound."""
        expiry = self.deadline() if expires_at is None else expires_at
        with self._lock:
            self._purge()
            dropped = []
            while self._entries and len(self._entries) >= self._max:
                _, (old, _) = self._entries.popitem(last=False)
                dropped.append(old)
            self._entries[key] = (value, expiry)
        return dropped

    def pop(self, key: K) -> V | None:
        """Remove and return the live entry under `key`."""
        with self._lock:
            self._purge()
            entry = self._entries.pop(key, None)
        return entry[0] if entry is not None else None

    def pop_first(self, match: Callable[[V], bool]) -> V | None:
        """Remove and return the oldest live entry `match` accepts."""
        with self._lock:
            self._purge()
            for key, (value, _) in self._entries.items():
                if match(value):
                    del self._entries[key]
                    return value
        return None

    def count(self, match: Callable[[V], bool]) -> int:
        """How many live entries `match` accepts."""
        with self._lock:
            self._purge()
            return sum(1 for value, _ in self._entries.values() if match(value))

    def clear(self) -> int:
        """Drop every entry; returns how many there were."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        return count

    def _purge(self) -> None:
        now = self._clock()
        expired = [key for key, (_, expiry) in self._entries.items() if expiry <= now]
        for key in expired:
            del self._entries[key]
        if expired:
            logger.debug("%d %s expired", len(expired), self._label)
