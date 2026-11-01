"""In-process CachePort: an LRU dict with monotonic-clock TTLs."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)


class MemoryCache:
    """LRU byte cache bounded by `max_entries`; TTLs in seconds on `clock` (monotonic)."""

    def __init__(
        self, *, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 1:
            raise ConfigError(f"MemoryCache max_entries must be >= 1, got {max_entries}")
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[bytes, float | None]] = OrderedDict()
        # The cache may be shared by loops running in several threads (run_sync).
        self._lock = threading.Lock()
        self._purge_budget = 0  # overflows left before the next full expiry scan

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    async def get(self, key: str) -> bytes | None:
        """The value, or None when missing or expired."""
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if expires is not None and self._clock() >= expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        """Store `value`; `ttl <= 0` removes the key; `inf` / `nan` never expire."""
        if ttl is not None and (math.isnan(ttl) or ttl == math.inf):
            ttl = None
        with self._lock:
            if ttl is not None and ttl <= 0:
                self._data.pop(key, None)
                return
            expires = None if ttl is None else self._clock() + ttl
            self._data[key] = (value, expires)
            self._data.move_to_end(key)
            if len(self._data) > self.max_entries:
                self._evict()

    async def delete(self, key: str) -> None:
        """Remove `key` if present."""
        with self._lock:
            self._data.pop(key, None)

    def _evict(self) -> None:
        # Caller holds _lock. Expired entries go first so they rarely displace live ones; the
        # O(n) scan runs once per max_entries overflows, so a set stays O(1) amortised.
        if self._purge_budget <= 0:
            now = self._clock()
            for key in [k for k, (_, exp) in self._data.items() if exp is not None and now >= exp]:
                del self._data[key]
            self._purge_budget = self.max_entries
        self._purge_budget -= 1
        evicted = 0
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
            evicted += 1
        if evicted:
            logger.debug("memory cache evicted %d least recently used entr(y/ies)", evicted)
