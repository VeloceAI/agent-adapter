"""The `BudgetStore` port, its value types and the in-process `MemoryBudgetStore`."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

# Float sums drift; a total this close to the limit still counts as within it.
EPSILON = 1e-9


@dataclass(frozen=True)
class BudgetUsage:
    """A counter's settled spend and its live holds (USD)."""

    spent_usd: float = 0.0
    reserved_usd: float = 0.0

    @property
    def committed_usd(self) -> float:
        """Spent plus held: what a new reservation is checked against."""
        return self.spent_usd + self.reserved_usd


@dataclass(frozen=True)
class Reserved:
    """The outcome of `reserve`: accepted or not, and the usage it was checked against."""

    accepted: bool
    usage: BudgetUsage


@runtime_checkable
class BudgetStore(Protocol):
    """Spend counters shared by every hook that uses the store; `reserve` is atomic.

    A hold (`reserve`) counts against the limit until it is settled with the real cost,
    released, or `hold_s` passes (a crashed caller's hold frees itself). `expire_at` (epoch
    seconds) is when the whole counter may be dropped, its window having ended; None
    leaves the counter's current expiry as it is.
    Backend errors propagate, so a budget check fails closed.
    """

    async def reserve(
        self,
        key: str,
        reservation_id: str,
        amount_usd: float,
        *,
        limit_usd: float | None,
        hold_s: float,
        expire_at: float | None,
    ) -> Reserved: ...

    async def settle(
        self, key: str, reservation_id: str, cost_usd: float, *, expire_at: float | None
    ) -> None: ...

    async def release(self, key: str, reservation_id: str) -> None: ...

    async def usage(self, key: str) -> BudgetUsage: ...


def admits(usage: BudgetUsage, amount_usd: float, limit_usd: float | None) -> bool:
    """True when a hold of `amount_usd` fits: committed below the limit, and stays within it."""
    if limit_usd is None:
        return True
    committed = usage.committed_usd
    return committed < limit_usd and committed + amount_usd <= limit_usd + EPSILON


def check_reserve(key: str, reservation_id: str, amount_usd: float, hold_s: float) -> None:
    """Refuse malformed `reserve` arguments.

    Raises:
        ConfigError: an empty key or id, a negative or non-finite amount, or `hold_s` <= 0.
    """
    check_ids(key, reservation_id)
    check_usd(amount_usd, "amount_usd")
    if not _finite(hold_s) or hold_s <= 0:
        raise ConfigError(f"hold_s must be a finite number > 0, got {hold_s!r}")


def check_ids(key: str, reservation_id: str) -> None:
    """Refuse an empty or non-string key or reservation id.

    Raises:
        ConfigError: either is not a non-empty str.
    """
    for name, value in (("key", key), ("reservation_id", reservation_id)):
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{name} must be a non-empty str, got {value!r}")


def check_usd(value: float, name: str) -> None:
    """Refuse a negative, non-finite or non-numeric amount.

    Raises:
        ConfigError: `value` is not a finite number >= 0.
    """
    if not _finite(value) or value < 0:
        raise ConfigError(f"{name} must be a finite number >= 0, got {value!r}")


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


@dataclass
class _Counter:
    spent: float = 0.0
    holds: dict[str, tuple[float, float]] = field(default_factory=dict)  # id -> (usd, until)
    expire_at: float | None = None

    def live(self, now: float) -> BudgetUsage:
        for rid in [rid for rid, (_, until) in self.holds.items() if until <= now]:
            del self.holds[rid]
        return BudgetUsage(self.spent, sum(usd for usd, _ in self.holds.values()))


class MemoryBudgetStore:
    """A `BudgetStore` in this process (threads and event loops share it), the default.

    Keeps at most `max_keys` live counters; ended windows are dropped first, and a new key
    past that is refused (fails closed) rather than forgetting another key's spend.

    Raises:
        ConfigError: `max_keys` is not a positive int.
    """

    def __init__(self, *, max_keys: int = 100_000, clock: Callable[[], float] = time.time) -> None:
        if isinstance(max_keys, bool) or not isinstance(max_keys, int) or max_keys < 1:
            raise ConfigError(f"max_keys must be a positive int, got {max_keys!r}")
        self.max_keys = max_keys
        self._clock = clock
        self._counters: OrderedDict[str, _Counter] = OrderedDict()
        self._lock = threading.Lock()

    async def reserve(
        self,
        key: str,
        reservation_id: str,
        amount_usd: float,
        *,
        limit_usd: float | None,
        hold_s: float,
        expire_at: float | None,
    ) -> Reserved:
        """Hold `amount_usd` when it fits `limit_usd` (None: always); one atomic step."""
        check_reserve(key, reservation_id, amount_usd, hold_s)
        now = self._clock()
        with self._lock:
            counter = self._counter(key, now, expire_at)
            usage = counter.live(now)
            if not admits(usage, amount_usd, limit_usd):
                return Reserved(False, usage)
            counter.holds[reservation_id] = (float(amount_usd), now + hold_s)
            return Reserved(True, usage)

    async def settle(
        self, key: str, reservation_id: str, cost_usd: float, *, expire_at: float | None
    ) -> None:
        """Drop the hold and add the real cost (even when the hold already expired)."""
        check_ids(key, reservation_id)
        check_usd(cost_usd, "cost_usd")
        with self._lock:
            counter = self._counter(key, self._clock(), expire_at)
            counter.holds.pop(reservation_id, None)
            counter.spent += float(cost_usd)

    async def release(self, key: str, reservation_id: str) -> None:
        """Drop the hold; nothing was spent."""
        check_ids(key, reservation_id)
        with self._lock:
            counter = self._counters.get(key)
            if counter is not None:
                counter.holds.pop(reservation_id, None)

    async def usage(self, key: str) -> BudgetUsage:
        """The counter's spend and live holds; zero for an unknown or expired key."""
        now = self._clock()
        with self._lock:
            counter = self._counters.get(key)
            if counter is None or _expired(counter, now):
                return BudgetUsage()
            return counter.live(now)

    def _counter(self, key: str, now: float, expire_at: float | None) -> _Counter:
        counter = self._counters.get(key)
        if counter is None or _expired(counter, now):
            self._make_room(key, now)
            counter = _Counter()
            self._counters[key] = counter
        if expire_at is not None:
            counter.expire_at = expire_at
        self._counters.move_to_end(key)
        return counter

    def _make_room(self, key: str, now: float) -> None:
        """Drop ended windows when full.

        Raises:
            ConfigError: still `max_keys` live counters: the new key is refused.
        """
        if key in self._counters or len(self._counters) < self.max_keys:
            return
        for ended in [k for k, c in self._counters.items() if _expired(c, now)]:
            del self._counters[ended]
        if len(self._counters) >= self.max_keys:
            logger.error("budget store full (%d live keys): refusing a new counter", self.max_keys)
            raise ConfigError(f"MemoryBudgetStore is full ({self.max_keys} keys); raise max_keys")


def _expired(counter: _Counter, now: float) -> bool:
    return counter.expire_at is not None and counter.expire_at <= now
