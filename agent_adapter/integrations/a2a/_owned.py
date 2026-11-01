"""An expiring table keyed by (owner, key), capped in total and per owner."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")
Entry = tuple[str, str, V]  # (owner, key, value)


class OwnerFull(Exception):
    """The owner already holds its cap of entries (internal signal, never leaves the package)."""


class OwnedTable(Generic[V]):
    """Entries live `ttl` seconds after their last `put`; caps keep one owner from evicting others.

    Over `max_per_owner`, `put` either raises `OwnerFull` (`evict_own=False`) or drops that
    owner's oldest entry. Over `max_entries`, the oldest entry is dropped. Both prefer entries
    `disposable` accepts, then the oldest.
    """

    def __init__(
        self,
        *,
        ttl: float,
        max_entries: int,
        max_per_owner: int,
        evict_own: bool,
        disposable: Callable[[V], bool] = lambda _: True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not (ttl > 0 and max_entries > 0 and max_per_owner > 0):
            raise ValueError("ttl, max_entries and max_per_owner must be positive")
        self._ttl = ttl
        self._max = max_entries
        self._max_owner = max_per_owner
        self._evict_own = evict_own
        self._disposable = disposable
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], tuple[V, float]] = OrderedDict()
        self._owners: dict[str, OrderedDict[str, None]] = {}

    def __len__(self) -> int:
        with self._lock:
            self._purge()
            return len(self._entries)

    def count(self, owner: str) -> int:
        """Live entries held by `owner`."""
        with self._lock:
            self._purge()
            return len(self._owners.get(owner, ()))

    def put(self, owner: str, key: str, value: V) -> tuple[list[Entry[V]], list[Entry[V]]]:
        """Store `value` and restart its TTL; returns (expired, evicted) entries.

        Raises:
            OwnerFull: a new key over the owner's cap, with `evict_own=False`.
        """
        with self._lock:
            expired = self._purge()
            evicted: list[Entry[V]] = []
            owned = self._owners.get(owner)
            if (owner, key) not in self._entries:
                if owned is not None and len(owned) >= self._max_owner:
                    if not self._evict_own:
                        raise OwnerFull(owner)
                    evicted.append(self._evict([(owner, k) for k in owned]))
                while len(self._entries) >= self._max:
                    evicted.append(self._evict(list(self._entries)))
            self._entries[(owner, key)] = (value, self._clock() + self._ttl)
            self._entries.move_to_end((owner, key))
            self._owners.setdefault(owner, OrderedDict())[key] = None
            self._owners[owner].move_to_end(key)
            return expired, evicted

    def get(self, owner: str, key: str) -> V | None:
        """The live value under (owner, key)."""
        with self._lock:
            self._purge()
            entry = self._entries.get((owner, key))
        return entry[0] if entry is not None else None

    def pop(self, owner: str, key: str) -> V | None:
        """Remove and return the live value under (owner, key)."""
        with self._lock:
            self._purge()
            if (owner, key) not in self._entries:
                return None
            return self._remove((owner, key))

    def expired(self) -> list[Entry[V]]:
        """Remove and return the entries past their TTL."""
        with self._lock:
            return self._purge()

    def _evict(self, candidates: list[tuple[str, str]]) -> Entry[V]:
        """Drop the oldest disposable candidate, else the oldest one (candidates are ordered)."""
        victim = next(
            (ref for ref in candidates if self._disposable(self._entries[ref][0])), candidates[0]
        )
        return victim[0], victim[1], self._remove(victim)

    def _remove(self, ref: tuple[str, str]) -> V:
        value, _ = self._entries.pop(ref)
        owned = self._owners[ref[0]]
        del owned[ref[1]]
        if not owned:
            del self._owners[ref[0]]
        return value

    def _purge(self) -> list[Entry[V]]:
        now = self._clock()
        gone: list[Entry[V]] = []
        while self._entries:
            ref, (_, deadline) = next(iter(self._entries.items()))
            if deadline > now:  # entries are in deadline order: TTL is fixed, put moves to end
                break
            gone.append((ref[0], ref[1], self._remove(ref)))
        return gone
