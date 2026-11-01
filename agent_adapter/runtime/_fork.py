"""A module-level lock that a forked child gets back fresh and unheld."""

from __future__ import annotations

import os
import threading
import weakref
from types import TracebackType

_LOCKS: weakref.WeakSet[ForkSafeLock] = weakref.WeakSet()


class ForkSafeLock:
    """A `threading.Lock` re-created in a forked child, where a parent thread's hold never frees."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        _LOCKS.add(self)

    def __enter__(self) -> bool:
        return self._lock.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._lock.__exit__(exc_type, exc, tb)

    def locked(self) -> bool:
        """True while some thread holds the lock."""
        return self._lock.locked()

    def _renew(self) -> None:
        self._lock = threading.Lock()


def renew_locks_after_fork() -> None:
    """Give every ForkSafeLock a fresh lock; runs in a forked child."""
    for lock in list(_LOCKS):
        lock._renew()


if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=renew_locks_after_fork)
