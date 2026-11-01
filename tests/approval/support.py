"""Helpers shared by the approval tests."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable

from agent_adapter.runtime.sync import run_sync


def ids_from_threads(burst: Callable[[], Awaitable[list[str]]], *, threads: int = 2) -> list[str]:
    """Start `burst` on `threads` separate `run_sync` loops at once; every id they return."""
    barrier = threading.Barrier(threads)
    ids: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        found = run_sync(burst, timeout=10)
        with lock:
            ids.extend(found)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=10)
    return ids
