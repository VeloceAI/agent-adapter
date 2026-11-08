"""A background writer: entries go on a bounded queue, one thread stores them in order."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from agent_adapter.audit._actions import AuditEntry

DEFAULT_QUEUE = 10_000
_STOP = object()


class Writer:
    """Stores entries with `write` on its own thread, so emitting never waits on the disk."""

    def __init__(self, write: Callable[[AuditEntry], None], *, max_queue: int) -> None:
        self._write = write
        self._queue: queue.Queue[object] = queue.Queue(max_queue)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def pending(self) -> int:
        """Entries queued or being stored."""
        return self._queue.unfinished_tasks

    def put(self, entry: AuditEntry) -> bool:
        """Queue one entry; False when the queue is full (the entry is not kept)."""
        self._start()
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            return False
        return True

    def drain(self, timeout: float) -> bool:
        """Wait until every queued entry is stored (or failed); False on timeout."""
        deadline = time.monotonic() + timeout
        done = self._queue.all_tasks_done
        with done:
            while self._queue.unfinished_tasks:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                done.wait(left)
        return True

    def stop(self, timeout: float) -> bool:
        """Store what is queued, then end the thread; False when it did not finish in time."""
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is None:
            return True
        self._queue.put(_STOP)
        thread.join(timeout)
        return not thread.is_alive()

    def _start(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name=f"agent-adapter-audit-{id(self):x}", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, AuditEntry)
                self._write(item)
            finally:
                self._queue.task_done()
