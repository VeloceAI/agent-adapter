"""In-process memory of interrupted LangGraph tasks, so a resumed task does not redo its work.

LangGraph re-runs an interrupted node from the top. A task that pauses keeps what it already
finished here, keyed by run id and the task's checkpoint namespace (which names the exact
task of the exact checkpoint); a re-run in the same process picks it up. The memory is
bounded (size and age) and holds nothing a resume value does not confirm again.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Generic, TypeVar, Unpack

from agent_adapter.core._bounded import Bounded, Bounds

logger = logging.getLogger(__name__)

V = TypeVar("V")
Key = tuple[str, ...]


class TaskMemo(Bounded, Generic[V]):
    """A thread-safe map of task state, oldest write evicted first, with a time to live."""

    def __init__(self, **bounds: Unpack[Bounds]) -> None:
        super().__init__(**bounds)
        self._items: OrderedDict[Key, tuple[float, V]] = OrderedDict()

    def get(self, key: Key) -> V | None:
        """The value kept for `key`, or None."""
        with self._lock:
            self._expire()
            found = self._items.get(key)
            return None if found is None else found[1]

    def put(self, key: Key, value: V) -> None:
        """Keep `value` for `key` (evicting the oldest entries past `max_entries`)."""
        with self._lock:
            self._items[key] = (self._clock(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                evicted, _ = self._items.popitem(last=False)
                logger.debug("task memo full: forgot %s", evicted[-1])

    def drop(self, key: Key) -> None:
        """Forget `key` (the task finished)."""
        with self._lock:
            self._items.pop(key, None)

    def drop_if(self, key: Key, value: V) -> None:
        """Forget `key` while it still holds `value` (not a newer write)."""
        with self._lock:
            found = self._items.get(key)
            if found is not None and found[1] is value:
                del self._items[key]

    def clear(self) -> None:
        """Forget everything (tests; a restarted process starts empty)."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _expire(self) -> None:
        horizon = self._horizon()
        while self._items and next(iter(self._items.values()))[0] < horizon:
            self._items.popitem(last=False)
