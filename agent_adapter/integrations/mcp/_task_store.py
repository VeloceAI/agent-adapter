"""Task state shared by every worker of an MCP server: in memory, or in Redis.

The store only mirrors what a task shows to its caller. It never decides an approval:
that stays in the executor's `ApprovalStore`.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from agent_adapter.core.errors import AdapterError, ConfigError, log_failure
from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)


class TaskStoreCorrupt(AdapterError):
    """A stored task state is not readable JSON."""

    kind = "task-store-corrupt"


@runtime_checkable
class TaskStore(Protocol):
    """Task states (written by the worker running the task) and signals (from any worker)."""

    async def put(self, task_id: str, state: Mapping[str, Any], ttl_s: float) -> None:
        """Store `state` for `ttl_s` seconds, replacing the previous one."""
        ...

    async def get(self, task_id: str) -> dict[str, Any] | None:
        """The live state, or None."""
        ...

    async def delete(self, task_id: str) -> None:
        """Forget the state."""
        ...

    async def flag(self, task_id: str, name: str, ttl_s: float) -> None:
        """Raise the signal `name` (e.g. a cancel request) for `ttl_s` seconds."""
        ...

    async def flagged(self, task_id: str, names: Sequence[str]) -> set[str]:
        """Which of `names` are raised."""
        ...


class MemoryTaskStore:
    """In-process `TaskStore`: shared by the servers of one process only.

    Past `max_entries` the oldest entries go first (WARNING).
    """

    def __init__(
        self, *, max_entries: int = 100_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ConfigError(f"max_entries must be a positive int, got {max_entries!r}")
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()

    async def put(self, task_id: str, state: Mapping[str, Any], ttl_s: float) -> None:
        self._set("t:" + task_id, json.loads(_encode(state)), ttl_s)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._live("t:" + task_id)
        return None if value is None else dict(value)

    async def delete(self, task_id: str) -> None:
        with self._lock:
            self._entries.pop("t:" + task_id, None)

    async def flag(self, task_id: str, name: str, ttl_s: float) -> None:
        self._set(f"f:{task_id}:{name}", True, ttl_s)

    async def flagged(self, task_id: str, names: Sequence[str]) -> set[str]:
        with self._lock:
            return {n for n in names if self._live(f"f:{task_id}:{n}") is not None}

    def _set(self, key: str, value: Any, ttl_s: float) -> None:
        with self._lock:
            self._entries[key] = (self._clock() + _checked_ttl(ttl_s), value)
            self._entries.move_to_end(key)
            if len(self._entries) > self._max:
                self._entries.popitem(last=False)
                logger.warning("MCP task store full (%d entries): dropped the oldest", self._max)

    def _live(self, key: str) -> Any:
        entry = self._entries.get(key)
        if entry is not None and entry[0] <= self._clock():
            del self._entries[key]
            return None
        return None if entry is None else entry[1]


class RedisTaskStore:
    """`TaskStore` on Redis (`agent-adapter[redis]`), shared by every worker and host.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop), e.g.
    `lambda: Redis.from_url(url)`. States are `SET PX` JSON values under `prefix`.
    Backend errors propagate; a task then stays readable only on its own worker.
    """

    def __init__(self, client_factory: Callable[[], Any], *, prefix: str = "aa:mcp-task:") -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be callable (it returns a redis client)")
        self.prefix = prefix
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    async def put(self, task_id: str, state: Mapping[str, Any], ttl_s: float) -> None:
        await self._clients.get().set(self._state(task_id), _encode(state), px=_ms(ttl_s))

    async def get(self, task_id: str) -> dict[str, Any] | None:
        raw = await self._clients.get().get(self._state(task_id))
        return None if raw is None else _decode(raw)

    async def delete(self, task_id: str) -> None:
        await self._clients.get().delete(self._state(task_id))

    async def flag(self, task_id: str, name: str, ttl_s: float) -> None:
        await self._clients.get().set(self._flag(task_id, name), b"1", px=_ms(ttl_s))

    async def flagged(self, task_id: str, names: Sequence[str]) -> set[str]:
        if not names:
            return set()
        values = await self._clients.get().mget([self._flag(task_id, n) for n in names])
        return {name for name, value in zip(names, values, strict=True) if value is not None}

    async def aclose(self) -> None:
        """Close the running loop's client."""
        await self._clients.aclose()

    def _state(self, task_id: str) -> str:
        return f"{self.prefix}t:{task_id}"

    def _flag(self, task_id: str, name: str) -> str:
        return f"{self.prefix}f:{task_id}:{name}"


def _encode(state: Mapping[str, Any]) -> str:
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _decode(raw: Any) -> dict[str, Any]:
    """The state in a stored JSON text; TaskStoreCorrupt when it is not a JSON object."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError(f"stored task state is a JSON {type(data).__name__}, not an object")
    except (ValueError, TypeError, RecursionError) as exc:
        log_failure(logger, logging.ERROR, exc, "MCP task store holds an unreadable state")
        raise TaskStoreCorrupt("MCP task store holds an unreadable state") from exc
    return data


def _ms(seconds: float) -> int:
    return max(1, math.ceil(_checked_ttl(seconds) * 1000))


def _checked_ttl(seconds: float) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        raise ConfigError(f"ttl_s must be a number, got {type(seconds).__name__}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ConfigError(f"ttl_s must be finite and > 0, got {seconds!r}")
    return float(seconds)


class OutageLog:
    """Logs a store failure at WARNING once per key, repeats at DEBUG, and the recovery at INFO."""

    def __init__(self, log: logging.Logger) -> None:
        self._log = log
        self._down: set[str] = set()

    def failed(self, key: str, exc: BaseException, message: str, *args: object) -> None:
        level = logging.DEBUG if key in self._down else logging.WARNING
        self._down.add(key)
        log_failure(self._log, level, exc, message, *args)

    def ok(self, key: str, message: str, *args: object) -> None:
        if key in self._down:
            self._down.discard(key)
            self._log.info(message, *args)

    def forget(self, key: str) -> None:
        self._down.discard(key)
