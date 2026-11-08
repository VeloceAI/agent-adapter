"""`CallableMemory`: any memory backend from host functions, with the shared checks around it."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import FilterValue, MemoryHit, MemoryRecord, MemoryScope

# add(records) -> stored records (or None: stored as given)
# search(query, scope, k, filters) -> MemoryHit | MemoryRecord | (record, score), iterable
# get(id, scope) -> MemoryRecord | None; delete(ids, scope) -> count; export(scope) -> records
MemoryFn = Callable[..., Any]


class CallableMemory(GuardedMemory):
    """Wraps host functions (sync or async) as a `MemoryPort`.

    `add` and `search` are required; `get`, `delete` and `export` are optional and raise
    `ToolUsageError` when missing. Whatever they return still passes the scope, expiry and
    filter checks, so a backend bug cannot leak another scope's records.

    Raises:
        ToolUsageError: a given function is not callable.
    """

    def __init__(
        self,
        *,
        add: MemoryFn,
        search: MemoryFn,
        get: MemoryFn | None = None,
        delete: MemoryFn | None = None,
        export: MemoryFn | None = None,
        backend: str = "callable",
        retention_days: float | None = None,
        capture_content: bool = False,
    ) -> None:
        super().__init__(retention_days=retention_days, capture_content=capture_content)
        fns = {"add": add, "search": search, "get": get, "delete": delete, "export": export}
        for name, fn in fns.items():
            if fn is not None and not callable(fn):
                raise ToolUsageError(f"CallableMemory {name}= must be callable")
        if not isinstance(backend, str) or not backend:
            raise ToolUsageError("CallableMemory backend= must be a non-empty str")
        self.backend = backend
        self._fns = fns

    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        stored = await self._call("add", records)
        if stored is None:
            return records
        return [_as_record(item) for item in _iterable(stored, "add")]

    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]:
        found = await self._call("search", query, scope, k, filters)
        return [_as_hit(item) for item in _iterable(found, "search")], "backend"

    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None:
        found = await self._call("get", id, scope)
        return None if found is None else _as_record(found)

    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int:
        count = await self._call("delete", ids, scope)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise IntegrationError("CallableMemory delete= must return a count (int >= 0)")
        return count

    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]:
        found = await self._call("export", scope)
        return [_as_record(item) for item in _iterable(found, "export")]

    async def _call(self, name: str, *args: Any) -> Any:
        fn = self._fns[name]
        if fn is None:
            raise ToolUsageError(f"this CallableMemory has no {name}= function")
        if is_async(fn):
            result = await fn(*args)
        else:
            result = await asyncio.to_thread(fn, *args)  # copies the contextvars
        if inspect.isawaitable(result):
            result = await result
        return result


def _iterable(value: Any, name: str) -> Iterable[Any]:
    if isinstance(value, str | bytes | dict) or not isinstance(value, Iterable):
        raise IntegrationError(f"CallableMemory {name}= must return a list")
    return value


def _as_record(value: Any) -> MemoryRecord:
    if isinstance(value, MemoryRecord):
        return value
    return MemoryRecord.model_validate(value)


def _as_hit(value: Any) -> MemoryHit:
    if isinstance(value, MemoryHit):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return MemoryHit(record=_as_record(value[0]), score=float(value[1]))
    return MemoryHit(record=_as_record(value))
