"""`CallableRetriever`: any search function (vector DB, search service) as a retriever."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.documents import FilterValue

# fn(query, k, filters) -> documents (Document, str, or a mapping with "text"); sync or async.
RetrieveFn = Callable[[str, int, dict[str, FilterValue]], Any]


class CallableRetriever(GuardedRetriever):
    """Wraps `fn(query, k, filters)`; a sync `fn` runs in a worker thread with the contextvars.

    Raises:
        ToolUsageError: `fn` is not callable.
        ConfigError: a bad `timeout_s`.
    """

    backend = "callable"

    def __init__(
        self,
        fn: RetrieveFn,
        *,
        name: str = "",
        timeout_s: float | None = None,
        capture_content: bool = False,
    ) -> None:
        if not callable(fn):
            raise ToolUsageError(f"CallableRetriever fn must be callable, got {type(fn).__name__}")
        super().__init__(name=name, timeout_s=timeout_s, capture_content=capture_content)
        self._fn = fn
        self._is_async = is_async(fn)

    async def _retrieve(self, query: str, k: int, filters: dict[str, FilterValue]) -> Sequence[Any]:
        if self._is_async:
            raw = await self._fn(query, k, dict(filters))
        else:
            raw = await asyncio.to_thread(self._fn, query, k, dict(filters))
        if inspect.isawaitable(raw):
            raw = await raw
        return results(raw, self.name)


def results(raw: Any, name: str) -> list[Any]:
    """`raw` (None, or any iterable but a str, bytes or mapping) as a list of items.

    Raises:
        ToolUsageError: `raw` is a str, bytes, mapping or not iterable.
    """
    if raw is None:
        return []
    if isinstance(raw, str | bytes | Mapping) or not isinstance(raw, Iterable):
        raise ToolUsageError(f"retriever {name!r} must return a list, got {type(raw).__name__}")
    return list(raw)
