"""Helpers the third-party memory adapters share: sync-or-async calls, fields, timestamps."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.memory.records import MemoryScope

EXPORT_LIMIT = 10_000


async def call(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """`fn(*args, **kwargs)`, awaited when async; a sync client runs in a worker thread."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
    if inspect.isawaitable(result):
        result = await result
    return result


def field(item: Any, name: str, default: Any = None) -> Any:
    """`item[name]` for a dict, else `item.name`."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def results(found: Any, key: str = "results") -> list[Any]:
    """A backend's list: the value itself, or its `key` field (dict or object).

    Raises:
        IntegrationError: neither shape.
    """
    if found is None:
        return []
    if isinstance(found, list):
        return found
    if not isinstance(found, str | bytes):
        inner = field(found, key, ...)
        if inner is None:
            return []  # the backend found nothing
        if isinstance(inner, list):
            return inner
    raise IntegrationError(f"memory backend returned {type(found).__name__}, not a list")


def timestamp(value: Any) -> datetime | None:
    """A backend's time (datetime or ISO text) as aware UTC, or None."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def user_only(scope: MemoryScope, backend: str) -> str:
    """The user id of a user-only scope.

    Raises:
        ToolUsageError: the scope has no user id, or narrows by session or agent.
    """
    if not scope.user_id or scope.session_id or scope.agent_id:
        raise ToolUsageError(
            f"{backend} memory partitions by user only: use MemoryScope(user_id=...) "
            "(memory_tools(scope_from=user_scope))"
        )
    return scope.user_id


def group_id(user_id: str) -> str:
    """A graph group id for a user: safe characters, and the raw id is not stored."""
    return "aa_" + hashlib.sha256(user_id.encode("utf-8", "surrogatepass")).hexdigest()[:40]
