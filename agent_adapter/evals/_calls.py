"""Call user code the way the chat host does: sync in a worker thread, streams joined."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from contextlib import aclosing
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import ToolCallDelta


def on_loop(fn: object) -> bool:
    """True when calling `fn` only builds a coroutine or an async generator."""
    call = type(fn).__call__ if not inspect.isroutine(fn) else fn
    return is_async(fn) or inspect.isasyncgenfunction(fn) or inspect.isasyncgenfunction(call)


async def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`fn(*args, **kwargs)` awaited; a sync `fn` runs in a worker thread.

    A timeout cannot stop a worker thread: a sync `fn` runs on after its case timed out.
    """
    if on_loop(fn):
        result = fn(*args, **kwargs)
    else:
        result = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def joined(result: Any, who: str) -> Any:
    """`result` itself, or the joined text of an async iterator of str chunks (deltas skipped).

    Raises:
        ToolUsageError: a streamed chunk is not text.
    """
    if not hasattr(result, "__aiter__"):
        return result
    chunks: list[str] = []
    if hasattr(result, "aclose"):
        async with aclosing(result) as stream:
            async for chunk in stream:
                if not isinstance(chunk, ToolCallDelta):  # a draft tool card, not text
                    chunks.append(_chunk(chunk, who))
    else:
        async for chunk in result:
            if not isinstance(chunk, ToolCallDelta):
                chunks.append(_chunk(chunk, who))
    return "".join(chunks)


def _chunk(chunk: object, who: str) -> str:
    if not isinstance(chunk, str):
        raise ToolUsageError(f"{who} must stream str chunks, got {type(chunk).__name__}")
    return chunk
