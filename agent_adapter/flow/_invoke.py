"""Calling user functions from flow nodes."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.callables import is_async
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError


def describe_error(exc: BaseException) -> str:
    """An error's kind and message for AdapterErrors; only the class name otherwise."""
    if isinstance(exc, AdapterError):
        return f"{type(exc).kind}: {exc}"
    return type(exc).__name__


def wants_ctx(fn: Callable[..., Any]) -> bool:
    """True when `fn` takes a second positional parameter (the node context)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    kinds = [p.kind for p in params]
    if inspect.Parameter.VAR_POSITIONAL in kinds:
        return True
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(kind in positional for kind in kinds) >= 2


async def invoke_node_fn(fn: Callable[..., Any], state: BaseModel, ctx: ToolContext) -> Any:
    """Call a node function as `fn(state)` or `fn(state, ctx)`."""
    if wants_ctx(fn):
        return await invoke(fn, state, ctx)
    return await invoke(fn, state)


async def invoke(fn: Callable[..., Any], *args: Any) -> Any:
    """Call `fn`; sync functions run in a worker thread (context variables are copied)."""
    if is_async(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        result = await result
    return result
