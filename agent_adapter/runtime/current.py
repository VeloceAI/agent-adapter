"""The ToolContext of the call or flow node running now (a ContextVar)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from agent_adapter.core.context import ToolContext
from agent_adapter.runtime.ids import new_id

_CURRENT: ContextVar[ToolContext | None] = ContextVar("agent_adapter_context", default=None)


def current_context() -> ToolContext | None:
    """The context set by the innermost `use_context`, or None."""
    return _CURRENT.get()


def resolve_context(bound: ToolContext | None) -> ToolContext:
    """The bound context, else the current one, else a fresh run (an empty run id gets one)."""
    ctx = bound if bound is not None else current_context()
    if ctx is None:
        return ToolContext(run_id=new_id("run_"), call_id=new_id("call_"))
    if not ctx.run_id:
        return dataclasses.replace(ctx, run_id=new_id("run_"))
    return ctx


@contextmanager
def use_context(ctx: ToolContext) -> Iterator[ToolContext]:
    """Make `ctx` current inside the block; child tasks read it, writes never flow back up."""
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
