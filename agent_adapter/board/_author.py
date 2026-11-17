"""Who is posting now: the running agent, else the running flow node."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from agent_adapter.core.context import NODE_KEY
from agent_adapter.runtime.current import current_context

_AUTHOR: ContextVar[str] = ContextVar("aa_board_author", default="")


@contextmanager
def posting_as(name: str) -> Iterator[None]:
    """Board posts inside the block default to `by=name` (agent tools set their name)."""
    token = _AUTHOR.set(name)
    try:
        yield
    finally:
        _AUTHOR.reset(token)


def current_author() -> str:
    """The innermost `posting_as` name, else the current flow node's name, else ""."""
    name = _AUTHOR.get()
    if name:
        return name
    ctx = current_context()
    node = ctx.extras.get(NODE_KEY) if ctx is not None else None
    return node if isinstance(node, str) else ""
