"""Small checks on user callables, shared by every package (a base module)."""

from __future__ import annotations

import inspect


def is_async(func: object) -> bool:
    """True for coroutine functions and objects whose `__call__` is one."""
    return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(type(func).__call__)
