"""The process-wide default Executor, and the shortcuts that run tools on it."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.ports import EventSink
from agent_adapter.core.registry import Registry
from agent_adapter.exec._sugar import ToolRef, blocking
from agent_adapter.exec.executor import Executor
from agent_adapter.runtime._fork import ForkSafeLock

logger = logging.getLogger(__name__)

R = TypeVar("R")

_default: Executor | None = None
_default_sinks: list[EventSink] = []  # follow every default executor (the run recorder)
_lock = ForkSafeLock()


def default_executor() -> Executor:
    """The default Executor, created on first use, on the default registry."""
    global _default
    with _lock:
        if _default is None:
            _default = Executor()
            _attach_default_sinks(_default)
            logger.debug("created the default executor")
        return _default


def set_default_executor(executor: Executor) -> None:
    """Make `executor` the default for every integration that is not given one.

    Raises:
        ToolUsageError: `executor` is not an Executor.
    """
    global _default
    if not isinstance(executor, Executor):
        raise ToolUsageError(f"set_default_executor needs an Executor, got {executor!r}")
    with _lock:
        if _default is not None and _default is not executor:
            _detach_default_sinks(_default)  # they follow the default, not the old executor
        _attach_default_sinks(executor)
        _default = executor


def reset_default_executor() -> None:
    """Drop the default Executor and its default sinks; the next use starts fresh."""
    global _default
    with _lock:
        if _default is not None:
            _detach_default_sinks(_default)
        _default = None
        _default_sinks.clear()
    logger.debug("reset the default executor")


def add_default_sink(sink: EventSink) -> Executor:
    """Send the default executor's events to `sink`, now and after `configure` replaces it.

    Returns the current default executor.
    """
    executor = default_executor()
    with _lock:
        if not any(sink is known for known in _default_sinks):
            _default_sinks.append(sink)
        _attach_default_sinks(_default if _default is not None else executor)
    return executor


def remove_default_sink(sink: EventSink) -> bool:
    """Stop sending the default executor's events to `sink`; False when it was not added."""
    with _lock:
        kept = [known for known in _default_sinks if known is not sink]
        if len(kept) == len(_default_sinks):
            return False
        _default_sinks[:] = kept
        if _default is not None:
            _default.remove_sink(sink)
        return True


def _attach_default_sinks(executor: Executor) -> None:
    for sink in _default_sinks:
        executor.add_sink(sink)


def _detach_default_sinks(executor: Executor) -> None:
    for sink in _default_sinks:
        executor.remove_sink(sink)


def runner_for(executor: R | None) -> R | Executor:
    """`executor` when given, else the default Executor."""
    return executor if executor is not None else default_executor()


def configure(registry: Registry | None = None, **options: Any) -> Executor:
    """Make `Executor(registry, **options)` the default (`call`, integrations) and return it.

    Raises:
        TypeError: an unknown option name.
        ToolUsageError: a bad option value (see `Executor`).
    """
    executor = Executor(registry, **options)
    set_default_executor(executor)
    logger.debug("configured the default executor: %s", sorted(options))
    return executor


async def acall(tool: ToolRef, /, **args: Any) -> Any:
    """Run `tool` (a function, spec or name) on the default executor and return its value."""
    return await default_executor().run(tool, args)


def call(tool: ToolRef, /, **args: Any) -> Any:
    """`acall` from sync code, on the process's shared loop; raises like `Executor.run`."""
    executor = default_executor()
    return blocking(lambda: executor.run(tool, args), None)
