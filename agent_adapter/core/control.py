"""Control-flow exceptions: never retried, never hooked, never turned into results.

Base layer so that `approval` and `exec` share one registry.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from agent_adapter.core.errors import ToolUsageError

logger = logging.getLogger(__name__)

# Framework signals that unwind through tool calls (e.g. a host's interrupt type).
_registered: tuple[type[BaseException], ...] = ()
_lock = threading.Lock()

GRAPH_BUBBLE_UP = "GraphBubbleUp"  # LangGraph's interrupt base; matched by name, never imported


def register_control_flow(*types: type[BaseException]) -> None:
    """Treat `types` (and subclasses) as control flow in the executor and the approval ports.

    Raises:
        ToolUsageError: no types given, or one is not a BaseException subclass.
    """
    global _registered
    if not types:
        raise ToolUsageError("register_control_flow needs at least one exception type")
    bad = [t for t in types if not (isinstance(t, type) and issubclass(t, BaseException))]
    if bad:
        raise ToolUsageError(f"register_control_flow takes BaseException subclasses, got {bad!r}")
    with _lock:
        _registered = (*_registered, *(t for t in types if t not in _registered))
    logger.debug("control-flow types: %s", [t.__qualname__ for t in _registered])


def is_control_flow(exc: BaseException) -> bool:
    """True for CancelledError, a registered type, a class named GraphBubbleUp in the MRO,
    or an exception group holding any of them."""
    return control_flow_member(exc) is not None


def control_flow_member(exc: BaseException) -> BaseException | None:
    """`exc` if it is control flow, else the first control-flow leaf of a group, else None."""
    if isinstance(exc, (asyncio.CancelledError, *_registered)) or any(
        cls.__name__ == GRAPH_BUBBLE_UP for cls in type(exc).__mro__
    ):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for member in exc.exceptions:
            found = control_flow_member(member)
            if found is not None:
                return found
    return None
