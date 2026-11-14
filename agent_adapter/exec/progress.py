"""Short updates from a running tool ("One moment, checking that"), spoken or shown by the host."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from agent_adapter.core.errors import log_failure
from agent_adapter.runtime.current import current_context
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

ProgressListener = Callable[[str], None]
_LISTENER: ContextVar[ProgressListener | None] = ContextVar("aa_progress", default=None)


def progress(text: str) -> None:
    """Tell the host what a slow tool is doing; never a result, never sent to the model.

    The text goes to the listener set with `listen_progress` (a voice agent speaks it); the
    event (`tool.progress`) carries only the call id.
    """
    ctx = current_context()
    if ctx is not None:
        emit(ctx.sink, make_event(EventType.TOOL_PROGRESS, ctx))
    listener = _LISTENER.get()
    if listener is None:
        return
    try:
        listener(text)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "progress listener failed")


@contextmanager
def listen_progress(listener: ProgressListener) -> Iterator[None]:
    """Send `progress(...)` texts of the tools run inside this block to `listener`."""
    token = _LISTENER.set(listener)
    try:
        yield
    finally:
        _LISTENER.reset(token)
