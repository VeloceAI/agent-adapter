"""Shared by the LiveKit and Pipecat voice bridges: caller events in order, background speech."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import log_failure

if TYPE_CHECKING:
    from agent_adapter.voice import VoiceAgent

logger = logging.getLogger(__name__)
_BACKGROUND: set[asyncio.Task[Any]] = set()


def spawn(work: Coroutine[Any, Any, Any]) -> None:
    """Run `work` in the background; keep a reference so it is not collected mid-flight."""
    task = asyncio.ensure_future(work)
    _BACKGROUND.add(task)
    task.add_done_callback(_finished)


def _finished(task: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(task)
    error = None if task.cancelled() else task.exception()
    if error is not None:
        log_failure(logger, logging.WARNING, error, "voice bridge task failed")


class EventPump:
    """Hands a framework's caller events to a `VoiceAgent` one at a time, in order.

    Framework callbacks must return at once, but a cut can wait for a booking to finish;
    the pump queues events so the framework's loop never waits and order is kept.
    """

    def __init__(self, agent: VoiceAgent) -> None:
        self.agent = agent
        self._queue: asyncio.Queue[str | None] | None = None

    def started_speaking(self) -> None:
        self._put(None)

    def said(self, text: str) -> None:
        if text.strip():
            self._put(text)

    def _put(self, event: str | None) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
            spawn(self._drain(self._queue))
        self._queue.put_nowait(event)

    async def _drain(self, queue: asyncio.Queue[str | None]) -> None:
        while True:
            event = await queue.get()
            try:
                if event is None:
                    await self.agent.user_started_speaking()
                else:
                    await self.agent.user_said(event)
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "voice agent event failed")
