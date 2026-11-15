"""Feed a `VoiceAgent` from any transport's callbacks: in order, and without blocking them."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from agent_adapter.core.errors import log_failure

if TYPE_CHECKING:
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)


class EventPump:
    """Hands caller events to a `VoiceAgent` one at a time, in order.

    Transport callbacks must return at once, but a cut can wait for a booking to finish and
    an approval waits for the caller's next words; the pump queues events so the transport
    never waits (and never deadlocks on its own next message) and order is kept.
    """

    def __init__(self, agent: VoiceAgent) -> None:
        self.agent = agent
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def started_speaking(self) -> None:
        """The caller started talking (VAD)."""
        self._put(None)

    def said(self, text: str) -> None:
        """A final transcript; blank text is ignored."""
        if text.strip():
            self._put(text)

    async def idle(self) -> None:
        """Wait until every queued event has been handed over."""
        await self._queue.join()

    async def aclose(self) -> None:
        """Stop handing events over (queued ones are dropped)."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _put(self, event: str | None) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._drain())
        self._queue.put_nowait(event)

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    await self.agent.user_started_speaking()
                else:
                    await self.agent.user_said(event)
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "voice agent event failed")
            finally:
                self._queue.task_done()
