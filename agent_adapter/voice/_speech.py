"""Speaks model text sentence by sentence, and knows which sentences really played."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from agent_adapter.core.errors import log_failure
from agent_adapter.voice.ports import Speaker

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SpeechQueue:
    """Feeds `speaker` one sentence at a time; `stop` cuts it and drops what is left."""

    def __init__(self, speaker: Speaker) -> None:
        self._speaker = speaker
        self._buffer = ""
        self._queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.played: list[str] = []  # model sentences fully spoken (not progress lines)

    def feed(self, chunk: str) -> None:
        """Add streamed model text; each finished sentence is queued to speak."""
        self._buffer += chunk
        *done, self._buffer = _SENTENCE_END.split(self._buffer)
        for sentence in done:
            self.put(sentence)

    def flush(self) -> None:
        """Queue whatever text is left after the model finished."""
        rest, self._buffer = self._buffer.strip(), ""
        if rest:
            self.put(rest)

    def put(self, text: str, *, record: bool = True) -> None:
        """Queue `text`; `record=False` for lines that are not model text (progress)."""
        if not text.strip():
            return
        self._queue.put_nowait((text.strip(), record))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._speak())

    async def drain(self) -> None:
        """Wait until everything queued has played."""
        await self._queue.join()

    async def stop(self) -> None:
        """Cut speech now: stop the speaker and drop the queue and the unfinished text."""
        self._buffer = ""
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            await self._speaker.stop()
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "speaker.stop failed")

    def take_played(self) -> str:
        """The model text that really played since the last call, then forget it."""
        text, self.played = " ".join(self.played), []
        return text

    async def _speak(self) -> None:
        while True:
            text, record = await self._queue.get()
            try:
                await self._speaker.say(text)
                if record:
                    self.played.append(text)
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "speaker.say failed")
                self._queue.task_done()
            else:
                self._queue.task_done()
