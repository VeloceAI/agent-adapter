"""Speaks model text sentence by sentence, and knows which sentences really played."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from agent_adapter.core.errors import log_failure
from agent_adapter.voice.ports import Speaker

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Line:
    text: str
    record: bool  # model text: kept in `played`
    hold: bool = False  # the caller cannot cut it
    heard: Callable[[str], None] | None = None  # told what of it played, once


class SpeechQueue:
    """Feeds `speaker` one sentence at a time; `stop` cuts it and drops what is left."""

    def __init__(self, speaker: Speaker) -> None:
        self._speaker = speaker
        self._buffer = ""
        self._queue: asyncio.Queue[_Line] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.played: list[str] = []  # model sentences fully spoken (not progress lines)
        self._current: _Line | None = None
        self._holds = 0

    @property
    def holding(self) -> bool:
        """A line the caller cannot cut is queued or playing."""
        return self._holds > 0

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

    def put(
        self,
        text: str,
        *,
        record: bool = True,
        hold: bool = False,
        heard: Callable[[str], None] | None = None,
    ) -> None:
        """Queue `text`; `record=False` for lines that are not model text (progress)."""
        if not text.strip():
            return
        self._holds += hold
        self._queue.put_nowait(_Line(text.strip(), record, hold, heard))
        prepare = getattr(self._speaker, "prepare", None)
        if callable(prepare):  # a speaker that can synthesise ahead (our Player)
            prepare(text.strip())
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._speak())

    async def drain(self) -> None:
        """Wait until everything queued has played."""
        await self._queue.join()

    async def stop(self) -> None:
        """Cut speech now: stop the speaker and drop the queue and the unfinished text."""
        self._buffer = ""
        current, self._current = self._current, None
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._holds = 0
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            await self._speaker.stop()
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "speaker.stop failed")
            return
        heard = getattr(self._speaker, "heard_of_cut", "")  # the words of a cut line that played
        if current is None or not isinstance(heard, str) or not heard:
            return
        if current.record:
            self.played.append(heard)
        if current.heard is not None:
            current.heard(heard)

    def take_played(self) -> str:
        """The model text that really played since the last call, then forget it."""
        text, self.played = " ".join(self.played), []
        return text

    async def _speak(self) -> None:
        while True:
            line = await self._queue.get()
            self._current = line
            try:
                await self._speaker.say(line.text)
                if line.record:
                    self.played.append(line.text)
                if line.heard is not None:
                    line.heard(line.text)
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "speaker.say failed")
                self._queue.task_done()
            else:
                self._queue.task_done()
            finally:
                if line.hold and self._holds > 0:
                    self._holds -= 1
            self._current = None
