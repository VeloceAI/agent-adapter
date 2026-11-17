"""`Player`: the speaker of our own pipeline, with no gaps and an exact record of what played."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict, deque

from agent_adapter.core.errors import log_failure
from agent_adapter.voice.media import TTS
from agent_adapter.voice.pcm import Resampler
from agent_adapter.voice.playout import PlayoutBuffer, Segment

logger = logging.getLogger(__name__)


class _Synth:
    """One line being synthesised; its chunks queue up until the line is spoken."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None


class Player:
    """Speaks lines through a `PlayoutBuffer` and knows how much of each the caller heard.

    `prepare(text)` starts synthesis early, so the next line is ready when this one ends.
    After `stop()`, `heard_of_cut` holds the words of the cut line that played (estimated
    from the share of its audio that played), so the history keeps only what was heard.
    The last `cache_lines` short lines are kept as audio, so a repeated line ("One
    moment.") plays at once; 0 turns that off.
    """

    def __init__(self, tts: TTS, buffer: PlayoutBuffer, *, cache_lines: int = 64) -> None:
        self.tts = tts
        self.buffer = buffer
        self.cache_lines = cache_lines
        self.heard_of_cut = ""
        self._cache: OrderedDict[str, list[bytes]] = OrderedDict()
        self._ready: deque[_Synth] = deque()
        self._order = asyncio.Lock()  # lines synthesise one at a time, in order
        self._current: tuple[str, Segment] | None = None

    @property
    def speaking(self) -> bool:
        """Audio is queued or a line is being spoken."""
        live = self._current is not None and not self._current[1].done.done()
        return live or self.buffer.busy

    def prepare(self, text: str) -> None:
        """Start synthesising `text` now; `say(text)` will use it."""
        self._ready.append(self._start(text))

    async def say(self, text: str) -> None:
        """Speak `text`; return when it has played, or was cut."""
        synth = self._take(text)
        segment = self.buffer.open()
        self._current = (text, segment)
        try:
            while (chunk := await synth.chunks.get()) is not None:
                self.buffer.write(segment, chunk)
            self.buffer.finish(segment)
            await segment.done
        except asyncio.CancelledError:
            if synth.task is not None:
                synth.task.cancel()
            raise  # `_current` stays, so `stop()` can tell how much of this line played
        if self._current is not None and self._current[1] is segment:
            self._current = None

    def pause(self) -> None:
        """Hold playback (the caller may be cutting in)."""
        self.buffer.pause()

    def resume(self) -> None:
        self.buffer.resume()

    async def stop(self) -> None:
        """Cut now: drop queued audio and pending synthesis; record what was heard."""
        self.heard_of_cut = ""
        if self._current is not None:
            text, segment = self._current
            self.heard_of_cut = _words_heard(text, segment.heard)
            self._current = None
        self.buffer.clear()
        ready, self._ready = list(self._ready), deque()
        for synth in ready:
            if synth.task is not None:
                synth.task.cancel()
        for synth in ready:
            if synth.task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await synth.task

    def _take(self, text: str) -> _Synth:
        for synth in self._ready:
            if synth.text == text:
                self._ready.remove(synth)
                return synth
        return self._start(text)

    def _start(self, text: str) -> _Synth:
        synth = _Synth(text)
        synth.task = asyncio.create_task(self._synthesise(synth))
        return synth

    async def warm(self, lines: list[str]) -> None:
        """Synthesise `lines` into the cache now, so they play at once later."""
        for text in lines:
            synth = self._start(text)
            if synth.task is not None:
                await synth.task

    async def _synthesise(self, synth: _Synth) -> None:
        cached = self._cache.get(synth.text)
        if cached is not None:
            self._cache.move_to_end(synth.text)
            for chunk in cached:
                synth.chunks.put_nowait(chunk)
            synth.chunks.put_nowait(None)
            return
        resampler = Resampler(self.tts.sample_rate, self.buffer.sample_rate)
        audio: list[bytes] = []
        try:
            async with self._order:
                async for chunk in self.tts.synthesize(synth.text):
                    audio.append(resampler.feed(chunk))
                    synth.chunks.put_nowait(audio[-1])
            self._remember(synth.text, audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "text-to-speech failed")
        finally:
            synth.chunks.put_nowait(None)

    def _remember(self, text: str, audio: list[bytes]) -> None:
        if self.cache_lines <= 0 or len(text) > _CACHE_CHARS or not audio:
            return
        self._cache[text] = audio
        while len(self._cache) > self.cache_lines:
            self._cache.popitem(last=False)


_CACHE_CHARS = 120  # long lines rarely repeat word for word


def _words_heard(text: str, share: float) -> str:
    words = text.split()
    return " ".join(words[: int(len(words) * share)])


__all__ = ["Player"]
