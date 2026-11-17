"""A `TalkerStream` around a frame-in, frame-out model loop that runs in one thread."""

from __future__ import annotations

import asyncio
import contextlib
import queue
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from agent_adapter.voice._open_models import flat, floats, pcm16
from agent_adapter.voice.pcm import Resampler, rms
from agent_adapter.voice.talker import Cue

Frames = Callable[[int], Iterator[list[float]]]
Out = Callable[[Any, str], None]


class DuplexStream:
    """Feeds the caller's audio to `run(frames, out)` in a worker thread.

    `frames(n)` yields the caller's audio at `model_rate` in blocks of `n` samples until the
    stream closes; `run` calls `out(audio, text)` for every frame the model speaks. Quiet
    frames are dropped, except `hangover` frames after speech so a cue does not end clipped.
    """

    def __init__(
        self,
        run: Callable[[Frames, Out], None],
        *,
        in_rate: int,
        model_rate: int,
        min_level: float = 0.01,
        hangover: int = 2,
    ) -> None:
        self._audio: queue.Queue[bytes | None] = queue.Queue()
        self._resample = Resampler(in_rate, model_rate) if in_rate != model_rate else None
        self._cues: asyncio.Queue[Cue | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._min_level, self._hangover, self._quiet = min_level, hangover, hangover
        self._task = asyncio.ensure_future(asyncio.to_thread(self._work, run))

    def hear(self, pcm: bytes) -> None:
        self._audio.put(self._resample.feed(pcm) if self._resample else pcm)

    async def cues(self) -> AsyncIterator[Cue]:
        while (cue := await self._cues.get()) is not None:
            yield cue

    async def close(self) -> None:
        self._audio.put(None)
        with contextlib.suppress(Exception):
            await self._task

    def _work(self, run: Callable[[Frames, Out], None]) -> None:
        try:
            run(self._frames, self._out)
        finally:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                self._loop.call_soon_threadsafe(self._cues.put_nowait, None)

    def _frames(self, size: int) -> Iterator[list[float]]:
        pending: list[float] = []
        while (pcm := self._audio.get()) is not None:
            pending += floats(pcm)
            while len(pending) >= size:
                yield pending[:size]
                del pending[:size]

    def _out(self, audio: Any, text: str) -> None:
        pcm = pcm16(flat(audio))
        if rms(pcm) >= self._min_level or text.strip():
            self._quiet = 0
        elif self._quiet < self._hangover:
            self._quiet += 1
        else:
            return
        self._loop.call_soon_threadsafe(self._cues.put_nowait, Cue(pcm, text))
