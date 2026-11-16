"""Local speech models with permissive licences: Moonshine speech-to-text and Kokoro voices.

`moonshine-voice` is MIT; its Kokoro voices are Apache-2.0. Models download once (about
360 MB; set `MOONSHINE_VOICE_CACHE` to choose where) and then run offline on the CPU.

Both stream: speech is transcribed while the caller talks, so the text is ready as the
turn ends, and a line starts playing from its first chunk, not after the whole line.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
from array import array
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice.pcm import Resampler, bytes_of, samples_of


def _moonshine() -> Any:
    try:
        import moonshine_voice
    except ImportError as exc:
        raise MissingDependency(
            'Moonshine needs moonshine-voice: pip install "agent-adapter[voice-local]"'
        ) from exc
    return moonshine_voice


def _floats(pcm: bytes) -> list[float]:
    return [s / 32_768 for s in samples_of(pcm)]


def _add_audio(stream: Any, *, audio: list[float], rate: int) -> None:
    stream.add_audio(audio, rate)


def _text(transcript: Any) -> str:
    return " ".join(line.text for line in transcript.lines).strip() if transcript else ""


class MoonshineSTT:
    """Speech-to-text on the CPU; `model` is a Moonshine architecture name.

    A stream decodes what it has heard every `update_interval` seconds of audio. Only the
    final text is used, so a longer interval saves CPU; past about 1.5 s the final pass has
    more left to decode and the reply starts later.
    """

    def __init__(
        self, model: str = "medium-streaming", *, language: str = "en", update_interval: float = 1.0
    ) -> None:
        mv = _moonshine()
        path, arch = mv.get_model_for_language(language, mv.string_to_model_arch(model))
        self._transcriber = mv.Transcriber(model_path=path, model_arch=arch)
        self.update_interval = update_interval
        self._lock = threading.Lock()  # one engine call at a time

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return await asyncio.to_thread(self._batch, pcm, sample_rate)

    def open(self, sample_rate: int) -> _MoonshineStream:
        return _MoonshineStream(self, sample_rate)

    def _batch(self, pcm: bytes, sample_rate: int) -> str:
        with self._lock:
            return _text(self._transcriber.transcribe_without_streaming(_floats(pcm), sample_rate))


class _MoonshineStream:
    """Feeds audio to a Moonshine stream from a task, one engine call at a time."""

    def __init__(self, stt: MoonshineSTT, sample_rate: int) -> None:
        self._stt = stt
        self._rate = sample_rate
        self._pending = bytearray()
        self._wake = asyncio.Event()
        self._ending = False
        self._stream: Any = None
        self._task = asyncio.create_task(self._pump())

    def feed(self, pcm: bytes) -> None:
        self._pending += pcm
        self._wake.set()

    async def finish(self) -> str:
        self._ending = True
        self._wake.set()
        await self._task
        return await asyncio.to_thread(self._call, lambda s: _text(s.stop()))

    async def cancel(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._call, lambda s: s.stop())

    async def _pump(self) -> None:
        self._stream = await asyncio.to_thread(self._create)
        while True:
            await self._wake.wait()
            self._wake.clear()
            chunk, self._pending = bytes(self._pending), bytearray()
            if chunk:
                add = functools.partial(_add_audio, audio=_floats(chunk), rate=self._rate)
                await asyncio.to_thread(self._call, add)
            if self._ending and not self._pending:
                return

    def _create(self) -> Any:
        with self._stt._lock:
            stream = self._stt._transcriber.create_stream(update_interval=self._stt.update_interval)
            stream.start()
            return stream

    def _call(self, fn: Any) -> Any:
        with self._stt._lock:
            return fn(self._stream)


class MoonshineTTS:
    """Text-to-speech with a Kokoro voice at 24 kHz, streamed chunk by chunk."""

    sample_rate = 24_000

    def __init__(self, voice: str = "kokoro_af_heart", *, language: str = "en_us") -> None:
        self._tts = _moonshine().TextToSpeech().language(language).voice(voice).load()
        self._lock = threading.Lock()  # one synthesis at a time on the shared model

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        stop = threading.Event()

        def run() -> None:
            try:
                with self._lock:
                    for chunk in self._tts.stream(text):
                        if stop.is_set():
                            self._tts.cancel_stream()
                            break
                        pcm = self._pcm(chunk.samples, chunk.sample_rate)
                        loop.call_soon_threadsafe(chunks.put_nowait, pcm)
            finally:
                with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                    loop.call_soon_threadsafe(chunks.put_nowait, None)

        worker = asyncio.ensure_future(asyncio.to_thread(run))
        try:
            while (pcm := await chunks.get()) is not None:
                yield pcm
            await worker
        finally:
            stop.set()

    def _pcm(self, samples: Sequence[float], rate: int) -> bytes:
        clipped = (max(-1.0, min(1.0, float(s))) for s in samples)
        pcm = bytes_of(array("h", (int(s * 32_767) for s in clipped)))
        return Resampler(int(rate), self.sample_rate).feed(pcm)


__all__ = ["MoonshineSTT", "MoonshineTTS"]
