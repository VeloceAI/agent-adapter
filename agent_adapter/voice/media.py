"""The ports of our own voice pipeline: audio transports, speech-to-text and text-to-speech.

Every port is a small protocol, so any provider fits: a local model, a hosted API, or a
function you already have (wrap it with `CallableSTT` or `CallableTTS`).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_adapter.voice.playout import PlayoutBuffer


@runtime_checkable
class AudioTransport(Protocol):
    """Where the call's audio comes from and goes to: a sound card, a socket, a phone line.

    `frames()` yields the caller's audio as 16-bit mono PCM at `in_rate`. `start(buffer)`
    makes the transport drain `buffer` (PCM16 at `out_rate`) in real time.
    """

    in_rate: int
    out_rate: int

    def frames(self) -> AsyncIterator[bytes]: ...

    async def start(self, buffer: PlayoutBuffer) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class STT(Protocol):
    """Speech-to-text for one finished turn of 16-bit mono PCM."""

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...


class STTStream(Protocol):
    """One turn transcribed while the caller talks; `finish` returns the final text."""

    def feed(self, pcm: bytes) -> None:
        """Add audio; must return at once (do the work in a thread or a task)."""
        ...

    async def finish(self) -> str: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class StreamingSTT(STT, Protocol):
    """An `STT` that can also transcribe as the caller speaks: at the end of the turn
    only the last moment is left to do, instead of the whole turn."""

    def open(self, sample_rate: int) -> STTStream: ...


@runtime_checkable
class TTS(Protocol):
    """Text-to-speech: 16-bit mono PCM chunks at `sample_rate`, as soon as each is ready."""

    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class CallableSTT:
    """An `STT` from `fn(pcm, sample_rate) -> str`; a sync function runs in a thread."""

    def __init__(self, fn: Callable[[bytes, int], str | Awaitable[str]]) -> None:
        self._fn = fn

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if inspect.iscoroutinefunction(self._fn):
            text = await self._fn(pcm, sample_rate)
        else:
            text = await asyncio.to_thread(self._fn, pcm, sample_rate)
            if inspect.isawaitable(text):
                text = await text
        return str(text).strip()


class CallableTTS:
    """A `TTS` from `fn(text)` returning PCM16 bytes, or an iterable or async iterable of
    chunks; a sync function runs in a thread."""

    def __init__(self, fn: Callable[[str], Any], *, sample_rate: int) -> None:
        self._fn = fn
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if inspect.iscoroutinefunction(self._fn):
            out = await self._fn(text)
        elif inspect.isasyncgenfunction(self._fn):
            out = self._fn(text)
        else:
            out = await asyncio.to_thread(self._fn, text)
        if isinstance(out, bytes | bytearray):
            yield bytes(out)
        elif hasattr(out, "__aiter__"):
            async for chunk in out:
                yield bytes(chunk)
        else:
            for chunk in _iterable(out):
                yield bytes(chunk)


def _iterable(out: Any) -> Iterable[Any]:
    if out is None:
        return ()
    if not isinstance(out, Iterable):
        raise TypeError(f"TTS function returned {type(out).__name__}, not bytes or chunks")
    return out


__all__ = [
    "STT",
    "TTS",
    "AudioTransport",
    "CallableSTT",
    "CallableTTS",
    "STTStream",
    "StreamingSTT",
]
