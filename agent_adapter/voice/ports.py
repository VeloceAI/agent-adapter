"""What a voice agent needs from the outside: something that speaks text, and can stop.

Two optional speed ports sit beside the media ports: `TurnSignals`, a speech-to-text
stream that says early when the provider thinks the turn ended, and `StreamingTTS`,
text-to-speech that takes text while the model is still writing it.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class Speaker(Protocol):
    """Speaks text aloud: LiveKit, Pipecat, a phone line, a browser, any TTS."""

    async def say(self, text: str) -> None:
        """Speak `text`; return when it has played (or was stopped)."""
        ...

    async def stop(self) -> None:
        """Cut the audio now (the caller started talking)."""
        ...


class CallableSpeaker:
    """A `Speaker` from plain functions: `say(text)` and an optional `stop()`, sync or async."""

    def __init__(
        self,
        say: Callable[[str], Awaitable[None] | None],
        stop: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._say = say
        self._stop = stop

    async def say(self, text: str) -> None:
        await _maybe_await(self._say(text))

    async def stop(self) -> None:
        if self._stop is not None:
            await _maybe_await(self._stop())


TurnEvent = Literal["eager_end", "resumed", "end"]


@dataclass(frozen=True)
class TurnSignal:
    """The provider's view of the turn: `eager_end` (probably done: start a reply now,
    but keep it unspoken), `resumed` (the caller went on: drop that reply) or `end`
    (done). `text` is the transcript of the turn so far."""

    event: TurnEvent
    text: str
    confidence: float | None = None


@runtime_checkable
class TurnSignals(Protocol):
    """An `STTStream` that also reports `TurnSignal`s as they arrive.

    The callback runs on the event loop and must return at once. A stream without this
    method still works; the session then decides the end of the turn on its own.
    """

    def on_turn(self, callback: Callable[[TurnSignal], None]) -> None: ...


@dataclass(frozen=True)
class WordTime:
    """A spoken word and when it plays, in seconds from the start of the stream's audio."""

    word: str
    start: float
    end: float


class TTSStream(Protocol):
    """One reply spoken while its text is still being written.

    `push` text as it comes (`flush=True` speaks what is buffered now, at the end of a
    clause), then `end()`. `audio()` yields 16-bit mono PCM at the TTS `sample_rate`.
    `words` fills with `WordTime`s when the provider sends timestamps.
    """

    words: list[WordTime]

    async def push(self, text: str, *, flush: bool = False) -> None: ...

    async def end(self) -> None: ...

    def audio(self) -> AsyncIterator[bytes]: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class StreamingTTS(Protocol):
    """A `TTS` that can also take text in pieces: `open()` starts one `TTSStream`."""

    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

    def open(self) -> TTSStream: ...


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


__all__ = [  # Speaker and CallableSpeaker are documented at agent_adapter.voice
    "StreamingTTS",
    "TTSStream",
    "TurnEvent",
    "TurnSignal",
    "TurnSignals",
    "WordTime",
]
