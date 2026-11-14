"""What a voice agent needs from the outside: something that speaks text, and can stop."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


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


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value
