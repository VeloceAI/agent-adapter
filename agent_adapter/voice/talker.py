"""The talker port: a full-duplex speech model that keeps the caller company while the
agent thinks.

A talker (Haru-4B, Moshi) hears the caller frame by frame and answers in its own voice at
the same time: "mm-hm", "right", "let me check". It gives no facts and runs no tools. The
agent (the thinker) still owns the turn: it calls tools, decides and speaks the answer.
Play talker cues only while the agent is silent, and drop them once it speaks.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Cue:
    """A short piece of the talker's speech: PCM16 mono at the talker's `sample_rate`,
    and the text it said, when the model gives one."""

    audio: bytes
    text: str = ""


class TalkerStream(Protocol):
    """One call's duplex link to a talker."""

    def hear(self, pcm: bytes) -> None:
        """Add the caller's audio (PCM16 at the rate given to `open`); returns at once."""
        ...

    def cues(self) -> AsyncIterator[Cue]:
        """The talker's speech as it comes; silence between cues is left out."""
        ...

    async def close(self) -> None: ...


@runtime_checkable
class Talker(Protocol):
    """A full-duplex model. `warm()` loads it ahead of the call; `open(rate)` starts one."""

    sample_rate: int

    async def warm(self) -> None: ...

    def open(self, sample_rate: int) -> TalkerStream: ...


async def backchannel(
    stream: TalkerStream,
    play: Callable[[bytes], Awaitable[None] | None],
    *,
    may_speak: Callable[[], bool],
) -> None:
    """Play each cue while `may_speak()` is true (the agent is not speaking); drop the rest.

    Run it as a task next to the call; it ends when the stream closes.
    """
    async for cue in stream.cues():
        if may_speak():
            result = play(cue.audio)
            if inspect.isawaitable(result):
                await result


__all__ = ["Cue", "Talker", "TalkerStream", "backchannel"]
