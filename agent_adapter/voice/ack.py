"""`Acknowledgements`: a short pre-rendered line while a slow reply is on its way."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable


class Acknowledgements:
    """Plays a short line ("Mm-hm.") when the reply will be slow, and never before a fast one.

    The agent learns how long its replies take to start (a moving average from `expect_ms`).
    When a turn is handed over and the reply is expected later than `fast_ms`, a line plays at
    once; otherwise it waits `fast_ms` and plays only if the reply has still not started. A
    reply that is already waiting (a kept speculation, a cached answer) gets none.

    Lines play only once rendered at call start (`VoiceAgent.warm_up`, which a `VoiceSession`
    runs), so an acknowledgement is never slower than the reply it stands in for.
    """

    def __init__(
        self,
        lines: Iterable[str] = ("Mm-hm.", "Okay.", "Sure."),
        *,
        fast_ms: float = 300.0,
        expect_ms: float = 700.0,
        smoothing: float = 0.3,
    ) -> None:
        self.lines = tuple(line.strip() for line in lines if line.strip())
        if not self.lines:
            raise ValueError("Acknowledgements needs at least one line")
        if fast_ms < 0 or expect_ms < 0 or not 0 < smoothing <= 1:
            raise ValueError("fast_ms and expect_ms >= 0; 0 < smoothing <= 1")
        self.fast_ms = fast_ms
        self.expected_ms = expect_ms
        self.smoothing = smoothing
        self.rendered: set[str] = set()
        self._next = 0

    def learn(self, reply_ms: float) -> None:
        """A reply started `reply_ms` after its thinking began."""
        self.expected_ms += self.smoothing * (reply_ms - self.expected_ms)

    def delay_ms(self, thought_ms: float) -> float:
        """How long to wait before acknowledging, `thought_ms` into the thinking."""
        return 0.0 if self.expected_ms - thought_ms > self.fast_ms else self.fast_ms

    def pick(self) -> str | None:
        """The next rendered line, in turn; None before any is rendered."""
        for _ in range(len(self.lines)):
            line = self.lines[self._next % len(self.lines)]
            self._next += 1
            if line in self.rendered:
                return line
        return None

    async def run(
        self, thought_ms: float, ready: Callable[[], bool], say: Callable[[str], None]
    ) -> str | None:
        """Wait as planned, then `say` a line unless the reply is ready; returns the line."""
        if ready() or not self.rendered:
            return None
        delay = self.delay_ms(thought_ms)
        if delay:
            await asyncio.sleep(delay / 1000)
        if ready():
            return None
        line = self.pick()
        if line is not None:
            say(line)
        return line


__all__ = ["Acknowledgements"]
