"""Offline fakes for voice agents: test interruptions and approvals with no audio and no keys."""

from __future__ import annotations

import asyncio

__all__ = ["RecordingSpeaker", "settle"]


class RecordingSpeaker:
    """A `Speaker` that records what it said; with `hold=True` each line waits for `release`."""

    def __init__(self, *, hold: bool = False) -> None:
        self.started: list[str] = []  # lines it began to say
        self.said: list[str] = []  # lines that played to the end
        self.stops = 0
        self._hold = hold
        self._allowance = asyncio.Semaphore(0)

    def release(self, lines: int = 1) -> None:
        """Let `lines` more held lines play to the end."""
        for _ in range(lines):
            self._allowance.release()

    async def say(self, text: str) -> None:
        self.started.append(text)
        if self._hold:
            await self._allowance.acquire()
        self.said.append(text)

    async def stop(self) -> None:
        self.stops += 1


async def settle(rounds: int = 50) -> None:
    """Let every ready task run (a few event-loop turns)."""
    for _ in range(rounds):
        await asyncio.sleep(0)
