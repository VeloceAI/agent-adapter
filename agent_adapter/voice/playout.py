"""Audio waiting to play, drained in real time, so we always know what the caller heard.

A transport drains the `PlayoutBuffer`: a sound card callback, a WebRTC track, or a
`Pacer` that sends it over a socket at the pace it plays. Because nothing is handed over
before its time, a pause or a cut takes effect within one frame, and the count of bytes
pulled for each line is what was heard of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Segment:
    """One line of speech in the buffer."""

    id: int
    done: asyncio.Future[None]
    written: int = 0  # bytes written
    played: int = 0  # bytes pulled by the transport
    finished: bool = False  # no more audio will be written
    started_at: float | None = None  # monotonic time its first byte was pulled
    ambient: bool = False  # a background sound, not speech: it never counts as speaking

    @property
    def heard(self) -> float:
        """The share of the written audio that played, 0 to 1."""
        return self.played / self.written if self.written else 0.0


@dataclass(eq=False, slots=True)
class _Chunk:
    segment: Segment
    data: bytes
    offset: int = 0


@dataclass(eq=False)
class PlayoutBuffer:
    """Thread-safe queue of PCM16 mono audio at `sample_rate`, pulled in real time.

    With `confirm_playout`, a line counts as played only when the transport calls
    `confirm(id)` (a phone network reporting a mark), not when its last byte is pulled.
    """

    sample_rate: int
    confirm_playout: bool = False
    on_clear: list[Callable[[], None]] = field(default_factory=list)
    on_start: list[Callable[[Segment], None]] = field(default_factory=list)
    on_drained: list[Callable[[Segment], None]] = field(default_factory=list)
    on_pull: list[Callable[[bytes], None]] = field(default_factory=list)  # may be any thread

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: deque[_Chunk] = deque()
        self._live: dict[int, Segment] = {}
        self._paused = False
        self._next = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def idle(self) -> bool:
        """Nothing is queued to play."""
        with self._lock:
            return not self._chunks

    @property
    def busy(self) -> bool:
        """Speech is queued (background sound does not count)."""
        with self._lock:
            return any(not chunk.segment.ambient for chunk in self._chunks)

    @property
    def paused(self) -> bool:
        return self._paused

    def open(self, *, ambient: bool = False) -> Segment:
        """Start a new line; write its audio, then `finish` it and await `done`."""
        self._loop = asyncio.get_running_loop()
        self._next += 1
        segment = Segment(self._next, self._loop.create_future(), ambient=ambient)
        with self._lock:
            self._live[segment.id] = segment
        return segment

    def write(self, segment: Segment, pcm: bytes) -> None:
        if not pcm or segment.id not in self._live:
            return
        with self._lock:
            segment.written += len(pcm)
            self._chunks.append(_Chunk(segment, pcm))

    def finish(self, segment: Segment) -> None:
        """No more audio for `segment`; it is done once all of it has played."""
        with self._lock:
            segment.finished = True
            drained = segment.played >= segment.written and segment.id in self._live
        if drained:
            self._drained(segment)

    def pull(self, size: int, *, pad: bool = True) -> bytes:
        """Up to `size` bytes to play now; padded with silence when `pad` (a sound card).

        Paused or empty: silence, or b"" without `pad`.
        """
        out = bytearray()
        started: list[Segment] = []
        drained: list[Segment] = []
        with self._lock:
            while not self._paused and self._chunks and len(out) < size:
                chunk = self._chunks[0]
                segment, data, offset = chunk.segment, chunk.data, chunk.offset
                take = data[offset : offset + size - len(out)]
                out += take
                if segment.started_at is None:
                    segment.started_at = time.monotonic()
                    started.append(segment)
                segment.played += len(take)
                if offset + len(take) >= len(data):
                    self._chunks.popleft()
                    if segment.finished and segment.played >= segment.written:
                        drained.append(segment)
                else:
                    chunk.offset = offset + len(take)
        if out and self.on_pull:
            played = bytes(out)
            for hook in self.on_pull:
                hook(played)
        for segment in started:
            if not segment.ambient:
                self._later(self.on_start, segment)
        for segment in drained:
            self._drained(segment)
        if pad and len(out) < size:
            out += bytes(size - len(out))
        return bytes(out)

    def pause(self) -> None:
        """Hold playback where it is (silence until `resume`)."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def clear(self) -> None:
        """Drop everything queued, unpause, and end every unfinished line as cut."""
        with self._lock:
            self._chunks.clear()
            live, self._live = list(self._live.values()), {}
            self._paused = False
        for segment in live:
            self._resolve(segment)
        for hook in self.on_clear:
            hook()

    def confirm(self, segment_id: int) -> None:
        """The far end reports `segment_id` played (with `confirm_playout`)."""
        with self._lock:
            segment = self._live.pop(segment_id, None)
        if segment is not None:
            self._resolve(segment)

    def _drained(self, segment: Segment) -> None:
        for hook in self.on_drained:
            hook(segment)
        if not self.confirm_playout:
            with self._lock:
                self._live.pop(segment.id, None)
            self._resolve(segment)

    def _resolve(self, segment: Segment) -> None:
        self._later([_set_done], segment)

    def _later(self, hooks: list[Callable[[Segment], None]], segment: Segment) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for hook in hooks:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(hook, segment)


def _set_done(segment: Segment) -> None:
    if not segment.done.done():
        segment.done.set_result(None)


class Pacer:
    """Sends a buffer's audio over a socket at the pace it plays, `lead_ms` ahead.

    The far end buffers what it receives, so audio is never sent more than `lead_ms`
    early: a pause or a cut stops what the caller hears within that much.
    """

    def __init__(
        self,
        buffer: PlayoutBuffer,
        send: Callable[[bytes], Awaitable[None]],
        *,
        frame_ms: int = 20,
        lead_ms: int = 60,
    ) -> None:
        self.buffer = buffer
        self._send = send
        self._frame = buffer.sample_rate * 2 * frame_ms // 1000
        self._frame_s = frame_ms / 1000
        self._lead_s = lead_ms / 1000
        self._bytes_per_s = buffer.sample_rate * 2
        self._deadline = 0.0  # when the far end will have played all we sent
        buffer.on_clear.append(self._reset)

    async def run(self) -> None:
        while True:
            now = time.monotonic()
            ahead = self._deadline - now - self._lead_s
            if ahead > 0:
                await asyncio.sleep(ahead)
                continue
            pcm = self.buffer.pull(self._frame, pad=False)
            if not pcm:
                await asyncio.sleep(self._frame_s)
                continue
            await self._send(pcm)
            self._deadline = max(self._deadline, now) + len(pcm) / self._bytes_per_s

    def _reset(self) -> None:
        self._deadline = 0.0


_background: set[asyncio.Task[Any]] = set()


def fire(coro: Coroutine[Any, Any, Any]) -> None:
    """Run `coro` in the background, held until it ends so it is not collected mid-send."""
    task = asyncio.ensure_future(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


__all__ = ["Pacer", "PlayoutBuffer", "Segment", "fire"]
