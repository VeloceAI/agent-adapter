"""Shared parts of the open-model adapters: one warm model per process, PCM conversions and
a thread-to-async audio stream. No third-party imports here; adapters pass their modules in.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from array import array
from collections.abc import AsyncIterator, Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.voice.pcm import bytes_of, samples_of


@dataclass
class Warm:
    """A loaded model and the lock that keeps it to one engine call at a time."""

    model: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


_MODELS: dict[Hashable, Warm] = {}
_GUARD = threading.Lock()


def shared(key: Hashable, load: Callable[[], Any]) -> Warm:
    """The process-wide model for `key`, loaded on first use; later adapters reuse it."""
    with _GUARD:
        warm = _MODELS.get(key)
        if warm is None:
            warm = _MODELS[key] = Warm(load())
        return warm


def forget_models() -> None:
    """Drop every shared model (tests, or to free GPU memory)."""
    with _GUARD:
        _MODELS.clear()


class Lazy:
    """Loads the shared model on first use, from a worker thread, so the event loop never
    waits on a download or a GPU upload."""

    def __init__(self, key: Hashable, load: Callable[[], Any]) -> None:
        self._key, self._load = key, load
        self._warm: Warm | None = None

    def get(self) -> Warm:
        if self._warm is None:
            self._warm = shared(self._key, self._load)
        return self._warm

    async def aget(self) -> Warm:
        if self._warm is None:
            return await asyncio.to_thread(self.get)
        return self._warm


def floats(pcm: bytes) -> list[float]:
    """PCM16 as floats in [-1, 1)."""
    return [s / 32_768 for s in samples_of(pcm)]


def flat(audio: Any) -> list[float]:
    """A model's audio (a torch tensor, a NumPy array or a list) as a flat list of floats."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu()
    if hasattr(audio, "reshape"):
        return [float(s) for s in audio.reshape(-1).tolist()]
    return [float(s) for s in audio]


def pcm16(samples: Iterable[float]) -> bytes:
    """Floats in [-1, 1] as PCM16, clipped."""
    return bytes_of(array("h", (int(max(-1.0, min(1.0, s)) * 32_767) for s in samples)))


_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")


def sentences(text: str) -> list[str]:
    """Split a line at sentence ends, so a model that renders whole clips starts sooner."""
    return [part for part in (p.strip() for p in _SENTENCE.split(text)) if part]


async def thread_stream(
    produce: Callable[[Callable[[bytes], None], threading.Event], None],
) -> AsyncIterator[bytes]:
    """Run `produce(emit, stop)` in a thread and yield what it emits, as it emits it.

    When the consumer stops early (a barge-in), `stop` is set; `produce` should check it
    between chunks and return.
    """
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop = threading.Event()

    def emit(pcm: bytes) -> None:
        if pcm:
            loop.call_soon_threadsafe(chunks.put_nowait, pcm)

    def run() -> None:
        try:
            produce(emit, stop)
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
