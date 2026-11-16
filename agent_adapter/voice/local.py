"""`LocalAudio`: the computer's microphone and speakers as a voice transport (sounddevice).

Use headphones: with no echo cancellation, the agent's own voice from the speakers reaches
the microphone and can cut the agent off.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice.playout import PlayoutBuffer


def _sounddevice() -> Any:
    try:
        import sounddevice
    except ImportError as exc:
        raise MissingDependency(
            'LocalAudio needs sounddevice: pip install "agent-adapter[voice-local]"'
        ) from exc
    return sounddevice


class LocalAudio:
    """Microphone in at `in_rate`, speakers out at `out_rate`, in `frame_ms` blocks.

    The speaker callback pulls straight from the playout buffer, so a pause or a cut is
    heard within one block.
    """

    def __init__(
        self,
        *,
        in_rate: int = 16_000,
        out_rate: int = 24_000,
        frame_ms: int = 20,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        speakers: bool = True,
    ) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.frame_ms = frame_ms
        self._devices = (input_device, output_device)
        self._speakers = speakers
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._streams: list[Any] = []

    async def start(self, buffer: PlayoutBuffer) -> None:
        sd, loop = _sounddevice(), asyncio.get_running_loop()

        def heard(data: Any, frames: int, when: Any, status: Any) -> None:
            with contextlib.suppress(RuntimeError):  # the loop closed at hang-up
                loop.call_soon_threadsafe(self._frames.put_nowait, bytes(data))

        def play(out: Any, frames: int, when: Any, status: Any) -> None:
            out[:] = buffer.pull(len(out))

        mic = sd.RawInputStream(
            samplerate=self.in_rate,
            blocksize=self.in_rate * self.frame_ms // 1000,
            channels=1,
            dtype="int16",
            device=self._devices[0],
            callback=heard,
        )
        self._streams.append(mic)
        if self._speakers:
            speaker = sd.RawOutputStream(
                samplerate=self.out_rate,
                blocksize=self.out_rate * self.frame_ms // 1000,
                channels=1,
                dtype="int16",
                device=self._devices[1],
                callback=play,
            )
            self._streams.append(speaker)
        for stream in self._streams:
            stream.start()

    async def frames(self) -> AsyncIterator[bytes]:
        while (frame := await self._frames.get()) is not None:
            yield frame

    def hang_up(self) -> None:
        """End the call: `frames()` stops."""
        self._frames.put_nowait(None)

    async def close(self) -> None:
        streams, self._streams = self._streams, []
        for stream in streams:
            stream.stop()
            stream.close()


__all__ = ["LocalAudio"]
