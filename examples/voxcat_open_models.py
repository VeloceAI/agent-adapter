"""VoxCat on open models: speech-to-text, text-to-speech and a talker beside the agent.

Offline by default: stand-ins play each part, so it needs no model, GPU or download.
With `--real` on a GPU machine (extras `voice-whisper` and `voice-kyutai`), it uses
faster-whisper, Kyutai TTS and Moshi; they download on first use.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.voice import CallableSTT, CallableTTS
from agent_adapter.voice.pcm import rms
from agent_adapter.voice.talker import Cue, backchannel


class NodStream:
    """A stand-in talker: says "mm-hm" after every 300 ms of the caller's speech."""

    def __init__(self, rate: int) -> None:
        self._rate, self._heard = rate, 0
        self._cues: asyncio.Queue[Cue | None] = asyncio.Queue()

    def hear(self, pcm: bytes) -> None:
        if rms(pcm) > 0.01:
            self._heard += len(pcm) // 2
        if self._heard >= self._rate * 0.3:
            self._heard = 0
            self._cues.put_nowait(Cue(bytes(4_800), "mm-hm"))

    async def cues(self) -> AsyncIterator[Cue]:
        while (cue := await self._cues.get()) is not None:
            yield cue

    async def close(self) -> None:
        self._cues.put_nowait(None)


class NodTalker:
    sample_rate = 16_000

    async def warm(self) -> None:
        pass

    def open(self, sample_rate: int) -> NodStream:
        return NodStream(sample_rate)


def parts(real: bool) -> tuple[Any, Any, Any]:
    if real:
        from agent_adapter.voice.kyutai import KyutaiTTS
        from agent_adapter.voice.moshi_talker import MoshiTalker
        from agent_adapter.voice.whisper import FasterWhisperSTT

        return FasterWhisperSTT("large-v3-turbo", device="cuda"), KyutaiTTS(), MoshiTalker()
    stt = CallableSTT(lambda pcm, rate: "where is my order")
    tts = CallableTTS(lambda text: bytes(2 * 240 * len(text)), sample_rate=24_000)
    return stt, tts, NodTalker()


async def main(real: bool = False) -> None:
    stt, tts, talker = parts(real)
    for part in (stt, tts, talker):
        if hasattr(part, "warm"):
            await part.warm()  # load each model once, before the first caller

    stream = talker.open(16_000)
    played: list[bytes] = []
    agent_speaking = False
    nods = asyncio.create_task(
        backchannel(stream, played.append, may_speak=lambda: not agent_speaking)
    )

    caller = b"\x00\x10" * 16_000  # one second of speech at 16 kHz, in 20 ms frames
    for start in range(0, len(caller), 640):
        stream.hear(caller[start : start + 640])
    text = await stt.transcribe(caller, 16_000)
    print(f"caller: {text}")
    await asyncio.sleep(0.05)  # the agent looks the order up; the talker fills the gap
    print(f"talker cues while the agent thinks: {len(played)}")

    agent_speaking = True
    answer = "It ships today."
    audio = b"".join([chunk async for chunk in tts.synthesize(answer)])
    print(f"agent: {answer} ({len(audio) // 2} samples at {tts.sample_rate} Hz)")
    await stream.close()
    await nods


if __name__ == "__main__":
    asyncio.run(main(real="--real" in sys.argv))
