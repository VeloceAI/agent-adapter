"""VoxCat's hosted providers with no keys and no network.

Each adapter talks to an in-memory fake that answers with the provider's documented
messages. The script prints the turn signals Deepgram Flux sends, then the time our own
code adds per audio chunk for every adapter (the provider's own delay is not in it).
Run: python examples/voxcat_providers_offline.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable, Iterable

import msgpack

from agent_adapter.voice.assemblyai import AssemblyAISTT
from agent_adapter.voice.cartesia import CartesiaInkSTT, CartesiaLiveTTS
from agent_adapter.voice.deepgram import DeepgramFluxSTT
from agent_adapter.voice.elevenlabs import ElevenLabsLiveTTS, ElevenLabsScribeSTT
from agent_adapter.voice.fish import FishAudioTTS
from agent_adapter.voice.rime import RimeTTS
from agent_adapter.voice.soniox import SonioxSTT

KEY = "offline-not-a-key"
CHUNKS = 2000
AUDIO = b"\x01\x00" * 480  # 20 ms at 24 kHz
Reply = Callable[[str | bytes], Iterable[str | bytes]]


class FakeSocket:
    """Answers each frame we send with the frames `reply` returns."""

    def __init__(self, reply: Reply, greeting: Iterable[str | bytes]) -> None:
        self.reply = reply
        self.inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        for frame in greeting:
            self.inbox.put_nowait(frame)

    async def send(self, frame: str | bytes) -> None:
        for out in self.reply(frame):
            self.inbox.put_nowait(out)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self.inbox.get()) is not None:
            yield frame

    async def close(self) -> None:
        self.inbox.put_nowait(None)


def fake(reply: Reply, greeting: Iterable[str | bytes] = ()):
    async def connect(url: str, headers: dict[str, str]) -> FakeSocket:
        return FakeSocket(reply, greeting)

    return connect


def at_end(is_end: Callable[[str | bytes], bool], replies: Callable[[str | bytes], list]) -> Reply:
    return lambda frame: replies(frame) if is_end(frame) else []


def flux_turns() -> list[str]:
    def turn(event: str, text: str, confidence: float) -> str:
        message = {"type": "TurnInfo", "event": event, "turn_index": 0, "transcript": text}
        return json.dumps({**message, "end_of_turn_confidence": confidence})

    return [
        turn("EagerEndOfTurn", "book two", 0.55),
        turn("TurnResumed", "book two", 0.3),
        turn("EndOfTurn", "Book two seats.", 0.9),
    ]


async def show_turn_signals() -> None:
    turns = iter(flux_turns())
    flux = DeepgramFluxSTT(api_key=KEY, connect=fake(lambda f: [next(turns, "{}")]))
    stream = flux.open(16_000)
    signals = []
    stream.on_turn(signals.append)
    for _ in range(3):
        stream.feed(AUDIO)
    while len(signals) < 3:
        await asyncio.sleep(0)
    print("final:", await stream.finish())
    for signal in signals:
        print(f"signal: {signal.event} {signal.text!r} {signal.confidence}")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def tts_cases() -> dict[str, object]:
    def cartesia(frame):
        context = json.loads(frame)["context_id"]
        chunk = json.dumps({"type": "chunk", "data": b64(AUDIO), "context_id": context})
        return [chunk] * CHUNKS + [json.dumps({"type": "done", "context_id": context})]

    def eleven(frame):
        return [json.dumps({"audio": b64(AUDIO)})] * CHUNKS + [json.dumps({"isFinal": True})]

    def fish(frame):
        audio = msgpack.packb({"event": "audio", "audio": AUDIO}, use_bin_type=True)
        return [audio] * CHUNKS + [msgpack.packb({"event": "finish", "reason": "stop"})]

    def rime(frame):
        context = json.loads(frame)["contextId"]
        chunk = json.dumps({"contextId": context, "audio": b64(AUDIO)})
        return [chunk] * CHUNKS + [json.dumps({"contextId": context, "done": {}})]

    def ends(marker: str | bytes):
        return lambda frame: marker in frame

    ready = [json.dumps({"ready": {"protocol": 1}})]
    return {
        "cartesia sonic": CartesiaLiveTTS(
            "v", api_key=KEY, connect=fake(at_end(ends('"continue":false'), cartesia))
        ),
        "elevenlabs stream-input": ElevenLabsLiveTTS(
            "v", api_key=KEY, connect=fake(at_end(lambda f: f == '{"text":""}', eleven))
        ),
        "fish audio live": FishAudioTTS(
            "v", api_key=KEY, connect=fake(at_end(ends(b"stop"), fish))
        ),
        "rime v1": RimeTTS("v", api_key=KEY, connect=fake(at_end(ends('"end":{}'), rime), ready)),
    }


def stt_cases() -> dict[str, object]:
    final = {
        "deepgram flux": json.dumps({"type": "TurnInfo", "event": "EndOfTurn", "transcript": "ok"}),
        "assemblyai": json.dumps(
            {"type": "Turn", "turn_order": 0, "end_of_turn": True, "transcript": "ok"}
        ),
        "soniox": json.dumps(
            {"tokens": [{"text": "ok", "is_final": True}, {"text": "<fin>", "is_final": True}]}
        ),
        "cartesia ink": json.dumps({"type": "transcript", "is_final": True, "text": "ok"}),
        "elevenlabs scribe": json.dumps({"message_type": "committed_transcript", "text": "ok"}),
    }
    ends = {
        "deepgram flux": lambda f: "ForceEndTurn" in f,
        "assemblyai": lambda f: "ForceEndpoint" in f,
        "soniox": lambda f: "finalize" in f,
        "cartesia ink": lambda f: f == "finalize",
        "elevenlabs scribe": lambda f: '"commit":true' in f,
    }

    def connect(name: str):
        replies = [final[name]] + (
            [json.dumps({"type": "flush_done"})] if name == "cartesia ink" else []
        )
        return fake(at_end(lambda f: isinstance(f, str) and ends[name](f), lambda f: replies))

    return {
        "deepgram flux": DeepgramFluxSTT(api_key=KEY, connect=connect("deepgram flux")),
        "assemblyai": AssemblyAISTT(api_key=KEY, connect=connect("assemblyai")),
        "soniox": SonioxSTT(api_key=KEY, connect=connect("soniox")),
        "cartesia ink": CartesiaInkSTT(api_key=KEY, connect=connect("cartesia ink")),
        "elevenlabs scribe": ElevenLabsScribeSTT(api_key=KEY, connect=connect("elevenlabs scribe")),
    }


async def tts_overhead(tts) -> float:
    stream = tts.open()
    await stream.push("Hello there.")
    started = time.perf_counter()
    await stream.end()
    total = sum([len(chunk) async for chunk in stream.audio()])
    assert total == len(AUDIO) * CHUNKS
    return (time.perf_counter() - started) / CHUNKS * 1e6


async def stt_overhead(stt) -> float:
    pcm = b"\x01\x00" * 1600  # 100 ms at 16 kHz
    started = time.perf_counter()
    stream = stt.open(16_000)
    for _ in range(CHUNKS):
        stream.feed(pcm)
    assert await stream.finish() == "ok"
    return (time.perf_counter() - started) / CHUNKS * 1e6


async def main() -> None:
    await show_turn_signals()
    print(f"{'adapter':<26}{'kind':<6}{'us per chunk':>12}")
    for name, tts in tts_cases().items():
        print(f"{name:<26}{'tts':<6}{await tts_overhead(tts):>12.1f}")
    for name, stt in stt_cases().items():
        print(f"{name:<26}{'stt':<6}{await stt_overhead(stt):>12.1f}")


if __name__ == "__main__":
    asyncio.run(main())
