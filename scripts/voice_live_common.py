"""Shared parts of the live voice runs: the booking service, the scripted model, the caller.

Used by voice_live_pipecat.py and voice_live_livekit.py; run those, not this.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Registry, ToolCall
from agent_adapter.exec import progress
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.smart_turn import RATE

FRAME = 320  # 20 ms at 16 kHz
LINES = ("Book two seats for the jazz show on Friday, please.", "Wait, did that go through?")
CHARGE_S = 3.0
T0 = time.monotonic()


def log(text: str) -> None:
    print(f"  {time.monotonic() - T0:6.2f}s  {text}", flush=True)


class Bookings:
    """Reserve, then charge after CHARGE_S: a cut in between leaves it half-done."""

    def __init__(self) -> None:
        self.reserved = self.charged = 0
        self.started = asyncio.Event()

    async def book(self, show: str, seats: int) -> str:
        self.reserved += 1
        self.started.set()
        log(f"[tool] book started: reserved {seats} seats for {show}, charging...")
        await asyncio.sleep(CHARGE_S)
        self.charged += 1
        log("[tool] book finished: charged")
        return f"booking J-1042, {seats} seats for the {show} show"


def registry(bookings: Bookings) -> Registry:
    tools = Registry()

    @tools.tool(effects="destructive")
    async def book(show: str, seats: int) -> str:
        """Book seats for a show and charge the card on file."""
        progress("Booking that now, one moment.")
        return await bookings.book(show, seats)

    return tools


def _reply(text: str = "", *calls: ToolCall) -> LLMResponse:
    message = Message(role="assistant", content=text, tool_calls=calls)
    finish = "tool_calls" if calls else "stop"
    return LLMResponse(
        provider="script",
        model="script",
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        message=message,
    )


def scripted_model() -> CallableModel:
    """Books when asked; asked "did it work?" without seeing a result, it books again."""
    count = 0

    async def answer(request: LLMRequest) -> LLMResponse:
        nonlocal count
        last = request.messages[-1]
        if last.role == "tool":
            return _reply(f"All done: {last.content}.")
        notes = [str(m.content) for m in request.messages if "Already done" in str(m.content)]
        if notes:
            return _reply("Yes, it went through. " + notes[-1].split("finished: ")[-1].strip("'.)"))
        if any(word in str(last.content).lower() for word in ("book", "did", "through")):
            count += 1
            seats = {"show": "jazz", "seats": 2}
            return _reply("", ToolCall(name="book", arguments=seats, id=f"call_{count}"))
        return _reply("Sorry, could you say that again?")

    return CallableModel(answer, provider="script", model="script")


def to_pcm16(samples: Any, rate: int, out_rate: int | None = None) -> bytes:
    """Float samples in [-1, 1] as 16-bit PCM, resampled to `out_rate` if given."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if out_rate and out_rate != rate:
        audio = np.interp(np.arange(0, len(audio), rate / out_rate), np.arange(len(audio)), audio)
    return bytes((np.clip(audio, -1, 1) * 32_767).astype(np.int16).tobytes())


def caller_lines() -> list[bytes]:
    """The scripted caller's two lines, spoken by a second Kokoro voice, as 16 kHz PCM."""
    from moonshine_voice import TextToSpeech

    tts = TextToSpeech().language("en_us").voice("kokoro_am_michael").load()
    return [to_pcm16(*tts.synthesize(line), out_rate=RATE) for line in LINES]


def answered(agent: VoiceAgent) -> bool:
    """The agent's last turn ended with words, not a pending tool call."""
    last = agent.history[-1] if agent.history else None
    return last is not None and last.role == "assistant" and not last.tool_calls


def model_paths(silero: Path | None, smart: Path | None) -> tuple[Path, Path]:
    """The Silero and Smart Turn files: as given, or the copies bundled with pipecat-ai."""
    if silero and smart:
        return silero, smart
    try:
        import pipecat
    except ImportError:
        raise SystemExit("pass --silero and --smart-turn, or install pipecat-ai") from None
    base = Path(pipecat.__file__).parent
    found = sorted((base / "audio/turn/smart_turn/data").glob("smart-turn-v3*cpu*.onnx"))
    return silero or base / "audio/vad/data/silero_vad.onnx", smart or found[-1]
