"""Shared parts of the live voice runs: the booking service, the scripted model, the caller.

Used by voice_live.py, voice_live_pipecat.py and voice_live_livekit.py; run those, not this.
`Measure` times every stack the same way, from what the caller hears.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Registry, ToolCall
from agent_adapter.exec import progress
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.pcm import rms
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
        log("[tool] book called")
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


LOUD = 0.005  # RMS above this is speech, not silence
QUIET_S = 0.3  # the agent counts as stopped after this much unbroken quiet


class Measure:
    """What the caller heard and when, so every stack is timed the same way.

    `hear(pcm)` is called for each block of agent audio as it starts to play. Results:
    first_audio_ms (the caller's first line ended -> the agent's first audio), agent_cut_ms
    (the cut starts -> the VoiceAgent is told) and audio_stop_ms (the cut starts -> the
    agent's audio stops for good).
    """

    def __init__(self) -> None:
        self.blocks: list[tuple[float, bool]] = []
        self.cuts: list[float] = []
        self.line_end: float | None = None
        self.onset: float | None = None
        self._cpu = time.process_time()

    def begin(self) -> None:
        """Start the CPU clock as the call starts: model loading is not the call's cost."""
        self._cpu = time.process_time()

    def hear(self, pcm: bytes) -> None:
        self.blocks.append((time.monotonic(), rms(pcm) > LOUD))

    def watch(self, agent: VoiceAgent) -> None:
        told = agent.user_started_speaking

        async def user_started_speaking() -> None:
            self.cuts.append(time.monotonic())
            await told()

        agent.user_started_speaking = user_started_speaking  # type: ignore[method-assign]

    def summary(self, stack: str, cut_s: float, bookings: Bookings) -> dict[str, Any]:
        return {
            "stack": stack,
            "cut_s": cut_s,
            "reserved": bookings.reserved,
            "charged": bookings.charged,
            "first_audio_ms": self._first_loud_after(self.line_end),
            "agent_cut_ms": self._first_cut_after(self.onset),
            "audio_stop_ms": self._stop_after(self.onset),
            "cpu_s": round(time.process_time() - self._cpu, 2),
        }

    def _first_loud_after(self, t: float | None) -> float | None:
        if t is None:
            return None
        loud = next((at for at, is_loud in self.blocks if at >= t and is_loud), None)
        return None if loud is None else round((loud - t) * 1000)

    def _first_cut_after(self, t: float | None) -> float | None:
        cut = next((at for at in self.cuts if t is not None and at >= t), None)
        return None if cut is None or t is None else round((cut - t) * 1000)

    def _stop_after(self, t: float | None) -> float | None:
        """None when the agent was silent as the cut began (nothing to stop)."""
        if t is None or not any(t - 0.1 <= at < t and loud for at, loud in self.blocks):
            return None
        # An output may write nothing while silent (Pipecat does), so a gap counts as quiet.
        quiet_from: float | None = None
        last = t
        for at, loud in self.blocks:
            if at < t:
                continue
            if at - last >= QUIET_S:
                return round(((quiet_from or last) - t) * 1000)
            last = at
            if loud:
                quiet_from = None
            elif quiet_from is None:
                quiet_from = at
            elif at - quiet_from >= QUIET_S:
                return round((quiet_from - t) * 1000)
        return round(((quiet_from or last) - t) * 1000)


def loud_span(pcm: bytes) -> tuple[int, int]:
    """The first and last 20 ms frame of `pcm` with speech in it."""
    frames = [pcm[i : i + FRAME * 2] for i in range(0, len(pcm), FRAME * 2)]
    loud = [i for i, frame in enumerate(frames) if rms(frame) > LOUD] or [0]
    return loud[0], loud[-1]


async def caller_script(
    send: Callable[[bytes], Awaitable[None]],
    lines: list[bytes],
    bookings: Bookings,
    agent: VoiceAgent,
    cut_s: float,
    measure: Measure,
) -> None:
    """The scripted caller in real time, 20 ms a frame: ask, wait, cut in mid-booking."""
    measure.begin()
    quiet, clock = b"\x00" * FRAME * 2, time.monotonic()

    async def play(pcm: bytes, on_frame: Callable[[int], None] = lambda i: None) -> None:
        nonlocal clock
        for n, i in enumerate(range(0, len(pcm), FRAME * 2)):
            on_frame(n)
            await send(pcm[i : i + FRAME * 2].ljust(FRAME * 2, b"\x00"))
            clock += FRAME / RATE
            await asyncio.sleep(max(0.0, clock - time.monotonic()))

    async def quiet_until(done: Callable[[], bool], limit: float) -> None:
        end = time.monotonic() + limit
        while not done() and time.monotonic() < end:
            await play(quiet)

    def mark(attr: str, frame: int, after: float = 0.0) -> Callable[[int], None]:
        def at(n: int) -> None:
            if n == frame:
                setattr(measure, attr, time.monotonic() + after)

        return at

    await quiet_until(lambda: False, 1.0)
    log(f"[caller] says: {LINES[0]!r}")
    await play(lines[0], mark("line_end", loud_span(lines[0])[1], after=FRAME / RATE))
    await quiet_until(bookings.started.is_set, 20)
    await quiet_until(lambda: False, cut_s)
    log(f"[caller] cuts in: {LINES[1]!r}")
    await play(lines[1], mark("onset", loud_span(lines[1])[0]))
    await quiet_until(lambda: bool(bookings.charged) and answered(agent), 25)
    await quiet_until(lambda: False, 5)  # let the answer play


def report(result: dict[str, Any], path: Path | None) -> None:
    """Print the run's numbers; with `path`, append them as one JSON line."""
    print(json.dumps(result))
    if path is not None:
        with path.open("a", encoding="utf-8") as out:
            print(json.dumps(result), file=out)
