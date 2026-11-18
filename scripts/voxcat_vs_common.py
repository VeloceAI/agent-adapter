"""What every framework in voxcat_vs.py shares: the caller, the fake parts' timing, results.

Stdlib only. Each harness (voxcat_vs_voxcat, voxcat_vs_livekit, voxcat_vs_pipecat) exposes
`async def run_trial(setup: Setup, scenario: Scenario, delays: Delays) -> Trial` and builds
its framework's own STT, LLM, TTS and VAD on these rules, on the real clock:

- caller: `caller_frames(scenario)`, 20 ms frames of 16 kHz mono PCM, sent in real time;
- VAD: speech is a frame with `is_speech(pcm)`; the end-of-turn silence is `setup.end_ms`
  (None: the framework's own default);
- STT (streaming): interim text `heard(scenario, ms)` as speech arrives; the final text
  `delays.stt_ms` after the STT is asked to finish (or after its own endpoint, quiet for
  `delays.stt_endpoint_ms`). With `setup.provider_tricks` the STT also says the turn ended,
  like Deepgram Flux: an eager end `delays.flux_eager_ms` after speech and a final one
  `delays.flux_end_ms` after speech, and resumes if speech comes back;
- LLM: `reply_tokens(delays, counters)`, first token after `delays.llm_first_ms`;
- TTS: first audio `delays.tts_first_ms` after a request, then `tts_audio(text)` (a tone),
  streamed in 20 ms frames.

Times are ms. `Trial` times are after the caller's last speech frame was sent.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from array import array
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

RATE = 16_000
FRAME_MS = 20
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
REPLY = "Sure, I can book a table for two at seven tonight. Would you like a window seat?"


@dataclass(frozen=True)
class Delays:
    """Approximate vendor-stated figures; change them with the script's flags."""

    stt_ms: float = 100.0  # streaming STT: final words after it is asked to finish
    stt_endpoint_ms: float = 300.0  # its own endpoint when nobody asks
    flux_eager_ms: float = 150.0  # provider trick: eager end of turn after speech
    flux_end_ms: float = 260.0  # provider trick: end of turn after speech
    llm_first_ms: float = 350.0
    tokens_per_s: float = 40.0
    tts_first_ms: float = 90.0
    word_audio_ms: float = 300.0


@dataclass(frozen=True)
class Setup:
    name: str
    end_ms: float | None  # end-of-turn silence; None: the framework's default
    provider_tricks: bool  # the STT gives Flux-like turn events; the framework uses them


SETUPS = (
    Setup("defaults", None, provider_tricks=False),
    Setup("same 500 ms wait", 500.0, provider_tricks=False),
    Setup("them: Flux-like STT", None, provider_tricks=True),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    segments: tuple[tuple[str, float], ...]  # ("speech" | "quiet", ms)
    transcript: str
    barge_after_ms: float | None = None  # caller cuts in this long after the agent starts


SCENARIOS = (
    Scenario(
        "plain",
        (("quiet", 300), ("speech", 1500), ("quiet", 2500)),
        "Book a table for two at seven tonight.",
    ),
    Scenario(
        "mid-turn pause",
        (("quiet", 300), ("speech", 900), ("quiet", 450), ("speech", 700), ("quiet", 2500)),
        "Book a table for two at seven tonight.",
    ),
    Scenario(
        "barge-in",
        (("quiet", 300), ("speech", 1500), ("quiet", 4000)),
        "Book a table for two at seven tonight.",
        barge_after_ms=600,
    ),
)


@dataclass
class Counters:
    llm_calls: int = 0
    tokens: int = 0


@dataclass
class Trial:
    framework: str
    setup: str
    scenario: str
    first_audio_ms: float | None = None  # caller's last speech -> agent's first audio
    cut_in: bool = False  # the agent began talking during the caller's pause
    stop_ms: float | None = None  # barge-in: caller speech start -> agent audio stops
    llm_calls: int = 0
    tokens: int = 0
    cpu_us_per_frame: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)


def tone_pcm(ms: float, freq: float = 220.0, level: int = 8000) -> bytes:
    count = int(RATE * ms / 1000)
    return array(
        "h", (int(level * math.sin(2 * math.pi * freq * i / RATE)) for i in range(count))
    ).tobytes()


def silence_pcm(ms: float) -> bytes:
    return bytes(int(RATE * ms / 1000) * 2)


def is_speech(pcm: bytes, threshold: float = 0.02) -> bool:
    samples = array("h", pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return False
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768 >= threshold


def caller_frames(scenario: Scenario) -> Iterator[tuple[bytes, bool]]:
    """(20 ms frame, is speech) for the scripted caller, without the barge-in."""
    for kind, ms in scenario.segments:
        pcm = tone_pcm(ms) if kind == "speech" else silence_pcm(ms)
        for i in range(0, len(pcm), FRAME_BYTES):
            yield pcm[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0"), kind == "speech"


def speech_ms(scenario: Scenario) -> float:
    return sum(ms for kind, ms in scenario.segments if kind == "speech")


def heard(scenario: Scenario, ms: float) -> str:
    """The interim transcript after `ms` of the caller's speech."""
    words = scenario.transcript.split()
    share = min(1.0, ms / max(1.0, speech_ms(scenario)))
    text = " ".join(words[: round(len(words) * share)])
    return text if share < 1 else scenario.transcript


async def reply_tokens(delays: Delays, counters: Counters) -> AsyncIterator[str]:
    """The fake model's streamed reply: word tokens, the first after `llm_first_ms`."""
    counters.llm_calls += 1
    await asyncio.sleep(delays.llm_first_ms / 1000)
    for i, word in enumerate(REPLY.split()):
        if i:
            await asyncio.sleep(1 / delays.tokens_per_s)
        counters.tokens += 1
        yield word if i == 0 else " " + word


def tts_audio(text: str, delays: Delays) -> bytes:
    return tone_pcm(max(1, len(text.split())) * delays.word_audio_ms, freq=330.0)


def now_ms() -> float:
    return time.perf_counter() * 1000


async def send_caller(
    scenario: Scenario,
    push: Callable[[bytes], object],  # may return an awaitable
    marks: dict[str, float],
    barge: asyncio.Event | None = None,
) -> None:
    """Sends the caller in real time and marks 'speech_start', 'speech_end' (last speech
    frame), 'pause_start' and 'barge_start'. Once `barge` is set, 600 ms of speech replace
    the quiet that follows (the caller cuts in)."""
    start = now_ms()
    barge_left, was_speech = 600.0, False
    for sent, (pcm, speech) in enumerate(caller_frames(scenario), start=1):
        if barge is not None and barge.is_set() and barge_left > 0 and not speech:
            pcm, barge_left = tone_pcm(FRAME_MS), barge_left - FRAME_MS
            marks.setdefault("barge_start", now_ms())
        result = push(pcm)
        if asyncio.iscoroutine(result):
            await result
        t = now_ms()
        if speech:
            marks["speech_end"] = t
            marks.setdefault("speech_start", t)
        elif was_speech:
            marks.setdefault("pause_start", t)
        was_speech = speech
        wait = start + sent * FRAME_MS - now_ms()
        if wait > 0:
            await asyncio.sleep(wait / 1000)
    marks["caller_done"] = now_ms()


class Playout:
    """The agent's speaker: frames play in real time; notes when each audible one played.

    A harness calls `await play(pcm)` for each output frame (any length); it sleeps for the
    frame's duration, so a framework that pushes audio faster than real time is paced.
    `on_first` (an asyncio.Event) is set when the first audible frame plays."""

    def __init__(self) -> None:
        self.audible: list[float] = []
        self.on_first = asyncio.Event()

    async def play(self, pcm: bytes) -> None:
        for i in range(0, len(pcm), FRAME_BYTES):
            frame = pcm[i : i + FRAME_BYTES]
            if is_speech(frame):
                self.audible.append(now_ms())
                self.on_first.set()
            await asyncio.sleep(len(frame) / 2 / RATE)


def score(trial: Trial, marks: dict[str, float], playout: Playout) -> Trial:
    """Fills the trial's times from the caller's marks and what played."""
    end = marks.get("speech_end")
    if playout.audible and end is not None:
        first = playout.audible[0]
        pause = marks.get("pause_start")
        trial.cut_in = pause is not None and pause <= first < end
        after = [t for t in playout.audible if t >= end]
        trial.first_audio_ms = round(after[0] - end, 1) if after else None
    barge = marks.get("barge_start")
    if barge is not None:
        later = [t for t in playout.audible if t >= barge]
        stop = barge
        for t in later:
            if t - stop > 100:
                break
            stop = t
        trial.stop_ms = round(stop - barge, 1)
    return trial


def summary(trials: list[Trial]) -> dict[str, float | int | None]:
    firsts = sorted(t.first_audio_ms for t in trials if t.first_audio_ms is not None)
    stops = sorted(t.stop_ms for t in trials if t.stop_ms is not None)

    def pct(values: list[float], q: float) -> float | None:
        return round(values[min(len(values) - 1, int(len(values) * q))]) if values else None

    return {
        "runs": len(trials),
        "errors": sum(1 for t in trials if t.error),
        "first_p50": pct(firsts, 0.5),
        "first_p95": pct(firsts, 0.95),
        "stop_p50": pct(stops, 0.5),
        "cut_ins": sum(t.cut_in for t in trials),
        "llm_calls": round(statistics.mean(t.llm_calls for t in trials), 2) if trials else 0,
        "tokens": round(statistics.mean(t.tokens for t in trials), 1) if trials else 0,
        "cpu_us": round(statistics.mean(t.cpu_us_per_frame for t in trials)) if trials else 0,
    }
