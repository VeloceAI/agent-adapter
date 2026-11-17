"""VoxCat speed benchmark: our own pipeline with fake STT, LLM and TTS on simulated delays.

    python scripts/voxcat_speed.py            # the table
    python scripts/voxcat_speed.py --json     # machine-readable
    python scripts/voxcat_speed.py --llm-ms 500 --tts-ms 75 --stt-ms 120

The real `VoiceSession` and `VoiceAgent` run on a virtual clock: provider delays are exact
and the result is the same on every machine. Only the pipeline's CPU time is real, and it is
measured on its own (overhead per 20 ms frame). Times are ms after the caller's last speech.

Default delays are approximate vendor-stated figures; check them for your providers:
streaming STT final words ~100 ms after the audio (Deepgram Nova-3, AssemblyAI Universal-
Streaming), first LLM token ~350 ms for a fast hosted model (900 ms for a large one), first
TTS audio ~90 ms (Cartesia Sonic ~90 ms, ElevenLabs Flash ~75 ms), a lookup API ~300 ms.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import selectors
import statistics
import sys
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_adapter import CallableModel, Executor, MemoryCache, Registry
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.types import ToolCall
from agent_adapter.voice import VoiceAgent, VoiceSession
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.answers import AnswerCache
from agent_adapter.voice.pcm import Resampler
from agent_adapter.voice.speculate import SpeculativeTools
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import MemoryTransport, silence, tone
from agent_adapter.voice.turns import SilenceTurns

TICK_MS = 5  # the virtual line moves audio in 5 ms steps: the timing resolution


@dataclass
class Delays:
    stt_ms: float = 100.0
    llm_ms: float = 350.0
    slow_llm_ms: float = 900.0
    tts_ms: float = 90.0
    tool_ms: float = 300.0


@dataclass(frozen=True)
class Profile:
    name: str
    end_ms: float  # quiet that ends a turn
    early_ms: float  # quiet that starts the final transcription (and the speculative reply)


PROFILES = (
    Profile("silence end-of-turn (800 ms)", end_ms=800, early_ms=200),
    Profile("smart end-of-turn (160 ms)", end_ms=160, early_ms=60),
)


# --- a virtual clock: the loop jumps to the next timer instead of waiting for it ---


class _Clock:
    now = 1000.0


class _JumpSelector(selectors.BaseSelector):
    def __init__(self, clock: _Clock) -> None:
        self._real = selectors.DefaultSelector()
        self._clock = clock

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._real.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.modify(fileobj, events, data)

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = self._real.select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            return self._real.select(None)
        self._clock.now += timeout
        return []

    def get_map(self) -> Any:
        return self._real.get_map()

    def close(self) -> None:
        self._real.close()


class _VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        super().__init__(_JumpSelector(clock))
        self._clock_resolution = 1e-9  # else timers up to 15.6 ms early count as due (Windows)

    def time(self) -> float:
        return self._clock.now


@contextlib.contextmanager
def virtual_time() -> Iterator[_VirtualLoop]:
    clock = _Clock()
    loop = _VirtualLoop(clock)
    real = time.monotonic
    time.monotonic = lambda: clock.now  # the pipeline's stage stamps use the same clock
    try:
        yield loop
    finally:
        time.monotonic = real
        loop.close()


# --- fake providers with the delays above ---


class FakeSTT:
    """Streaming STT: words arrive while the caller talks; `finish` takes `stt_ms`."""

    def __init__(self, text: str, delays: Delays) -> None:
        self.text, self.delays = text, delays

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        await asyncio.sleep(self.delays.stt_ms * 3 / 1000)
        return self.text

    def open(self, sample_rate: int) -> _FakeStream:
        return _FakeStream(self)


class _FakeStream:
    def __init__(self, stt: FakeSTT) -> None:
        self.stt = stt

    def feed(self, pcm: bytes) -> None:
        pass

    async def finish(self) -> str:
        await asyncio.sleep(self.stt.delays.stt_ms / 1000)
        return self.stt.text

    async def cancel(self) -> None:
        pass


class FakeTTS:
    """First audio after `tts_ms`, then the rest of the line faster than real time."""

    sample_rate = 16_000

    def __init__(self, delays: Delays) -> None:
        self.delays = delays

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.sleep(self.delays.tts_ms / 1000)
        for _ in text.split():
            yield tone(60)
            await asyncio.sleep(0.01)


def fake_llm(delays: Delays, *, slow: bool = False, tool: bool = False) -> CallableModel:
    first = (delays.slow_llm_ms if slow else delays.llm_ms) / 1000

    async def reply(request: LLMRequest) -> LLMResponse | str:
        await asyncio.sleep(first)
        if tool and request.messages[-1].role == "user":
            call = ToolCall(name="lookup", arguments={"city": "Oslo"}, id="call_1")
            message = Message(role="assistant", tool_calls=(call,))
            return LLMResponse(
                provider="fake",
                model="fake",
                tool_calls=(call,),
                message=message,
                finish_reason="tool_calls",
            )
        return "It is sunny in Oslo today. Anything else?"

    return CallableModel(reply, provider="fake", model="fake")


def registry_for(delays: Delays) -> Registry:
    registry = Registry()

    @registry.tool(effects="read_only", cache_ttl=60)
    async def lookup(city: str) -> str:
        """Weather for a city."""
        await asyncio.sleep(delays.tool_ms / 1000)
        return f"{city}: sunny"

    return registry


# --- one call: speak a question, wait for the answer, read the stage timings ---


@dataclass
class Result:
    scenario: str
    first_sound_ms: float | None
    answer_ms: float | None
    stages: dict[str, float] = field(default_factory=dict)


async def one_turn(
    profile: Profile,
    delays: Delays,
    *,
    speed: Speed | None,
    slow: bool = False,
    tool: bool = False,
    question: str = "What is the weather in Oslo today?",
) -> tuple[float | None, float | None, dict[str, float]]:
    transport = MemoryTransport()
    session = VoiceSession(
        transport,
        stt=FakeSTT(question, delays),
        tts=(speed.answers.voice(FakeTTS(delays)) if speed and speed.answers else FakeTTS(delays)),
        turns=SilenceTurns(start_ms=40, end_ms=profile.end_ms),
        early_ms=profile.early_ms,
    )
    executor = Executor(registry_for(delays), cache=MemoryCache())
    agent = VoiceAgent(
        fake_llm(delays, slow=slow, tool=tool), executor, session.speaker, speed=speed
    )
    run = asyncio.create_task(session.run(agent))
    await asyncio.sleep(0.5)  # call start: warm-up renders acknowledgements and answers
    step = 16_000 * 2 * TICK_MS // 1000
    caller = tone(1200) + silence(3000)
    for i in range(0, len(caller), step):
        transport.speak(caller[i : i + step], frame_ms=TICK_MS)
        transport.play(TICK_MS)
        await asyncio.sleep(TICK_MS / 1000)
    transport.hang_up()
    await asyncio.wait_for(run, 30)
    timing = session.timings[0]
    stages = timing.stages()
    replied = stages.get("replied")
    answer_ms = stages.get("first_audio")  # no acknowledgement: the first sound is the answer
    if timing.acked is not None and replied is not None:
        answer_ms = replied + delays.tts_ms
    return timing.voice_to_voice_ms, answer_ms, stages


def run_turn(profile: Profile, delays: Delays, **options: Any) -> tuple[Any, ...]:
    with virtual_time() as loop:
        return loop.run_until_complete(one_turn(profile, delays, **options))


def scenarios(profile: Profile, delays: Delays) -> list[Result]:
    acks = lambda: Acknowledgements(["Mm-hm."], expect_ms=delays.slow_llm_ms)  # noqa: E731
    tools = lambda: SpeculativeTools(["lookup"])  # noqa: E731
    runs: list[tuple[str, dict[str, Any]]] = [
        ("plain: no speed tricks", {"speed": None}),
        ("plain: speculative reply", {"speed": Speed()}),
        ("tool: no speed tricks", {"speed": None, "tool": True}),
        ("tool: speculative reply, tool waits", {"speed": Speed(), "tool": True}),
        ("tool: speculative reply + tool", {"speed": Speed(tools=tools()), "tool": True}),
        ("slow model: no speed tricks", {"speed": None, "slow": True}),
        (
            "slow model: acknowledgement",
            {"speed": Speed(speculate=False, acks=acks()), "slow": True},
        ),
    ]
    out = []
    for name, options in runs:
        first, answer, stages = run_turn(profile, delays, **options)
        out.append(Result(name, first, answer, stages))
    answers = AnswerCache()
    answers.put("bench", "What is the weather in Oslo today?", "It is sunny in Oslo today.")
    speed = Speed(speculate=False, answers=answers, tenant="bench")
    first, answer, stages = run_turn(profile, delays, speed=speed)
    out.append(Result("repeat question: answer cache", first, answer, stages))
    return out


# --- the pipeline's own CPU time per 20 ms frame ---


def frame_overhead(in_rate: int, frames: int = 20_000) -> dict[str, float]:
    async def measure() -> dict[str, float]:
        transport = MemoryTransport(in_rate=in_rate)
        session = VoiceSession(
            transport,
            stt=FakeSTT("hi", Delays()),
            tts=FakeTTS(Delays()),
            turns=SilenceTurns(start_ms=40, end_ms=500),
        )
        resampler = Resampler(in_rate, 16_000)
        speech, quiet = tone(20, rate=in_rate), silence(20, rate=in_rate)
        costs: list[float] = []
        for n in range(frames):
            frame = speech if (n // 50) % 2 == 0 else quiet  # 1 s talking, 1 s quiet
            start = time.perf_counter()
            session._frame(resampler.feed(frame))
            session.buffer.pull(640)
            costs.append((time.perf_counter() - start) * 1e6)
            if n % 50 == 0:
                await asyncio.sleep(0)
        costs.sort()
        return {
            "mean_us": round(statistics.fmean(costs), 1),
            "p99_us": round(costs[int(len(costs) * 0.99)], 1),
        }

    return asyncio.run(measure())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, value in vars(Delays()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    delays = Delays(**{name: getattr(args, name) for name in vars(Delays())})
    report: dict[str, Any] = {"delays_ms": vars(delays), "profiles": {}}
    for profile in PROFILES:
        rows = scenarios(profile, delays)
        report["profiles"][profile.name] = [vars(row) for row in rows]
    report["frame_overhead"] = {
        "16 kHz in": frame_overhead(16_000),
        "8 kHz phone in (resampled)": frame_overhead(8_000),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print_table(report)


def print_table(report: dict[str, Any]) -> None:
    print(f"VoxCat speed (ms after the caller's last speech; delays {report['delays_ms']})")
    for name, rows in report["profiles"].items():
        print(f"\n{name}")
        print(f"  {'scenario':40} {'first sound':>12} {'answer audio':>13}")
        for row in rows:
            first = _ms(row["first_sound_ms"])
            print(f"  {row['scenario']:40} {first:>12} {_ms(row['answer_ms']):>13}")
        saved = {r["scenario"]: r["answer_ms"] for r in rows}
        _saving(
            "speculative reply saves", saved, "plain: no speed tricks", "plain: speculative reply"
        )
        _saving(
            "speculative tools save",
            saved,
            "tool: speculative reply, tool waits",
            "tool: speculative reply + tool",
        )
        _saving(
            "both save on a tool question",
            saved,
            "tool: no speed tricks",
            "tool: speculative reply + tool",
        )
    print("\npipeline CPU per 20 ms frame (turn logic, playout, resampling):")
    for name, cost in report["frame_overhead"].items():
        print(f"  {name:28} {cost['mean_us']:7.1f} us mean {cost['p99_us']:8.1f} us p99")


def _saving(label: str, rows: dict[str, Any], before: str, after: str) -> None:
    if rows.get(before) is not None and rows.get(after) is not None:
        print(f"  {label}: {rows[before] - rows[after]:.0f} ms")


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


if __name__ == "__main__":
    main()
