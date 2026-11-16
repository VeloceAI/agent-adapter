import asyncio

import pytest

from agent_adapter import Registry
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.session import VoiceSession, _timer_of
from agent_adapter.voice.testing import (
    MemoryTransport,
    ScriptedSTT,
    WordTTS,
    settle,
    silence,
    tone,
)
from agent_adapter.voice.turns import EchoGuard, SilenceTurns
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink

charged: list[str] = []
gate = asyncio.Event()
registry = Registry()


@registry.tool(effects="destructive")
async def book(show: str) -> str:
    """Book a show."""
    await gate.wait()
    charged.append(show)
    return f"booked {show}"


@registry.tool(effects="destructive", confirm=True)
async def refund(order: str) -> str:
    """Refund an order."""
    charged.append(f"refund {order}")
    return "refunded"


def call(model, transcripts, *, approval=None, sink=None, confirm_ms=300, **options):
    global gate
    gate = asyncio.Event()
    charged.clear()
    transport = MemoryTransport()
    stt, tts = ScriptedSTT(transcripts), WordTTS(ms_per_word=100)
    session = VoiceSession(
        transport,
        stt=stt,
        tts=tts,
        turns=SilenceTurns(start_ms=40, end_ms=200),
        confirm_ms=confirm_ms,
        **options,
    )
    executor = Executor(registry, approval=approval, sink=sink)
    agent = VoiceAgent(model, executor, session.speaker)
    run = asyncio.create_task(session.run(agent))
    return transport, session, agent, tts, run


async def turn(transport: MemoryTransport, ms: float = 400) -> None:
    transport.speak(tone(ms) + silence(260))
    await settle()


async def play(transport: MemoryTransport, ms: float) -> None:
    for _ in range(int(ms // 20)):
        transport.play(20)
        await settle(5)


async def finish(transport, run) -> None:
    transport.hang_up()
    await asyncio.wait_for(run, 2)


async def test_a_turn_is_heard_answered_and_timed():
    sink = RecordingSink()
    transport, session, _agent, tts, run = call(
        ScriptedModel([text("Hi there.")]), ["hello"], sink=sink
    )
    await turn(transport)
    await play(transport, 300)
    assert tts.spoken == ["Hi there."]
    timing = session.timings[0]
    assert timing.text == "hello" and timing.cut_ms is not None
    assert timing.voice_to_voice_ms is not None and timing.voice_to_voice_ms >= 0
    events = [e for e in sink.events if e.type == "custom.voice_turn"]
    assert events and "text" not in events[0].data  # the caller's words stay out of traces
    await finish(transport, run)
    assert transport.closed


async def test_the_turn_audio_has_a_pre_roll():
    transport, session, _, _, run = call(ScriptedModel([text("Ok.")]), ["hi"])
    stt = session.stt
    transport.speak(silence(400))
    await turn(transport, 400)
    assert isinstance(stt, ScriptedSTT)
    assert stt.heard[0] >= (400 + 300) * 32  # the speech plus 300 ms from before it started
    await finish(transport, run)


async def test_playback_hushes_at_once_and_a_cough_resumes_it():
    model = ScriptedModel([text("This is a long answer with many words in it.")])
    transport, session, agent, _, run = call(model, ["tell me", ""])
    await turn(transport)
    await play(transport, 200)
    transport.speak(tone(100))
    await settle()
    assert session.buffer.paused  # hushed within the frame that started the cut
    transport.speak(silence(260))
    await settle()
    assert not session.buffer.paused and session.timings[-1].resumed
    assert session.timings[-1].confirmed is None
    await play(transport, 1000)
    assert "many words in it." in agent.history[-1].content
    await finish(transport, run)


async def test_a_backchannel_while_speaking_does_not_cut():
    model = ScriptedModel([text("Here is a fairly long answer for you to hear.")])
    transport, session, agent, _, run = call(model, ["tell me", "uh-huh"], confirm_ms=1000)
    await turn(transport)
    await play(transport, 200)
    await turn(transport, 250)
    assert session.timings[-1].resumed
    await play(transport, 1000)
    assert agent.history[-1].content == "Here is a fairly long answer for you to hear."
    await finish(transport, run)


async def test_real_speech_cuts_and_a_started_booking_finishes_once():
    model = ScriptedModel([calls(("book", {"show": "jazz"})), text("Yes, it is booked.")])
    transport, session, _agent, _, run = call(model, ["book jazz", "did that go through"])
    await turn(transport)
    transport.speak(tone(400))
    await settle()
    assert session.timings[-1].confirmed is not None  # cut on speech length, before words
    gate.set()
    transport.speak(silence(260))
    await settle()
    await play(transport, 1000)
    assert charged == ["jazz"]
    assert any("Already done" in str(m.content) for m in model.requests[-1].messages)
    await finish(transport, run)


async def test_a_short_answer_with_words_still_cuts_and_is_delivered():
    transport, session, agent, _, run = call(ScriptedModel([text("A."), text("B.")]), ["one", "no"])
    await turn(transport)
    await play(transport, 200)
    await turn(transport, 100)  # shorter than confirm_ms, but it has words
    assert session.timings[-1].confirmed is not None
    await play(transport, 400)
    assert [m.content for m in agent.history if m.role == "user"] == ["one", "no"]
    await finish(transport, run)


async def test_a_yes_spoken_while_the_prompt_plays_does_not_answer_it():
    approval = VoiceApproval(timeout_s=0.3)
    model = ScriptedModel([calls(("refund", {"order": "o-1"})), text("Not refunded.")])
    transport, _session, _agent, tts, run = call(model, ["refund o-1", "yes"], approval=approval)
    await turn(transport)
    await play(transport, 100)
    await turn(transport, 150)  # "yes" over the prompt: dropped, and the prompt plays on
    await play(transport, 1500)
    await asyncio.sleep(0.4)
    await play(transport, 1500)
    assert charged == []
    assert tts.spoken[-1] == "Not refunded."
    await finish(transport, run)


async def test_yes_after_the_prompt_has_played_approves():
    approval = VoiceApproval(timeout_s=5)
    model = ScriptedModel([calls(("refund", {"order": "o-1"})), text("Refunded.")])
    transport, _session, _agent, tts, run = call(model, ["refund o-1", "yes"], approval=approval)
    await turn(transport)
    await play(transport, 1500)
    assert tts.spoken == ["Shall I run refund with order o-1? Say yes to confirm."]
    await turn(transport, 150)
    await play(transport, 1000)
    assert charged == ["refund o-1"]
    await finish(transport, run)


async def test_a_turn_that_never_pauses_is_cut_at_max_turn_s():
    transport, session, agent, _, run = call(
        ScriptedModel([text("Okay.")]), ["a very long turn"], max_turn_s=1
    )
    transport.speak(tone(2000))
    await settle()
    await play(transport, 600)
    assert len(session.stt.heard) == 1 and session.stt.heard[0] <= 16_000 * 2
    assert agent.history[0].content == "a very long turn"
    await finish(transport, run)


async def test_stt_failure_resumes_playback():
    class Broken:
        async def transcribe(self, pcm: bytes, rate: int) -> str:
            raise RuntimeError("stt down")

    transport, session, _agent, _, run = call(ScriptedModel([]), [])
    session.stt = Broken()
    await turn(transport, 100)
    assert not session.buffer.paused
    await finish(transport, run)


def test_timer_is_found_inside_wrapping_detectors():
    inner = SilenceTurns()

    class Wrapper:
        detector = inner

    assert _timer_of(inner) is inner.timer
    assert _timer_of(Wrapper()) is inner.timer
    assert _timer_of(object()) is None


class FakeStream:
    def __init__(self, owner: "StreamingFake") -> None:
        self.owner = owner
        self.fed = 0
        self.cancelled = False

    def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    async def finish(self) -> str:
        self.owner.finished_at.append(len(self.owner.events))
        await asyncio.sleep(self.owner.delay)
        if self.owner.fail:
            raise RuntimeError("stream broke")
        return self.owner.texts.pop(0) if self.owner.texts else ""

    async def cancel(self) -> None:
        self.cancelled = True


class StreamingFake:
    def __init__(self, texts, *, fail=False, delay=0.0) -> None:
        self.texts = list(texts)
        self.fail = fail
        self.delay = delay
        self.streams: list[FakeStream] = []
        self.batch: list[int] = []
        self.finished_at: list[int] = []
        self.events: list[str] = []

    def open(self, sample_rate: int) -> FakeStream:
        self.streams.append(FakeStream(self))
        return self.streams[-1]

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.batch.append(len(pcm))
        return "batch text"


def streaming_call(stt, model):
    transport = MemoryTransport()
    turns = SilenceTurns(start_ms=40, end_ms=400)
    session = VoiceSession(transport, stt=stt, tts=WordTTS(), turns=turns, early_ms=100)
    original = session._ended

    def ended(now: float) -> None:
        stt.events.append("ended")
        original(now)

    session._ended = ended
    agent = VoiceAgent(model, Executor(registry), session.speaker)
    return transport, session, agent, asyncio.create_task(session.run(agent))


async def test_streaming_stt_hears_the_turn_and_finishes_before_the_turn_ends():
    stt = StreamingFake(["hello"])
    transport, session, _, run = streaming_call(stt, ScriptedModel([text("Hi.")]))
    transport.speak(tone(400) + silence(200))
    await settle()
    assert stt.finished_at == [0]  # the final pass began before the detector said "ended"
    transport.speak(silence(300))
    await settle()
    assert stt.events == ["ended"] and stt.batch == []
    assert stt.streams[0].fed >= 400 * 32 and session.timings[0].text == "hello"
    await play(transport, 300)
    await finish(transport, run)


async def test_speech_after_the_early_pass_drops_it_and_hears_the_whole_turn():
    stt = StreamingFake(["book the jazz show"], delay=0.05)
    transport, session, _, run = streaming_call(stt, ScriptedModel([text("Ok.")]))
    transport.speak(tone(300) + silence(200) + tone(300) + silence(500))
    await asyncio.sleep(0.1)
    await settle()
    first, second = stt.streams
    assert first.cancelled and second.fed > first.fed
    assert session.timings[0].text == "book the jazz show"
    await finish(transport, run)


async def test_a_broken_stream_falls_back_to_the_whole_turn():
    stt = StreamingFake([], fail=True)
    transport, session, _, run = streaming_call(stt, ScriptedModel([text("Ok.")]))
    transport.speak(tone(300) + silence(500))
    await settle()
    assert stt.batch and session.timings[0].text == "batch text"
    await finish(transport, run)


LONG = "This is a long answer with many words in it for the echo to talk over."
GUARD = EchoGuard(level=0.1, loud_ms=100, speech_ms=600)


async def test_echo_guard_ignores_the_agents_own_voice_in_the_mic():
    transport, session, agent, _, run = call(
        ScriptedModel([text(LONG)]), ["tell me", "echo words"], echo_guard=GUARD
    )
    await turn(transport)
    await play(transport, 200)
    transport.speak(tone(400, level=2_000) + silence(260))  # quiet and short: echo
    await settle()
    assert not session.buffer.paused and len(session.timings) == 1
    assert isinstance(session.stt, ScriptedSTT) and len(session.stt.heard) == 1
    await play(transport, 2000)
    assert agent.history[-1].content == LONG
    await finish(transport, run)


async def test_echo_guard_lets_a_loud_caller_cut_in():
    transport, session, agent, _, run = call(
        ScriptedModel([text(LONG), text("Ok.")]), ["tell me", "stop please"], echo_guard=GUARD
    )
    await turn(transport)
    await play(transport, 200)
    transport.speak(tone(140))
    await settle()
    assert session.buffer.paused and len(session.timings) == 2
    await turn(transport, 300)
    assert session.timings[1].confirmed is not None
    await play(transport, 400)
    assert [m.content for m in agent.history if m.role == "user"][-1] == "stop please"
    await finish(transport, run)


async def test_echo_guard_lets_long_quiet_speech_cut_in():
    transport, session, _agent, _, run = call(
        ScriptedModel([text(LONG), text("Ok.")]), ["tell me", "wait"], echo_guard=GUARD
    )
    await turn(transport)
    await play(transport, 200)
    transport.speak(tone(500, level=2_000))
    await settle()
    assert not session.buffer.paused
    transport.speak(tone(200, level=2_000))
    await settle()
    assert len(session.timings) == 2 and session.timings[1].confirmed is not None
    await finish(transport, run)


async def test_echo_guard_does_not_apply_while_the_agent_is_quiet():
    transport, session, _agent, _, run = call(
        ScriptedModel([text("Hi.")]), ["hello"], echo_guard=GUARD
    )
    transport.speak(tone(300, level=2_000) + silence(260))
    await settle()
    assert session.timings and session.timings[0].text == "hello"
    await finish(transport, run)


def test_echo_guard_rejects_bad_settings():
    for bad in ({"level": 0}, {"loud_ms": 0}, {"speech_ms": -1}):
        with pytest.raises(ValueError):
            EchoGuard(**bad)


class KeypadTransport(MemoryTransport):
    def __init__(self) -> None:
        super().__init__()
        self.on_digit: list = []

    def press(self, digits: str) -> None:
        for digit in digits:
            for callback in self.on_digit:
                callback(digit)


async def test_keypad_digits_reach_the_agent_as_words():
    transport = KeypadTransport()
    session = VoiceSession(
        transport, stt=ScriptedSTT([]), tts=WordTTS(), turns=SilenceTurns(), keypad_wait_ms=30
    )
    model = ScriptedModel([text("Got it."), text("Thanks.")])
    agent = VoiceAgent(model, Executor(registry), session.speaker)
    run = asyncio.create_task(session.run(agent))
    await settle()
    transport.press("12#")  # "#" sends at once
    await settle()
    await play(transport, 400)
    transport.press("5")
    await asyncio.sleep(0.06)  # the pause after the last digit sends it
    await settle()
    await play(transport, 400)
    users = [m.content for m in agent.history if m.role == "user"]
    assert users == ["(keypad) 12#", "(keypad) 5"]
    transport.press("9")  # pending when the call ends: dropped with the call
    await finish(transport, run)


async def test_a_long_run_of_digits_is_sent_at_max_digits():
    transport = KeypadTransport()
    session = VoiceSession(
        transport, stt=ScriptedSTT([]), tts=WordTTS(), turns=SilenceTurns(), max_digits=4
    )
    agent = VoiceAgent(ScriptedModel([text("One."), text("Two.")]), Executor(registry),
                       session.speaker)  # fmt: skip
    run = asyncio.create_task(session.run(agent))
    await settle()
    transport.press("1234")  # the fourth digit sends at once, with no "#" and no pause
    await settle()
    await play(transport, 400)
    assert [m.content for m in agent.history if m.role == "user"] == ["(keypad) 1234"]
    await finish(transport, run)
