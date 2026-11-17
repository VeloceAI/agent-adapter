import asyncio
import math
import struct
import wave

from agent_adapter import Registry
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.ambience import ThinkingSound
from agent_adapter.voice.ending import IdlePolicy, end_call_tool
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.recorder import CallRecorder
from agent_adapter.voice.session import VoiceSession
from agent_adapter.voice.testing import (
    MemoryTransport,
    ScriptedSTT,
    WordTTS,
    settle,
    silence,
    tone,
)
from agent_adapter.voice.turns import SilenceTurns
from agent_adapter.voice.voicemail import VoicemailDetector, tone_of
from tests.fakes.models import ScriptedModel, calls, text

registry = Registry()
gate = asyncio.Event()


@registry.tool
async def slow_lookup(item: str) -> str:
    """Look an item up slowly."""
    await gate.wait()
    return f"{item}: in stock"


def call(model, transcripts, *, agent_options=None, tools=None, **options):
    global gate
    gate = asyncio.Event()
    transport = MemoryTransport()
    tts = WordTTS(ms_per_word=100)
    session = VoiceSession(
        transport,
        stt=ScriptedSTT(transcripts),
        tts=tts,
        turns=SilenceTurns(start_ms=40, end_ms=200),
        **options,
    )
    reg = registry
    if tools:
        reg = Registry()
        reg.add_many([*registry, *(t(session) for t in tools)])
    agent = VoiceAgent(model, Executor(reg), session.speaker, **(agent_options or {}))
    run = asyncio.create_task(session.run(agent))
    return transport, session, agent, tts, run


async def turn(transport: MemoryTransport, ms: float = 400, *, voice=tone) -> None:
    transport.speak(voice(ms) + silence(260))
    await settle()


def babble(ms: float) -> bytes:
    """Loud, uneven audio: speech to the turn detector, never a steady tone."""
    count, seed, out = int(16 * ms), 7, []
    for _ in range(count):
        seed = (seed * 1103515245 + 12345) % 2**31
        out.append(seed % 16000 - 8000)
    return struct.pack(f"<{count}h", *out)


async def play(transport: MemoryTransport, ms: float) -> None:
    for _ in range(int(ms // 20)):
        transport.play(20)
        await settle(5)


async def finish(transport, run) -> None:
    transport.hang_up()
    await asyncio.wait_for(run, 2)


async def test_typed_text_is_a_turn():
    transport, session, agent, tts, run = call(ScriptedModel([text("Typed back.")]), [])
    await settle()
    session.send_text("hello from the keyboard")
    await settle()
    await play(transport, 300)
    assert tts.spoken == ["Typed back."]
    assert agent.history[0].content == "hello from the keyboard"
    await finish(transport, run)


async def test_the_agent_hangs_up_after_its_goodbye():
    model = ScriptedModel([calls(("end_call", {})), text("Goodbye now.")])
    transport, _, _, tts, run = call(model, ["that's all"], tools=[end_call_tool])
    await turn(transport)
    for _ in range(40):
        if run.done():
            break
        await play(transport, 100)
    await asyncio.wait_for(run, 2)
    assert tts.spoken == ["Goodbye now."] and transport.closed


async def test_a_line_said_without_interruptions_is_not_cut():
    transport, session, agent, tts, run = call(ScriptedModel([text("Unused.")]), ["stop"])
    await settle()
    agent.say("Welcome to the line, this message plays in full.", interruptible=False)
    await settle()
    await play(transport, 200)
    await turn(transport)
    assert not session.buffer.paused and session.timings == []
    await play(transport, 1200)
    assert agent.history[-1].content == "Welcome to the line, this message plays in full."
    assert tts.spoken == ["Welcome to the line, this message plays in full."]
    await finish(transport, run)


async def test_say_records_only_what_was_heard_when_cut():
    model = ScriptedModel([text("Sure.")])
    transport, _, agent, _, run = call(model, ["wait"])
    await settle()
    agent.say("One two three four five six seven eight nine ten.")
    await settle()
    await play(transport, 300)
    await turn(transport)
    await play(transport, 300)
    assert agent.history[0].role == "assistant"
    assert agent.history[0].content.startswith("One two")
    assert "ten." not in agent.history[0].content
    await finish(transport, run)


async def test_mute_during_tools_drops_speech_while_a_tool_runs():
    model = ScriptedModel([calls(("slow_lookup", {"item": "tea"})), text("Tea is in stock.")])
    transport, session, agent, tts, run = call(
        model, ["is tea in stock", "hello?"], agent_options={"mute_during_tools": True}
    )
    await turn(transport)
    assert agent.busy and not agent.interruptible
    await turn(transport)
    gate.set()
    await settle()
    await play(transport, 500)
    assert tts.spoken == ["Tea is in stock."]
    assert len(session.timings) == 1
    await finish(transport, run)


async def test_idle_caller_is_prompted_then_hung_up():
    idle = IdlePolicy(after_s=0.15, hang_up_after=1, check_s=0.02, prompt="Still there?")
    transport, _session, _agent, tts, run = call(ScriptedModel([]), [], observers=[idle])
    await settle()
    for _ in range(100):
        if run.done():
            break
        await play(transport, 40)
        await asyncio.sleep(0.02)
    await asyncio.wait_for(run, 2)
    assert tts.spoken == ["Still there?", "I'll hang up now. Goodbye."]
    assert idle.prompts == 1


async def test_recorder_writes_both_sides_in_stereo(tmp_path):
    path = tmp_path / "call.wav"
    recorder = CallRecorder(path)
    transport, _session, _agent, _tts, run = call(
        ScriptedModel([text("Hello caller.")]), ["hi"], observers=[recorder]
    )
    await turn(transport)
    await play(transport, 400)
    await finish(transport, run)
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2 and wav.getframerate() == 16_000
        frames = wav.readframes(wav.getnframes())
    left = [abs(s) for s in struct.unpack(f"<{len(frames) // 2}h", frames)[0::2]]
    right = [abs(s) for s in struct.unpack(f"<{len(frames) // 2}h", frames)[1::2]]
    assert max(left) > 1000 and max(right) > 1000


async def test_thinking_sound_fills_a_slow_tool_and_stops_for_speech():
    model = ScriptedModel([calls(("slow_lookup", {"item": "tea"})), text("Found it.")])
    sound = ThinkingSound.ticks(delay_s=0.05, chunk_ms=40)
    transport, session, _agent, tts, run = call(model, ["tea?"], observers=[sound])
    await turn(transport)
    heard = b""
    for _ in range(10):
        heard += transport.play(40)
        await asyncio.sleep(0.03)
    assert any(heard) and not session.speaker.speaking  # ambient audio is not speech
    gate.set()
    await settle()
    await play(transport, 600)
    assert tts.spoken == ["Found it."]
    assert session.timings[0].voice_to_voice_ms is not None
    await finish(transport, run)


def beep(ms: float, hz: float = 1000) -> bytes:
    count = int(16 * ms)
    return struct.pack(
        f"<{count}h", *(int(9000 * math.sin(2 * math.pi * hz * i / 16_000)) for i in range(count))
    )


def test_tone_of_finds_a_beep_and_not_noise():
    assert abs(tone_of(beep(20, 1000)) - 1000) < 20
    assert tone_of(babble(20)) == 0.0
    assert tone_of(silence(20)) == 0.0
    noise = struct.pack("<320h", *((i * 7919) % 20000 - 10000 for i in range(320)))
    assert tone_of(noise) == 0.0


async def test_voicemail_greeting_then_beep_leaves_the_message():
    found: list[str] = []
    detector = VoicemailDetector(
        message="Please call us back.", on_detect=lambda s, why: found.append(why)
    )
    transport, _session, agent, tts, run = call(
        ScriptedModel([]),
        ["Hi, I can't take your call, please leave a message"],
        observers=[detector],
    )
    await turn(transport, 600, voice=babble)
    assert found == ["greeting"] and agent.history == []
    transport.speak(beep(300))
    await settle()
    for _ in range(40):
        if run.done():
            break
        await play(transport, 100)
    await asyncio.wait_for(run, 2)
    assert found == ["greeting", "beep"] and tts.spoken == ["Please call us back."]


async def test_a_person_is_not_voicemail():
    detector = VoicemailDetector(message="Unused.")
    transport, _session, _agent, tts, run = call(
        ScriptedModel([text("Hi, this is the clinic.")]), ["hello?"], observers=[detector]
    )
    await turn(transport, voice=babble)
    await play(transport, 400)
    assert not detector.machine and tts.spoken == ["Hi, this is the clinic."]
    await finish(transport, run)


async def test_a_failing_observer_does_not_break_the_call():
    class Broken(CallObserver):
        def caller_said(self, text: str) -> bool:
            raise RuntimeError("boom")

        async def ended(self) -> None:
            raise RuntimeError("boom")

    transport, _session, _agent, tts, run = call(
        ScriptedModel([text("Still fine.")]), ["hi"], observers=[Broken()]
    )
    await turn(transport)
    await play(transport, 300)
    assert tts.spoken == ["Still fine."]
    await finish(transport, run)
