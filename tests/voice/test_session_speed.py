import asyncio

from agent_adapter import CallableModel, Registry
from agent_adapter.core.messages import LLMRequest
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.session import VoiceSession
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import MemoryTransport, ScriptedSTT, WordTTS, settle, silence, tone
from agent_adapter.voice.turns import SilenceTurns
from tests.fakes.sinks import RecordingSink


class EchoModel(CallableModel):
    """Replies with the caller's last words after `delay` seconds; records each question."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.asked: list[str] = []
        super().__init__(self._reply)

    async def _reply(self, request: LLMRequest) -> str:
        question = request.messages[-1].content
        self.asked.append(question)
        await asyncio.sleep(self.delay)
        return f"You said {question}."


def start(model, transcripts, *, sink=None, **speed):
    transport = MemoryTransport()
    session = VoiceSession(
        transport,
        stt=ScriptedSTT(transcripts),
        tts=WordTTS(ms_per_word=20),
        turns=SilenceTurns(start_ms=40, end_ms=200),
        early_ms=60,
    )
    executor = Executor(Registry(), sink=sink)
    agent = VoiceAgent(model, executor, session.speaker, speed=Speed(**speed))
    return transport, session, agent, asyncio.create_task(session.run(agent))


async def speak(transport: MemoryTransport, pcm: bytes) -> None:
    for i in range(0, len(pcm), 640):
        transport.speak(pcm[i : i + 640])
        await settle(5)


async def play(transport: MemoryTransport, ms: float) -> None:
    for _ in range(int(ms // 20)):
        transport.play(20)
        await settle(5)


async def test_the_early_transcript_starts_the_reply_and_the_final_one_keeps_it():
    sink = RecordingSink()
    model = EchoModel()
    transport, session, agent, run = start(model, ["hello there"], sink=sink)
    await speak(transport, tone(300) + silence(100))
    assert model.asked == ["hello there"]  # thinking began before the turn ended
    await speak(transport, silence(160))
    await play(transport, 400)
    timing = session.timings[0]
    assert timing.adopted and timing.early is not None and timing.speculated is not None
    assert timing.speculated <= timing.ended  # type: ignore[operator]
    assert set(timing.stages()) >= {"early", "ended", "speculated", "replied", "first_audio"}
    assert model.asked == ["hello there"] and agent.history[-1].content == "You said hello there."
    event = sink.of_type("custom.voice_turn")[0]
    assert event.data["adopted"] and "first_audio" in event.data["stages_ms"]
    transport.hang_up()
    await asyncio.wait_for(run, 2)


async def test_when_the_caller_goes_on_the_early_reply_is_dropped():
    model = EchoModel(delay=0.05)
    transport, session, agent, run = start(model, ["weather in", "weather in oslo"])
    await speak(transport, tone(300) + silence(100))
    await speak(transport, tone(300) + silence(260))
    await asyncio.sleep(0.1)
    await play(transport, 600)
    assert model.asked == ["weather in", "weather in oslo"]
    assert session.speaker.tts.spoken == ["You said weather in oslo."]  # type: ignore[attr-defined]
    assert [m.content for m in agent.history] == ["weather in oslo", "You said weather in oslo."]
    transport.hang_up()
    await asyncio.wait_for(run, 2)


async def test_the_session_warms_acknowledgements_at_call_start():
    acks = Acknowledgements(["Mm-hm."])
    transport, session, _agent, run = start(EchoModel(), [], acks=acks)
    await settle()
    await asyncio.sleep(0.02)
    assert acks.rendered == {"Mm-hm."}
    assert session.speaker.tts.spoken == ["Mm-hm."]  # type: ignore[attr-defined]
    transport.hang_up()
    await asyncio.wait_for(run, 2)
