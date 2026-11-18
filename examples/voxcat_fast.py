"""VoxCat with no provider tricks, offline: its own end of turn (EagerTurns), the first words
spoken before the sentence ends, and a whole call from one config."""

import asyncio
from collections.abc import AsyncIterator

from agent_adapter import CallableModel, Executor, LLMRequest, LLMResponse, Message, Registry
from agent_adapter.models import StreamDone, TextDelta
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.config import VoiceConfig
from agent_adapter.voice.endpoint import EagerTurns, looks_done
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import MemoryTransport, RecordingSpeaker, settle, silence, tone
from agent_adapter.voice.turns import SilenceTurns

REPLY = "Sure, I can book a table for two at seven tonight. Anything else?"


async def stream(request: LLMRequest) -> AsyncIterator[TextDelta | StreamDone]:
    """Stands in for your model: streams the reply a word at a time."""
    for i, word in enumerate(REPLY.split()):
        await asyncio.sleep(0.01)
        yield TextDelta(text=word if i == 0 else " " + word)
    message = Message(role="assistant", content=REPLY)
    yield StreamDone(
        response=LLMResponse(provider="demo", model="demo", text=REPLY, message=message)
    )


model = CallableModel(lambda request: REPLY, provider="demo", model="demo", stream_fn=stream)


def frames_until_ended(turns: EagerTurns, pcm: bytes) -> int | None:
    for i in range(0, len(pcm), 640):  # 20 ms frames of 16 kHz audio
        if "ended" in turns.feed(pcm[i : i + 640]):
            return (i + 640) // 32
    return None


def eager_turns() -> None:
    # --8<-- [start:eager]
    print(looks_done("Book a table for two."), looks_done("Book a table for"))  # True False
    turns = EagerTurns(SilenceTurns(end_ms=600), done_ms=250)
    turns.feed(tone(800))  # the caller speaks
    turns.hint("Book a table for two.")  # VoiceSession passes the early transcript
    print("ended after", frames_until_ended(turns, silence(700)), "ms of quiet")  # not 600
    # --8<-- [end:eager]


async def first_words() -> None:
    # --8<-- [start:first-words]
    speaker = RecordingSpeaker()
    agent = VoiceAgent(model, Executor(Registry()), speaker, speed=Speed(first_words=4))
    await agent.user_said("Can I book a table?")
    await agent.wait()
    print("spoken lines:", speaker.said)  # "Sure," goes out alone, at once
    # --8<-- [end:first-words]
    await agent.close()


async def from_config() -> None:
    # --8<-- [start:config]
    config = VoiceConfig.load(
        {
            "stt": {"use": "agent_adapter.voice.testing:ScriptedSTT", "texts": ["Book a table."]},
            "tts": {"use": "agent_adapter.voice.testing:WordTTS", "ms_per_word": 20},
            "model": "__main__:model",  # or {"use": "openai", "model": "gpt-4.1-mini"}
            "instructions": "You book tables. Keep answers short.",
            "turns": {"end_ms": 600, "eager": True},
            "speed": {"first_words": 4},
        }
    )
    print("\n".join(config.describe()))
    transport = MemoryTransport()
    session, agent = config.build(transport)  # or: agent-adapter voice serve voxcat.yaml
    # --8<-- [end:config]
    run = asyncio.create_task(session.run(agent))
    caller = tone(600) + silence(500)
    for i in range(0, len(caller), 640):
        transport.speak(caller[i : i + 640])
        await settle(5)
    for _ in range(40):
        transport.play(20)
        await settle(5)
    print("agent said:", session.speaker.tts.spoken)
    transport.hang_up()
    await asyncio.wait_for(run, 5)


if __name__ == "__main__":
    eager_turns()
    asyncio.run(first_words())
    asyncio.run(from_config())
