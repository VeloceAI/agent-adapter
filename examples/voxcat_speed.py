"""VoxCat speed tricks offline: a speculative reply, an early read-only tool, an
acknowledgement and a cached answer, with fake models and no audio."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    MemoryCache,
    Message,
    Registry,
    ToolCall,
)
from agent_adapter.voice import TurnTiming, VoiceAgent
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.answers import AnswerCache
from agent_adapter.voice.speculate import SpeculativeTools
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import RecordingSpeaker, settle
from agent_adapter.voice.warm import Warmup

registry = Registry()
calls: list[str] = []


# --8<-- [start:tools]
@registry.tool(effects="read_only", cache_ttl=60)  # may run before the caller has finished
async def weather(city: str) -> str:
    """Today's weather for a city."""
    calls.append(f"weather {city}")
    return f"{city}: sunny, 21 C"


@registry.tool(effects="destructive")  # never runs until the turn is final
async def book_table(time: str) -> str:
    """Book a table."""
    calls.append(f"book {time}")
    return f"booked {time}"


# --8<-- [end:tools]


def model(delay_s: float = 0.0) -> CallableModel:
    """Stands in for your model: answers after `delay_s`; looks up the weather when asked."""

    async def answer(request: LLMRequest) -> LLMResponse | str:
        await asyncio.sleep(delay_s)
        last = request.messages[-1]
        if last.role == "user" and "weather" in last.content.lower():
            call = ToolCall(name="weather", arguments={"city": "Oslo"}, id="call_1")
            message = Message(role="assistant", tool_calls=(call,))
            return LLMResponse(provider="demo", model="demo", tool_calls=(call,),
                               finish_reason="tool_calls", message=message)  # fmt: skip
        return "It is sunny." if last.role == "tool" else "Here you go."

    return CallableModel(answer, provider="demo", model="demo")


async def speculative_reply() -> None:
    # --8<-- [start:speculate]
    speed = Speed(tools=SpeculativeTools(["weather"], max_per_turn=2, max_per_minute=30))
    agent = VoiceAgent(model(), Executor(registry, cache=MemoryCache()), RecordingSpeaker(),
                       speed=speed)  # fmt: skip
    agent.speculate("weather in oslo")  # the early transcript: the reply starts, held back
    await settle()
    print("ran before the turn was final:", calls)
    timing = TurnTiming(started=0.0)
    await agent.user_said("Weather in Oslo?", timing=timing)  # same words: kept
    await agent.wait()
    print("kept the early reply:", timing.adopted)
    # --8<-- [end:speculate]
    await agent.close()


async def acknowledgement() -> None:
    # --8<-- [start:acks]
    acks = Acknowledgements(["Mm-hm.", "Okay."], fast_ms=300, expect_ms=700)
    speaker = RecordingSpeaker()
    agent = VoiceAgent(model(delay_s=0.4), Executor(registry), speaker, speed=Speed(acks=acks))
    await agent.warm_up()  # renders the lines; a VoiceSession does this at call start
    await agent.user_said("What is on the menu?")
    await agent.wait()
    print("heard:", speaker.said)
    # --8<-- [end:acks]
    await agent.close()


async def answer_cache() -> None:
    # --8<-- [start:answers]
    answers = AnswerCache(ttl_s=300)  # one per process; wrap your TTS with answers.voice(tts)
    answers.put("acme", "When do you open?", "We open at nine.")
    speed = Speed(answers=answers, tenant="acme", warmup=Warmup(lines=["Hello, Acme here."]))
    speaker = RecordingSpeaker()
    agent = VoiceAgent(model(delay_s=1.0), Executor(registry), speaker, speed=speed)
    print("warm-up steps:", sorted((await agent.warm_up()).ms))
    await agent.user_said("when do you open")  # no model call: the stored reply plays
    await agent.wait()
    print("heard:", speaker.said, "cache hits:", answers.hits)
    # --8<-- [end:answers]
    await agent.close()


async def main() -> None:
    await speculative_reply()
    await acknowledgement()
    await answer_cache()


if __name__ == "__main__":
    asyncio.run(main())
