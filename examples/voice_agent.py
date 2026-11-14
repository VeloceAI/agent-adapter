"""A voice agent offline: the caller talks over a booking, and a spoken yes approves a refund."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
)
from agent_adapter.exec import progress
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.testing import RecordingSpeaker, settle

registry = Registry()
tickets: list[str] = []


# --8<-- [start:tools]
@registry.tool(effects="destructive")
async def book_ticket(show: str) -> str:
    """Book a museum ticket."""
    progress("One moment, booking that")  # spoken while the tool works
    await asyncio.sleep(0.05)
    tickets.append(show)
    return f"ticket for {show}"


@registry.tool(effects="destructive", confirm=True)
async def refund_ticket(show: str) -> str:
    """Refund a museum ticket."""
    tickets.remove(show)
    return f"refunded {show}"


# --8<-- [end:tools]


def scripted_model() -> CallableModel:
    """Stands in for your model: books, books again after the cut, then refunds."""
    script = [
        ("book_ticket", {"show": "dinosaurs"}),
        ("book_ticket", {"show": "dinosaurs"}),  # re-issued after the interruption
        ("refund_ticket", {"show": "dinosaurs"}),
    ]

    async def answer(request: LLMRequest) -> LLMResponse:
        if request.messages[-1].role == "tool" or not script:
            reply = Message(role="assistant", content="All done.")
            return LLMResponse(provider="demo", model="demo", text="All done.", message=reply)
        name, args = script.pop(0)
        call = ToolCall(name=name, arguments=args, id=f"call_{len(script)}")
        message = Message(role="assistant", tool_calls=(call,))
        return LLMResponse(provider="demo", model="demo", tool_calls=(call,),
                           finish_reason="tool_calls", message=message)  # fmt: skip

    return CallableModel(answer, provider="demo", model="demo")


async def main() -> None:
    # --8<-- [start:agent]
    speaker = RecordingSpeaker()  # in production: your TTS, LiveKit, Pipecat, a phone line
    executor = Executor(registry, approval=VoiceApproval(timeout_s=5))
    agent = VoiceAgent(scripted_model(), executor, speaker, instructions="Help with tickets.")
    # --8<-- [end:agent]

    # --8<-- [start:events]
    await agent.user_said("Book the dinosaur show")
    await settle()
    await agent.user_started_speaking()  # the caller talks over the booking
    await agent.user_said("Did that work?")  # the model books again: the first result returns
    await agent.wait()
    # --8<-- [end:events]
    print("tickets after the cut:", tickets)

    await agent.user_said("Refund it")
    await settle()
    await agent.user_said("Yes please")  # only the caller's words can approve
    await agent.wait()
    print("tickets after the refund:", tickets)
    print("spoken:", " | ".join(speaker.said))
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
