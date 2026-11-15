"""The LiveKit connector against the real `livekit-agents` objects (no room, no network)."""

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import RunContext, llm
from livekit.agents.llm import ToolError
from livekit.agents.voice import SpeechHandle

from agent_adapter import Registry
from agent_adapter.exec import Executor, progress
from agent_adapter.integrations.livekit import livekit_tool

pytestmark = pytest.mark.contract

registry = Registry()
booked: list[str] = []


@registry.tool(effects="destructive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment")
    booked.append(seat)
    return f"booking #{seat}"


class Session:
    """Only what RunContext and the connector read from an AgentSession."""

    _global_run_state = None

    def __init__(self) -> None:
        self.said: list[tuple[str, dict]] = []

    def say(self, text: str, **options: object) -> None:
        self.said.append((text, options))


def context(handle: SpeechHandle, session: Session) -> RunContext:
    call = llm.FunctionCall(call_id="c1", name="book", arguments='{"seat": "4A"}')
    return RunContext(session=session, speech_handle=handle, function_call=call)


def test_the_tool_is_a_real_raw_function_tool_with_the_registry_schema():
    tool = livekit_tool(registry.get("book"), executor=Executor(registry))
    assert type(tool).__name__ == "RawFunctionTool"
    assert tool.info.raw_schema["name"] == "book"
    assert tool.info.raw_schema["parameters"]["required"] == ["seat"]


async def test_an_effect_holds_interruptions_speaks_progress_and_keeps_the_call_id():
    booked.clear()
    tool = livekit_tool(registry.get("book"), executor=Executor(registry))
    handle, session = SpeechHandle.create(allow_interruptions=True), Session()
    assert await tool._func({"seat": "4A"}, context(handle, session)) == "booking #4A"
    assert handle.allow_interruptions is False
    assert session.said == [("One moment", {"add_to_chat_ctx": False})]


async def test_an_effect_is_not_started_after_the_caller_cut_in():
    booked.clear()
    tool = livekit_tool(registry.get("book"), executor=Executor(registry))
    handle = SpeechHandle.create(allow_interruptions=True)
    handle.interrupt()
    with pytest.raises(ToolError, match="not run"):
        await tool._func({"seat": "4A"}, context(handle, Session()))
    assert booked == []
