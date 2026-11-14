"""LiveKit connector against a stand-in `livekit.agents`: tools, interruptions, voice bridge."""

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from agent_adapter import Registry
from agent_adapter.core.errors import MissingDependency, ToolError
from agent_adapter.exec import Executor, progress
from agent_adapter.integrations._voice_bridge import EventPump, spawn
from agent_adapter.integrations.livekit import (
    INTEGRATION,
    LiveKitSpeaker,
    attach_voice_agent,
    livekit_tool,
    livekit_tools,
)
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.testing import settle
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink
from tests.integrations.voice_fakes import LKToolError, RunContext, Session, install_livekit

registry = Registry()
booked: list[str] = []


@registry.tool(effects="destructive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment")
    booked.append(seat)
    return f"booking #{seat}"


@registry.tool(effects="read_only")
async def find(seat: str) -> str:
    """Look a seat up."""
    return "free"


@registry.tool(effects="read_only")
async def broken(seat: str) -> str:
    """Always fails."""
    raise ToolError("no such seat")


@pytest.fixture(autouse=True)
def _livekit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    install_livekit(monkeypatch)
    booked.clear()
    yield


def test_tools_carry_the_registry_schema():
    tools = {t.raw_schema["name"]: t for t in livekit_tools(registry, executor=Executor(registry))}
    assert set(tools) == {"book", "find", "broken"}
    assert tools["book"].raw_schema["description"] == "Book a seat."
    assert tools["book"].raw_schema["parameters"]["required"] == ["seat"]


async def test_a_tool_that_changes_things_holds_interruptions_and_speaks_progress():
    sink = RecordingSink()
    tool = livekit_tool(registry.get("book"), executor=Executor(registry, sink=sink))
    context = RunContext()
    assert await tool.handler({"seat": "4A"}, context) == "booking #4A"
    assert context.locked == 1
    assert context.session.said == [("One moment", {"add_to_chat_ctx": False})]
    assert [e.call_id for e in sink.events if e.type == "tool.started"] == ["lk_call_1"]


async def test_a_read_only_tool_leaves_interruptions_alone():
    context = RunContext()
    tool = livekit_tool(registry.get("find"), executor=Executor(registry))
    assert await tool.handler({"seat": "4A"}, context) == "free"
    assert context.locked == 0


async def test_an_effect_is_not_started_once_the_caller_cut_in():
    tool = livekit_tool(registry.get("book"), executor=Executor(registry))
    with pytest.raises(LKToolError, match="not run: the caller interrupted"):
        await tool.handler({"seat": "4A"}, RunContext(interrupted=True))
    assert booked == []


async def test_tool_errors_reach_the_model_as_livekit_tool_errors():
    tool = livekit_tool(registry.get("broken"), executor=Executor(registry))
    with pytest.raises(LKToolError, match="no such seat"):
        await tool.handler({"seat": "4A"}, RunContext())


def test_a_context_without_a_call_id_binds_an_empty_one():
    assert INTEGRATION.bind_context(SimpleNamespace()).call_id == ""
    assert INTEGRATION.export_tool(registry.get("find")).raw_schema["name"] == "find"


def test_missing_livekit_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(__import__("sys").modules, "livekit.agents")
    monkeypatch.delitem(__import__("sys").modules, "livekit")
    monkeypatch.setattr("agent_adapter.integrations.base._importable", lambda name: False)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[livekit\]"):
        livekit_tool(registry.get("find"))


async def test_the_session_drives_a_voice_agent_and_only_the_agent_cuts_speech():
    session = Session()
    model = ScriptedModel([calls(("book", {"seat": "4A"})), text("Booked. Anything else?")])
    agent = VoiceAgent(model, Executor(registry), LiveKitSpeaker(session))
    attach_voice_agent(session, agent)
    session.handlers["user_input_transcribed"](SimpleNamespace(transcript="hi", is_final=False))
    session.handlers["user_input_transcribed"](SimpleNamespace(transcript="book 4A", is_final=True))
    await settle()
    await agent.wait()
    assert booked == ["4A"]
    assert session.played == ["One moment", "Booked.", "Anything else?"]
    assert all(opts.get("allow_interruptions") is False for _, opts in session.said)
    cuts = len(session.interrupts)
    session.handlers["user_state_changed"](SimpleNamespace(new_state="listening"))
    session.handlers["user_state_changed"](SimpleNamespace(new_state="speaking"))
    await settle()
    assert session.interrupts[cuts:] == [True]
    await asyncio.wait_for(agent.close(), 1)


async def test_the_pump_skips_blank_text_and_outlives_a_failing_agent(caplog):
    seen: list[str] = []

    class Agent:
        async def user_started_speaking(self) -> None:
            raise RuntimeError("boom")

        async def user_said(self, text: str) -> None:
            seen.append(text)

    pump = EventPump(Agent())  # type: ignore[arg-type]
    pump.said("  ")
    pump.started_speaking()
    pump.said("hi")
    await settle()
    assert seen == ["hi"]
    assert "voice agent event failed" in caplog.text

    async def fails() -> None:
        raise RuntimeError("lost")

    spawn(fails())
    await settle()
    assert "voice bridge task failed" in caplog.text
