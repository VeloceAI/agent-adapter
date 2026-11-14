"""Pipecat connector against a stand-in `pipecat`: tools, interruption flags, processor."""

import asyncio
from collections.abc import Iterator

import pytest

from agent_adapter import Registry
from agent_adapter.core.errors import ToolError
from agent_adapter.exec import Executor, progress
from agent_adapter.integrations.pipecat import INTEGRATION, pipecat_tools, voice_processor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.testing import settle
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink
from tests.integrations.voice_fakes import (
    InterimTranscriptionFrame,
    LLMService,
    Params,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    install_pipecat,
)

registry = Registry()
booked: list[str] = []


@registry.tool(effects="additive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment")
    booked.append(seat)
    return f"booking #{seat}"


@registry.tool(effects="read_only")
async def find(seat: str, row: int = 1) -> str:
    """Look a seat up."""
    return "free"


@registry.tool
async def broken() -> str:
    """Always fails."""
    raise ToolError("no such seat")


@pytest.fixture(autouse=True)
def _pipecat(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    install_pipecat(monkeypatch)
    booked.clear()
    yield


def test_registers_each_tool_with_its_interruption_flag_and_schema():
    llm = LLMService()
    schema = pipecat_tools(llm, registry, executor=Executor(registry))
    assert {name: flag for name, (_, flag) in llm.registered.items()} == {
        "book": False,
        "find": True,
        "broken": True,
    }
    find_schema = next(s for s in schema.standard_tools if s.name == "find")
    assert set(find_schema.properties) == {"seat", "row"}
    assert find_schema.required == ["seat"]


async def test_a_call_returns_through_result_callback_and_speaks_progress():
    llm, sink = LLMService(), RecordingSink()
    pipecat_tools(llm, registry, executor=Executor(registry, sink=sink))
    params = Params({"seat": "4A"}, llm)
    await llm.registered["book"][0](params)
    await settle()
    assert params.results == ["booking #4A"]
    assert llm.pushed == [TTSSpeakFrame("One moment", append_to_context=False)]
    assert [e.call_id for e in sink.events if e.type == "tool.started"] == ["pc_call_1"]


async def test_errors_come_back_as_a_tool_result():
    llm = LLMService()
    pipecat_tools(llm, registry, executor=Executor(registry))
    params = Params({}, llm)
    await llm.registered["broken"][0](params)
    assert params.results[0]["error"]["message"] == "no such seat"


def test_export_tool_gives_schema_and_handler():
    tool = INTEGRATION.export_tool(registry.get("find"))
    assert tool.schema.name == "find" and tool.cancel_on_interruption


async def test_the_processor_drives_a_voice_agent():
    processor = voice_processor()
    model = ScriptedModel([calls(("book", {"seat": "4A"})), text("Booked. Anything else?")])
    agent = VoiceAgent(model, Executor(registry), processor.speaker)
    await processor.process_frame(TranscriptionFrame("ignored before use"), "down")
    await processor.process_frame(UserStartedSpeakingFrame(), "up")
    processor.pushed.clear()
    processor.use(agent)
    await processor.process_frame(InterimTranscriptionFrame("book"), "down")
    await processor.process_frame(TranscriptionFrame("book 4A"), "down")
    await settle()
    await agent.wait()
    assert booked == ["4A"]
    spoken = [frame.text for frame, _ in processor.pushed]
    assert spoken == ["One moment", "Booked.", "Anything else?"]
    assert all(not frame.append_to_context for frame, _ in processor.pushed)
    cuts = processor.interruptions
    started = UserStartedSpeakingFrame()
    await processor.process_frame(started, "up")
    await settle()
    assert processor.interruptions == cuts + 1
    assert processor.pushed[-1] == (started, "up")  # other frames pass through
    await asyncio.wait_for(agent.close(), 1)
