"""The Pipecat connector against the real `pipecat-ai` classes (no pipeline, no network)."""

import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService

from agent_adapter import Registry
from agent_adapter.exec import Executor, progress
from agent_adapter.integrations.pipecat import pipecat_tools, voice_processor

pytestmark = pytest.mark.contract

registry = Registry()


@registry.tool(effects="additive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment")
    return f"booking #{seat}"


@registry.tool(effects="read_only")
async def find(seat: str) -> str:
    """Look a seat up."""
    return "free"


class Service(LLMService):
    def __init__(self) -> None:
        super().__init__()
        self.pushed: list[object] = []

    async def push_frame(self, frame: object, direction: object = None) -> None:
        self.pushed.append(frame)


def test_tools_register_with_the_right_interruption_flags():
    llm = Service()
    schema = pipecat_tools(llm, registry, executor=Executor(registry))
    assert [s.name for s in schema.standard_tools] == ["book", "find"]
    flags = {name: item.cancel_on_interruption for name, item in llm._functions.items()}
    assert flags == {"book": False, "find": True}


async def test_a_call_returns_through_result_callback_and_progress_is_spoken():
    llm, results = Service(), []

    async def done(result: object, *, properties: object = None) -> None:
        results.append(result)

    pipecat_tools(llm, registry, executor=Executor(registry))
    params = FunctionCallParams(
        function_name="book",
        tool_call_id="t1",
        arguments={"seat": "4A"},
        llm=llm,
        pipeline_worker=None,
        context=None,
        result_callback=done,
    )
    await llm._functions["book"].handler(params)
    await asyncio.sleep(0.01)
    assert results == ["booking #4A"]
    assert [(type(f), f.text, f.append_to_context) for f in llm.pushed] == [
        (TTSSpeakFrame, "One moment", False)
    ]


def test_the_voice_processor_is_a_real_frame_processor():
    processor = voice_processor()
    assert isinstance(processor, FrameProcessor) and hasattr(processor, "speaker")
