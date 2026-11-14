"""Pipecat: registry tools registered on an LLM service, and a VoiceAgent as a frame processor.

A tool that changes things (`effects=`) registers with `cancel_on_interruption=False`, so an
interruption never stops it half-way; read-only tools are cancelled as Pipecat does by
default. `progress()` lines go out as `TTSSpeakFrame`s that stay out of the LLM context.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec import changes_things, listen_progress
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._voice_bridge import EventPump, spawn
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

if TYPE_CHECKING:
    from agent_adapter.voice import VoiceAgent

TARGET = "pipecat"


@dataclass(frozen=True)
class PipecatTool:
    """A schema for `ToolsSchema` and the handler to register on the LLM service."""

    schema: Any  # FunctionSchema
    handler: Callable[[Any], Any]
    cancel_on_interruption: bool

    def register(self, llm: Any) -> None:
        """`llm.register_function` with this tool's name, handler and interruption flag."""
        llm.register_function(
            self.schema.name, self.handler, cancel_on_interruption=self.cancel_on_interruption
        )


def pipecat_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> PipecatTool:
    """The schema and handler for `spec`; errors reach the model as `{"error": {...}}`.

    Raises:
        MissingDependency: `pipecat-ai` is not installed.
        ToolUsageError: `spec.targets` excludes `pipecat`.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema

    runner = lazy_runner(executor)

    async def handler(params: Any) -> None:
        with listen_progress(lambda text: _say(params, text)):
            result = await INTEGRATION._run_bound(params, spec, params.arguments, runner)
        await params.result_callback(result)

    parameters = spec.json_schema()
    schema = FunctionSchema(
        name=spec.name,
        description=spec.description,
        properties=parameters.get("properties", {}),
        required=list(parameters.get("required", [])),
    )
    return PipecatTool(schema, handler, cancel_on_interruption=not changes_things(spec))


def pipecat_tools(llm: Any, registry: Registry, *, executor: ToolRunner | None = None) -> Any:
    """Register every registry tool exported to `pipecat` on `llm`; returns their `ToolsSchema`."""
    INTEGRATION.require()
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema

    tools = [pipecat_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]
    for tool in tools:
        tool.register(llm)
    return ToolsSchema(standard_tools=[tool.schema for tool in tools])


def _say(params: Any, text: str) -> None:
    from pipecat.frames.frames import TTSSpeakFrame

    spawn(params.llm.push_frame(TTSSpeakFrame(text, append_to_context=False)))


def voice_processor() -> Any:
    """A frame processor that puts a `VoiceAgent` in the LLM's place in a pipeline.

    Use it as `stt -> processor -> tts`; give the agent `processor.speaker`, then call
    `processor.use(agent)`. Transcripts stop here; the agent's lines go on as speech.
    Pipecat does not report playout per line, so the history keeps lines sent to TTS.
    """
    INTEGRATION.require()
    return _processor_class()()


class _FrameSpeaker:
    """A `Speaker` that pushes `TTSSpeakFrame`s downstream and cuts with an interruption."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    async def say(self, text: str) -> None:
        from pipecat.frames.frames import TTSSpeakFrame

        await self._processor.push_frame(TTSSpeakFrame(text, append_to_context=False))

    async def stop(self) -> None:
        await self._processor.broadcast_interruption()


@functools.cache
def _processor_class() -> type[Any]:
    from pipecat.frames.frames import (
        InterimTranscriptionFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameProcessor

    class VoiceAgentProcessor(FrameProcessor):  # type: ignore[misc, unused-ignore]  # Any without pipecat
        """Caller frames in, agent speech out."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.speaker = _FrameSpeaker(self)
            self._pump: EventPump | None = None

        def use(self, agent: VoiceAgent) -> None:
            self._pump = EventPump(agent)

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                if self._pump is not None:
                    self._pump.said(frame.text)
                return  # the caller's words are not for the TTS
            if isinstance(frame, InterimTranscriptionFrame):
                return
            if isinstance(frame, UserStartedSpeakingFrame) and self._pump is not None:
                self._pump.started_speaking()
            await self.push_frame(frame, direction)

    return VoiceAgentProcessor


class PipecatIntegration(IntegrationBase):
    """Exports `PipecatTool`s (schema plus handler); the call id is `tool_call_id`."""

    name = TARGET
    requires = ("pipecat",)
    extra = "pipecat"
    call_id_fields = ("tool_call_id",)

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> PipecatTool:
        """`pipecat_tool(spec, executor=executor)`."""
        return pipecat_tool(spec, executor=executor)


INTEGRATION = PipecatIntegration()


__all__ = [
    "INTEGRATION",
    "PipecatIntegration",
    "PipecatTool",
    "pipecat_tool",
    "pipecat_tools",
    "voice_processor",
]
