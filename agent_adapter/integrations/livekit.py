"""LiveKit Agents: registry tools as raw-schema function tools, and a VoiceAgent on a session.

A tool that changes things (`effects=`) calls `disallow_interruptions()` before it runs, so
LiveKit keeps its result in the conversation when the caller talks over the agent; if the
caller has already cut in, it is not started. `progress()` lines are spoken with
`session.say` and never added to the chat context.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec import changes_things, listen_progress
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._voice_bridge import EventPump
from agent_adapter.integrations.base import IntegrationBase, ToolRunner, model_visible_error

if TYPE_CHECKING:
    from agent_adapter.voice import VoiceAgent

TARGET = "livekit"


def livekit_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """A LiveKit `RawFunctionTool` for `spec`, run through the executor.

    Tool errors, denials, guardrails and run limits reach the model as LiveKit `ToolError`s;
    other errors propagate.

    Raises:
        MissingDependency: `livekit-agents` is not installed.
        ToolUsageError: `spec.targets` excludes `livekit`.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    from livekit.agents import function_tool

    runner = lazy_runner(executor)

    async def handler(raw_arguments: dict[str, object], context: Any) -> Any:
        return await _call(spec, raw_arguments, context, runner)

    schema = {"name": spec.name, "description": spec.description, "parameters": spec.json_schema()}
    return function_tool(handler, raw_schema=schema)


def livekit_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """`livekit_tool` for every registry tool exported to `livekit`."""
    return [livekit_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]


async def _call(
    spec: ToolSpec, arguments: Mapping[str, Any], context: Any, runner: ToolRunner
) -> Any:
    from livekit.agents.llm import ToolError

    if changes_things(spec):
        try:
            context.disallow_interruptions()
        except RuntimeError:  # the speech is already interrupted: never start an effect now
            raise ToolError(f"{spec.name} was not run: the caller interrupted") from None
    with listen_progress(lambda text: _say(context, text)):
        return await INTEGRATION._run_bound(
            context, spec, arguments, runner, visible=_raise_tool_error
        )


def _say(context: Any, text: str) -> None:
    context.session.say(text, add_to_chat_ctx=False)


def _raise_tool_error(exc: BaseException) -> NoReturn:
    from livekit.agents.llm import ToolError

    error = model_visible_error(exc)["error"]
    raise ToolError(f"{error['kind']}: {error['message']}") from None


class LiveKitSpeaker:
    """A `Speaker` on an `AgentSession`: only the `VoiceAgent` decides when speech is cut."""

    def __init__(self, session: Any) -> None:
        self.session = session

    async def say(self, text: str) -> None:
        handle = self.session.say(text, allow_interruptions=False, add_to_chat_ctx=False)
        await handle.wait_for_playout()

    async def stop(self) -> None:
        await self.session.interrupt(force=True)


def attach_voice_agent(session: Any, agent: VoiceAgent) -> EventPump:
    """Drive `agent` from a LiveKit `AgentSession` started without an `llm`.

    The session hears (VAD, STT) and speaks (TTS, through `LiveKitSpeaker`); the agent
    decides. Speaking starts a cut; each final transcript is a turn or an approval answer.
    """
    INTEGRATION.require()
    pump = EventPump(agent)

    def on_state(event: Any) -> None:
        if event.new_state == "speaking":
            pump.started_speaking()

    def on_transcript(event: Any) -> None:
        if event.is_final:
            pump.said(event.transcript)

    session.on("user_state_changed", on_state)
    session.on("user_input_transcribed", on_transcript)
    return pump


class LiveKitIntegration(IntegrationBase):
    """Exports raw-schema function tools; the call id comes from `RunContext.function_call`."""

    name = TARGET
    requires = ("livekit.agents",)
    extra = "livekit"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """`livekit_tool(spec, executor=executor)`."""
        return livekit_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        call_id = getattr(getattr(native, "function_call", None), "call_id", "")
        return ToolContext(
            call_id=call_id if isinstance(call_id, str) else "", extras={self.name: native}
        )


INTEGRATION = LiveKitIntegration()


__all__ = [
    "INTEGRATION",
    "LiveKitIntegration",
    "LiveKitSpeaker",
    "attach_voice_agent",
    "livekit_tool",
    "livekit_tools",
]
