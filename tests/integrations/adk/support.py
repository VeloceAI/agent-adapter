"""Offline ADK helpers for the tests: a scripted `BaseLlm` and one-agent runs."""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec

_apps = itertools.count()


class ScriptedLlm(BaseLlm):
    """Answers each request with the next scripted response (the last one repeats)."""

    model: str = "scripted"
    script: list[LlmResponse] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)
    turn: int = 0

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["scripted"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        reply = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield reply


def function_call(name: str, args: dict[str, Any], call_id: str) -> LlmResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id))
    return LlmResponse(content=types.Content(role="model", parts=[part]))


def calls(*items: tuple[str, dict[str, Any], str]) -> LlmResponse:
    parts = [
        types.Part(function_call=types.FunctionCall(name=n, args=a, id=i)) for n, a, i in items
    ]
    return LlmResponse(content=types.Content(role="model", parts=parts))


def text(value: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


def user(value: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=value)])


class AgentRun:
    """One `LlmAgent` with the given tools and script, run turn by turn in one session."""

    def __init__(self, tools: Sequence[Any], *script: LlmResponse) -> None:
        self.llm = ScriptedLlm(script=list(script))
        self.app = f"adk_test_{next(_apps)}"
        agent = LlmAgent(name="helper", model=self.llm, tools=list(tools))
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name=self.app, agent=agent, session_service=self.sessions)
        self.session_id = ""

    async def turn(self, message: types.Content) -> list[Any]:
        if not self.session_id:
            session = await self.sessions.create_session(app_name=self.app, user_id="u")
            self.session_id = session.id
        run = self.runner.run_async(user_id="u", session_id=self.session_id, new_message=message)
        return [event async for event in run]


def responses(events: Sequence[Any], call_id: str) -> list[dict[str, Any]]:
    """The function responses for `call_id`, in order."""
    return [
        dict(r.response or {})
        for event in events
        for r in event.get_function_responses()
        if r.id == call_id
    ]


def confirmations(events: Sequence[Any]) -> list[Any]:
    """The `adk_request_confirmation` function calls in `events`."""
    return [
        call
        for event in events
        for call in event.get_function_calls()
        if call.name == "adk_request_confirmation"
    ]


def registry_of(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


async def run_tool(
    tool: Any,
    args: dict[str, Any],
    call_id: str,
    *,
    confirmation: Any = None,
    invocation_id: str = "inv_test",
) -> Any:
    """Call `tool` as ADK's tool step does: `run_async` with a `ToolContext` of a real session."""
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools import ToolContext

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="adk_tool", user_id="u")
    invocation = InvocationContext(
        session_service=sessions, invocation_id=invocation_id, session=session
    )
    context = ToolContext(invocation, function_call_id=call_id, tool_confirmation=confirmation)
    return await tool.run_async(args=args, tool_context=context)
