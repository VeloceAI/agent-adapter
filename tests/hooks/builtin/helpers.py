"""Invocation builders shared by the built-in hook tests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ToolCall, ToolResult, Usage
from agent_adapter.hooks import LLMInvocation, NodeInvocation, ToolInvocation


def login(user: str, password: str, settings: dict[str, Any] | None = None) -> str:
    """Log a user in."""
    return user


def tool_inv(
    ctx: ToolContext | None = None,
    *,
    arguments: dict[str, Any] | None = None,
    **policy: Any,
) -> ToolInvocation:
    spec = build_spec(login, **policy)
    args = arguments if arguments is not None else {"user": "ana", "password": "hunter2"}
    ctx = ctx or ToolContext(run_id="run_1", call_id="call_1")
    return ToolInvocation(spec, ToolCall(name="login", arguments=args, id="call_1"), ctx)


def llm_inv(ctx: ToolContext | None = None) -> LLMInvocation:
    request = LLMRequest(
        messages=(Message(role="user", content="my secret plan"),), stage="plan", stream=True
    )
    return LLMInvocation(request, "fake", "m1", ctx or ToolContext(run_id="run_1"))


def llm_response(text: str = "the secret answer") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="m1",
        text=text,
        usage=Usage(input_tokens=10, cached_input_tokens=2, output_tokens=5),
        message=Message(role="assistant", content=text),
        latency_ms=12.5,
    )


class State(BaseModel):
    n: int = 0


def node_inv(ctx: ToolContext | None = None) -> NodeInvocation:
    return NodeInvocation("triage", "classify", State(), ctx or ToolContext(run_id="run_1"))


def ok(value: Any = "ana") -> ToolResult:
    return ToolResult("call_1", "login", "ok", value=value, attempts=1, latency_ms=3.0)
