"""Offline Pydantic AI helpers: a `FunctionModel` that calls tools once, then answers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

Call = tuple[str, dict[str, Any] | str, str]  # (tool name, arguments, call id)
RESULT_KINDS = ("tool-return", "retry-prompt")


def call_then_answer(calls: Sequence[Call], *, answer: str = "done") -> FunctionModel:
    """Turn 1: the scripted calls; once results come back: a text answer."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(part.part_kind in RESULT_KINDS for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart(answer)])
        return ModelResponse(
            parts=[ToolCallPart(name, args, tool_call_id=call_id) for name, args, call_id in calls]
        )

    return FunctionModel(respond)


def agent_with(tools: Sequence[Any], calls: Sequence[Call]) -> Agent[None, Any]:
    """An agent that may stop with `DeferredToolRequests` for approval."""
    return Agent(
        call_then_answer(calls), tools=list(tools), output_type=[str, DeferredToolRequests]
    )


def result_part(result: Any, call_id: str) -> Any:
    """The tool-return or retry-prompt part answering `call_id`."""
    for message in result.all_messages():
        for part in message.parts:
            if part.part_kind in RESULT_KINDS and part.tool_call_id == call_id:
                return part
    raise AssertionError(f"no result for {call_id}")


def value_of(part: Any) -> Any:
    content = part.content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def is_error_part(part: Any) -> bool:
    if getattr(part, "part_kind", "") == "retry-prompt":
        return True
    if getattr(part, "outcome", "success") != "success":  # retried, denied, failed
        return True
    value = value_of(part) if hasattr(part, "content") else None
    return isinstance(value, dict) and set(value) == {"error"}
