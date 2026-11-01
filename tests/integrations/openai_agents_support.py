"""Offline Agents SDK helpers: a scripted `Model` and a runner that returns one call's output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agents import Agent, Model, ModelResponse, RunConfig, Runner, ToolCallOutputItem, Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

CONFIG = RunConfig(tracing_disabled=True)

Call = tuple[str, dict[str, Any] | str, str]  # (tool name, arguments, call id)


class CallThenAnswer(Model):
    """Turn 1: the scripted tool calls; once tool outputs arrive: a final text answer."""

    def __init__(self, calls: Sequence[Call], *, answer: str = "done") -> None:
        self.calls = list(calls)
        self.answer = answer
        self.inputs: list[Any] = []

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        model_input = args[1] if len(args) > 1 else kwargs["input"]
        self.inputs.append(model_input)
        if _has_tool_output(model_input):
            return ModelResponse(output=[_text(self.answer)], usage=Usage(), response_id=None)
        output = [_call(name, arguments, call_id) for name, arguments, call_id in self.calls]
        return ModelResponse(output=list(output), usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the scripted model does not stream")


def _has_tool_output(model_input: Any) -> bool:
    return isinstance(model_input, list) and any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in model_input
    )


def _call(name: str, arguments: dict[str, Any] | str, call_id: str) -> ResponseFunctionToolCall:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ResponseFunctionToolCall(
        type="function_call", call_id=call_id, name=name, arguments=text, id=f"fc_{call_id}"
    )


def _text(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def agent_with(tools: Sequence[Any], calls: Sequence[Call]) -> Agent[Any]:
    """An agent whose model calls `calls` once, then answers."""
    return Agent(
        name="kit", instructions="use the tools", tools=list(tools), model=CallThenAnswer(calls)
    )


async def run(agent: Agent[Any], start: Any = "go") -> Any:
    """`Runner.run` offline (tracing off)."""
    return await Runner.run(agent, start, run_config=CONFIG)


def output_of(result: Any, call_id: str) -> Any:
    """The output the run recorded for `call_id`; AssertionError when there is none."""
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem) and item.call_id == call_id:
            return item.output
    raise AssertionError(f"no output for {call_id}: {[type(i).__name__ for i in result.new_items]}")


def decoded(raw: Any) -> Any:
    """A tool output string as a value: JSON when it parses, else the string itself."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def is_error_output(raw: Any) -> bool:
    value = decoded(raw)
    return isinstance(value, dict) and set(value) == {"error"}
