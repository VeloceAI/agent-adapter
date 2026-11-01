"""OpenAI Chat Completions: request building, response parsing and stream assembly (no SDK)."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.messages import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models._convert import dump, dumps_arguments, get, items, loads_arguments
from agent_adapter.models._openai_common import add_tools, schema_format
from agent_adapter.models.usage import usage_openai_chat
from agent_adapter.schema.transforms import to_openai_parameters

PROVIDER = "openai"

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


def build_chat_request(
    request: LLMRequest, *, model: str, strict_tools: bool = True
) -> dict[str, Any]:
    """`chat.completions.create` kwargs for `request` (without `stream`)."""
    kwargs: dict[str, Any] = {"model": model, "messages": [_message(m) for m in request.messages]}
    add_tools(kwargs, request, lambda tool: _tool(tool, strict_tools), _tool_choice)
    if request.output_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": schema_format(request.output_schema),
        }
    _sampling(request, kwargs)
    if request.max_output_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_output_tokens
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.stop:
        kwargs["stop"] = list(request.stop)
    if request.reasoning is not None:
        kwargs["reasoning_effort"] = request.reasoning
    return kwargs


def _sampling(request: LLMRequest, kwargs: dict[str, Any]) -> None:
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p


def _message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    if message.role != "assistant" or not message.tool_calls:
        return {"role": message.role, "content": message.content}
    return {
        "role": "assistant",
        "content": message.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": dumps_arguments(call.arguments)},
            }
            for call in message.tool_calls
        ],
    }


def _tool(tool: ToolDef, strict_tools: bool) -> dict[str, Any]:
    strict = tool.strict and strict_tools
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": to_openai_parameters(tool.parameters, strict=strict),
            "strict": strict,
        },
    }


def _tool_choice(choice: str) -> Any:
    if choice in ("auto", "none", "required"):
        return choice
    return {"type": "function", "function": {"name": choice}}


def parse_chat_completion(raw: Any) -> LLMResponse:
    """A normalised response from a `ChatCompletion` (SDK object or dict); first choice only."""
    model = get(raw, "model") or ""
    choices = items(raw, "choices")
    choice = choices[0] if choices else None
    message = get(choice, "message")
    refusal = get(message, "refusal") or ""
    text = get(message, "content") or refusal
    calls = tuple(
        ToolCall(
            id=get(call, "id") or "",
            name=get(get(call, "function"), "name") or "",
            arguments=loads_arguments(
                get(get(call, "function"), "arguments"), provider=PROVIDER, model=model
            ),
        )
        for call in items(message, "tool_calls")
        if get(call, "type") in (None, "function")
    )
    raw_finish = get(choice, "finish_reason") or ""
    finish: FinishReason = "content_filter" if refusal else _FINISH.get(raw_finish, "other")
    return LLMResponse(
        provider=PROVIDER,
        model=model,
        id=get(raw, "id") or "",
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        raw_finish_reason=raw_finish,
        usage=usage_openai_chat(get(raw, "usage")),
        message=Message(role="assistant", content=text, tool_calls=calls),
    )


class ChatStreamAssembler:
    """Turns `ChatCompletionChunk`s into deltas; argument fragments are keyed by call index."""

    def __init__(self) -> None:
        self._id = ""
        self._model = ""
        self._text: list[str] = []
        self._refusal: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self._finish: str | None = None
        self._usage: Any = None

    def feed(self, chunk: Any) -> list[TextDelta | ToolCallDelta]:
        """Deltas carried by one chunk."""
        self._id = get(chunk, "id") or self._id
        self._model = get(chunk, "model") or self._model
        if get(chunk, "usage") is not None:
            self._usage = get(chunk, "usage")
        out: list[TextDelta | ToolCallDelta] = []
        for choice in items(chunk, "choices"):
            if (get(choice, "index") or 0) != 0:
                continue
            delta = get(choice, "delta")
            content = get(delta, "content")
            if content:
                self._text.append(content)
                out.append(TextDelta(text=content))
            if get(delta, "refusal"):
                self._refusal.append(get(delta, "refusal"))
            out.extend(self._call_delta(part) for part in items(delta, "tool_calls"))
            self._finish = get(choice, "finish_reason") or self._finish
        return out

    def _call_delta(self, part: Any) -> ToolCallDelta:
        index = get(part, "index") or 0
        function = get(part, "function")
        call_id = get(part, "id") or ""
        name = get(function, "name") or ""
        fragment = get(function, "arguments") or ""
        slot = self._calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        slot["id"] = call_id or slot["id"]
        slot["name"] = name or slot["name"]
        slot["arguments"] += fragment
        return ToolCallDelta(index=index, id=call_id, name=name, arguments_delta=fragment)

    def response(self) -> LLMResponse:
        """The assembled response, parsed like a non-streamed one."""
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self._text) or None}
        if self._refusal:
            message["refusal"] = "".join(self._refusal)
        if self._calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for _, slot in sorted(self._calls.items())
            ]
        return parse_chat_completion(
            {
                "id": self._id,
                "model": self._model,
                "choices": [{"index": 0, "finish_reason": self._finish, "message": message}],
                "usage": None if self._usage is None else dump(self._usage),
            }
        )
