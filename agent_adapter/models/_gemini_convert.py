"""Gemini `generate_content`: request building, response parsing and stream assembly (no SDK)."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderRaw,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models._convert import dump, dumps_arguments, get, items, tool_payload
from agent_adapter.models.usage import usage_gemini
from agent_adapter.schema.transforms import to_gemini_parameters

PROVIDER = "gemini"

_FINISH: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
    "UNEXPECTED_TOOL_CALL": "error",
}
_MODES = {"auto": "AUTO", "none": "NONE", "required": "ANY"}


def build_gemini_request(request: LLMRequest, *, model: str) -> dict[str, Any]:
    """`models.generate_content` kwargs: `{"model", "contents", "config"}` with a dict config.

    Automatic function calling is always off (the adapter runs tools). Assistant messages
    with Gemini `raw` parts are echoed verbatim so thought signatures survive.

    Raises:
        ConfigError: a tool message whose tool name cannot be found.
    """
    config: dict[str, Any] = {"automatic_function_calling": {"disable": True}}
    system = [m.content for m in request.messages if m.role == "system"]
    if system:
        config["system_instruction"] = "\n\n".join(system)
    if request.tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters_json_schema": to_gemini_parameters(tool.parameters),
                    }
                    for tool in request.tools
                ]
            }
        ]
        config["tool_config"] = {"function_calling_config": _calling(request.tool_choice)}
    if request.output_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = to_gemini_parameters(request.output_schema.json_schema)
    _generation(request, config)
    return {"model": model, "contents": _contents(request.messages), "config": config}


def _calling(choice: str) -> dict[str, Any]:
    if choice in _MODES:
        return {"mode": _MODES[choice]}
    return {"mode": "ANY", "allowed_function_names": [choice]}


def _generation(request: LLMRequest, config: dict[str, Any]) -> None:
    for field, key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_output_tokens"),
        ("seed", "seed"),
    ):
        value = getattr(request, field)
        if value is not None:
            config[key] = value
    if request.stop:
        config["stop_sequences"] = list(request.stop)
    if request.reasoning is not None:
        config["thinking_config"] = {"thinking_level": request.reasoning.upper()}


def _contents(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    names: dict[str, str] = {}
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            if message.content:  # a turn needs at least one non-empty part
                contents.append({"role": "user", "parts": [{"text": message.content}]})
        elif message.role == "assistant":
            parts = _model_parts(message, names)
            if parts:
                contents.append({"role": "model", "parts": parts})
        else:
            part = _function_response(message, names)
            last = contents[-1] if contents else None
            if last is not None and last["role"] == "user" and _all_responses(last["parts"]):
                last["parts"].append(part)  # parallel results answer in one turn
            else:
                contents.append({"role": "user", "parts": [part]})
    return contents


def _model_parts(message: Message, names: dict[str, str]) -> list[dict[str, Any]]:
    raw = message.raw
    if raw is not None and raw.provider == PROVIDER and raw.items:
        parts = [dict(item) for item in raw.items]
        for part in parts:
            call = get(part, "function_call", "functionCall")
            if get(call, "id"):
                names[call["id"]] = get(call, "name") or ""
    else:
        parts = [{"text": message.content}] if message.content else []
        for call in message.tool_calls:
            body: dict[str, Any] = {"name": call.name, "args": dict(call.arguments)}
            if call.id:
                body["id"] = call.id
            parts.append({"function_call": body})
    for call in message.tool_calls:
        if call.id:
            names.setdefault(call.id, call.name)
    return parts


def _function_response(message: Message, names: dict[str, str]) -> dict[str, Any]:
    name = message.name or names.get(message.tool_call_id, "")
    if not name:
        raise ConfigError(
            f"Gemini needs the tool name of tool message {message.tool_call_id!r}; set Message.name"
        )
    value, is_error = tool_payload(message.content)
    body: dict[str, Any] = {
        "name": name,
        "response": {"error": value["error"]} if is_error else {"output": value},
    }
    if message.tool_call_id in names:
        body["id"] = message.tool_call_id
    return {"function_response": body}


def _all_responses(parts: list[dict[str, Any]]) -> bool:
    return all("function_response" in part for part in parts)


def _text(value: Any) -> str:
    """An SDK enum or a plain string as its value."""
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def parse_gemini_response(raw: Any) -> LLMResponse:
    """A normalised response from a `GenerateContentResponse` (SDK object or dict)."""
    candidates = items(raw, "candidates")
    candidate = candidates[0] if candidates else None
    parts = items(get(candidate, "content"), "parts")
    texts = [get(p, "text") for p in parts if get(p, "text") and not get(p, "thought")]
    calls = tuple(_call(part) for part in parts if _function_call(part) is not None)
    if candidate is None:
        block = _text(
            get(get(raw, "prompt_feedback", "promptFeedback"), "block_reason", "blockReason")
        )
        finish: FinishReason = "content_filter" if block else "other"
        raw_finish = block
    else:
        raw_finish = _text(get(candidate, "finish_reason", "finishReason"))
        finish = _FINISH.get(raw_finish, "other")
        if finish == "stop" and calls:
            finish = "tool_calls"
    text = "".join(texts)
    raw_parts = ProviderRaw(provider=PROVIDER, items=[dump(part) for part in parts])
    return LLMResponse(
        provider=PROVIDER,
        model=get(raw, "model_version", "modelVersion") or "",
        id=get(raw, "response_id", "responseId") or "",
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        raw_finish_reason=raw_finish,
        usage=usage_gemini(get(raw, "usage_metadata", "usageMetadata")),
        message=Message(role="assistant", content=text, tool_calls=calls, raw=raw_parts),
    )


def _function_call(part: Any) -> Any:
    return get(part, "function_call", "functionCall")


def _call(part: Any) -> ToolCall:
    call = _function_call(part)
    return ToolCall(
        id=get(call, "id") or "",
        name=get(call, "name") or "",
        arguments=dict(get(call, "args") or {}),
    )


class GeminiStreamAssembler:
    """Collects streamed chunks; Gemini sends whole function calls, never argument fragments."""

    def __init__(self) -> None:
        self._parts: list[dict[str, Any]] = []
        self._calls = 0
        self._finish = ""
        self._usage: Any = None
        self._feedback: Any = None
        self._model = ""
        self._id = ""

    def feed(self, chunk: Any) -> list[TextDelta | ToolCallDelta]:
        """Deltas carried by one chunk (each is a partial `GenerateContentResponse`)."""
        self._model = get(chunk, "model_version", "modelVersion") or self._model
        self._id = get(chunk, "response_id", "responseId") or self._id
        self._usage = get(chunk, "usage_metadata", "usageMetadata") or self._usage
        self._feedback = get(chunk, "prompt_feedback", "promptFeedback") or self._feedback
        candidates = items(chunk, "candidates")
        if not candidates:
            return []
        self._finish = _text(get(candidates[0], "finish_reason", "finishReason")) or self._finish
        out: list[TextDelta | ToolCallDelta] = []
        for part in items(get(candidates[0], "content"), "parts"):
            self._parts.append(dump(part))
            if get(part, "text") and not get(part, "thought"):
                out.append(TextDelta(text=get(part, "text")))
            elif _function_call(part) is not None:
                call = _call(part)
                out.append(
                    ToolCallDelta(
                        index=self._calls,
                        id=call.id,
                        name=call.name,
                        arguments_delta=dumps_arguments(call.arguments),
                    )
                )
                self._calls += 1
        return out

    def response(self) -> LLMResponse:
        """The assembled response, parsed like a non-streamed one."""
        candidates = []
        if self._parts or self._finish:
            content = {"role": "model", "parts": self._parts}
            candidates.append({"content": content, "finish_reason": self._finish or None})
        return parse_gemini_response(
            {
                "candidates": candidates,
                "usage_metadata": None if self._usage is None else dump(self._usage),
                "prompt_feedback": None if self._feedback is None else dump(self._feedback),
                "model_version": self._model,
                "response_id": self._id,
            }
        )
