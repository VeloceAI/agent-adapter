"""Anthropic Messages: request building, response parsing and stream assembly (no SDK)."""

from __future__ import annotations

from typing import Any

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
from agent_adapter.models._anthropic_schema import to_anthropic_output_schema
from agent_adapter.models._convert import (
    dump,
    get,
    items,
    loads_arguments,
    tool_payload,
    warn_unsupported,
)
from agent_adapter.models.usage import usage_anthropic
from agent_adapter.schema.transforms import to_anthropic_input_schema

PROVIDER = "anthropic"

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}
# Anthropic's lowest effort is "low".
_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


def build_anthropic_request(request: LLMRequest, *, model: str, max_tokens: int) -> dict[str, Any]:
    """`messages.create` kwargs for `request` (without `stream`).

    System text and message content are sent as blocks, so a cache plug-in can mark them.
    `temperature`, `top_p` and `seed` are dropped: the SDK's `create` has no such parameters.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_output_tokens or max_tokens,
        "messages": _messages(request.messages),
    }
    system = [{"type": "text", "text": m.content} for m in request.messages if m.role == "system"]
    if system:
        kwargs["system"] = system
    if request.tools:
        kwargs["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": to_anthropic_input_schema(tool.parameters),
            }
            for tool in request.tools
        ]
        kwargs["tool_choice"] = _tool_choice(request.tool_choice, request.parallel_tool_calls)
    output: dict[str, Any] = {}
    if request.output_schema is not None:
        schema = to_anthropic_output_schema(request.output_schema.json_schema)
        output["format"] = {"type": "json_schema", "schema": schema}
    if request.reasoning is not None:
        output["effort"] = _EFFORT[request.reasoning]
    if output:
        kwargs["output_config"] = output
    if request.stop:
        kwargs["stop_sequences"] = list(request.stop)
    # anthropic>=2.3 `messages.create` takes no sampling parameters (TypeError if sent).
    for field in ("temperature", "top_p", "seed"):
        if getattr(request, field) is not None:
            warn_unsupported(PROVIDER, field, "anthropic Messages API")
    return kwargs


def _tool_choice(choice: str, parallel: bool | None) -> dict[str, Any]:
    kinds = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}
    out: dict[str, Any] = dict(kinds.get(choice, {"type": "tool", "name": choice}))
    if parallel is False and out["type"] != "none":
        out["disable_parallel_tool_use"] = True
    return out


def _messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            if message.content:  # the API rejects empty text blocks
                out.append({"role": "user", "content": [{"type": "text", "text": message.content}]})
        elif message.role == "assistant":
            blocks = _assistant_blocks(message)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:
            block = _tool_result(message)
            last = out[-1] if out else None
            if last is not None and last["role"] == "user" and _all_results(last["content"]):
                last["content"].append(block)  # parallel results answer in one turn
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    raw = message.raw
    if raw is not None and raw.provider == PROVIDER and raw.items:
        return [dict(item) for item in raw.items]  # keeps thinking blocks and signatures
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
        for call in message.tool_calls
    )
    return blocks


def _tool_result(message: Message) -> dict[str, Any]:
    _, is_error = tool_payload(message.content)
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": message.content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _all_results(blocks: list[dict[str, Any]]) -> bool:
    return all(block.get("type") == "tool_result" for block in blocks)


def parse_anthropic_message(raw: Any) -> LLMResponse:
    """A normalised response from an Anthropic `Message` (SDK object or dict)."""
    blocks = items(raw, "content")
    text = "".join(get(b, "text") or "" for b in blocks if get(b, "type") == "text")
    calls = tuple(
        ToolCall(
            id=get(b, "id") or "", name=get(b, "name") or "", arguments=dict(get(b, "input") or {})
        )
        for b in blocks
        if get(b, "type") == "tool_use"
    )
    stop = get(raw, "stop_reason") or ""
    raw_blocks = ProviderRaw(provider=PROVIDER, items=[dump(block) for block in blocks])
    return LLMResponse(
        provider=PROVIDER,
        model=get(raw, "model") or "",
        id=get(raw, "id") or "",
        text=text,
        tool_calls=calls,
        finish_reason=_FINISH.get(stop, "other"),
        raw_finish_reason=stop,
        usage=usage_anthropic(get(raw, "usage")),
        message=Message(role="assistant", content=text, tool_calls=calls, raw=raw_blocks),
    )


class AnthropicStreamAssembler:
    """Turns raw stream events into deltas; tool input arrives as `input_json_delta` fragments."""

    def __init__(self, *, model: str) -> None:
        self._model = model
        self._id = ""
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json: dict[int, str] = {}
        self._tool_index: dict[int, int] = {}
        self._usage: dict[str, Any] = {}
        self._stop = ""

    def feed(self, event: Any) -> list[TextDelta | ToolCallDelta]:
        """Deltas carried by one event."""
        kind = get(event, "type")
        if kind == "message_start":
            message = get(event, "message")
            self._id = get(message, "id") or self._id
            self._model = get(message, "model") or self._model
            self._update_usage(get(message, "usage"))
        elif kind == "content_block_start":
            return self._start(get(event, "index") or 0, dump(get(event, "content_block")))
        elif kind == "content_block_delta":
            return self._delta(get(event, "index") or 0, get(event, "delta"))
        elif kind == "message_delta":
            self._stop = get(get(event, "delta"), "stop_reason") or self._stop
            self._update_usage(get(event, "usage"))
        return []

    def _start(self, index: int, block: dict[str, Any]) -> list[TextDelta | ToolCallDelta]:
        self._blocks[index] = block
        if block.get("type") == "tool_use":
            self._tool_index[index] = len(self._tool_index)
            self._json[index] = ""
            return [
                ToolCallDelta(
                    index=self._tool_index[index],
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                )
            ]
        if block.get("type") == "text" and block.get("text"):
            return [TextDelta(text=block["text"])]
        return []

    def _delta(self, index: int, delta: Any) -> list[TextDelta | ToolCallDelta]:
        block = self._blocks.setdefault(index, {})
        kind = get(delta, "type")
        if kind == "text_delta":
            block["text"] = block.get("text", "") + (get(delta, "text") or "")
            return [TextDelta(text=get(delta, "text") or "")]
        if kind == "input_json_delta":
            fragment = get(delta, "partial_json") or ""
            self._json[index] = self._json.get(index, "") + fragment
            tool = self._tool_index.setdefault(index, len(self._tool_index))
            return [ToolCallDelta(index=tool, arguments_delta=fragment)]
        if kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + (get(delta, "thinking") or "")
        elif kind == "signature_delta":
            block["signature"] = get(delta, "signature") or ""
        return []

    def _update_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self._usage.update({k: v for k, v in dump(usage).items() if v is not None})

    def response(self) -> LLMResponse:
        """The assembled message, parsed like a non-streamed one."""
        for index, text in self._json.items():
            if text:
                self._blocks[index]["input"] = loads_arguments(
                    text, provider=PROVIDER, model=self._model
                )
        return parse_anthropic_message(
            {
                "id": self._id,
                "model": self._model,
                "content": [block for _, block in sorted(self._blocks.items())],
                "stop_reason": self._stop,
                "usage": self._usage,
            }
        )
