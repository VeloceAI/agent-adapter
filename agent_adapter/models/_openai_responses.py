"""OpenAI Responses API: request building, response parsing and stream assembly (no SDK)."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ModelError, ModelUnavailable
from agent_adapter.core.messages import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderRaw,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models._convert import (
    dump,
    dumps_arguments,
    get,
    items,
    loads_arguments,
    warn_unsupported,
)
from agent_adapter.models._openai_common import add_tools, schema_format
from agent_adapter.models.errors import class_for_error_type
from agent_adapter.models.usage import usage_openai_responses
from agent_adapter.schema.transforms import to_openai_parameters

PROVIDER = "openai"
API = "responses"

_INCOMPLETE: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}
_TERMINAL = ("response.completed", "response.incomplete", "response.failed")


def build_responses_request(
    request: LLMRequest, *, model: str, strict_tools: bool = True
) -> dict[str, Any]:
    """`responses.create` kwargs for `request` (without `stream`).

    Assistant messages that carry Responses `raw` items (reasoning, function calls) are
    echoed back verbatim; `seed` and `stop` have no Responses equivalent and are dropped.
    """
    kwargs: dict[str, Any] = {"model": model, "input": _input(request.messages)}
    add_tools(kwargs, request, lambda tool: _tool(tool, strict_tools), _tool_choice)
    if request.output_schema is not None:
        kwargs["text"] = {"format": {"type": "json_schema", **schema_format(request.output_schema)}}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.reasoning is not None:
        kwargs["reasoning"] = {"effort": request.reasoning}
    if request.seed is not None:
        warn_unsupported(PROVIDER, "seed", "Responses API")
    if request.stop:
        warn_unsupported(PROVIDER, "stop", "Responses API")
    return kwargs


def _input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        elif message.role != "assistant":
            out.append({"role": message.role, "content": message.content})
        elif (raw := _own_raw(message)) is not None:
            out.extend(dict(item) for item in raw.items)
        else:
            if message.content:
                out.append({"role": "assistant", "content": message.content})
            out.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": dumps_arguments(call.arguments),
                }
                for call in message.tool_calls
            )
    return out


def _own_raw(message: Message) -> ProviderRaw | None:
    """The message's Responses items when this API produced them."""
    raw = message.raw
    if raw is not None and raw.provider == PROVIDER and raw.api == API and raw.items:
        return raw
    return None


def _tool(tool: ToolDef, strict_tools: bool) -> dict[str, Any]:
    strict = tool.strict and strict_tools
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": to_openai_parameters(tool.parameters, strict=strict),
        "strict": strict,
    }


def _tool_choice(choice: str) -> Any:
    if choice in ("auto", "none", "required"):
        return choice
    return {"type": "function", "name": choice}


def parse_response(raw: Any) -> LLMResponse:
    """A normalised response from a Responses `Response` (SDK object or dict)."""
    model = get(raw, "model") or ""
    output = items(raw, "output")
    texts: list[str] = []
    calls: list[ToolCall] = []
    refused = False
    for item in output:
        kind = get(item, "type")
        if kind == "message":
            for part in items(item, "content"):
                if get(part, "type") == "output_text":
                    texts.append(get(part, "text") or "")
                elif get(part, "type") == "refusal":
                    texts.append(get(part, "refusal") or "")
                    refused = True
        elif kind == "function_call":
            arguments = loads_arguments(get(item, "arguments"), provider=PROVIDER, model=model)
            calls.append(
                ToolCall(
                    id=get(item, "call_id") or "", name=get(item, "name") or "", arguments=arguments
                )
            )
    finish, raw_finish = _finish(raw, has_calls=bool(calls), refused=refused)
    text = "".join(texts)
    raw_items = ProviderRaw(provider=PROVIDER, api=API, items=[dump(item) for item in output])
    return LLMResponse(
        provider=PROVIDER,
        model=model,
        id=get(raw, "id") or "",
        text=text,
        tool_calls=tuple(calls),
        finish_reason=finish,
        raw_finish_reason=raw_finish,
        usage=usage_openai_responses(get(raw, "usage")),
        message=Message(role="assistant", content=text, tool_calls=tuple(calls), raw=raw_items),
    )


def _finish(raw: Any, *, has_calls: bool, refused: bool) -> tuple[FinishReason, str]:
    status = get(raw, "status") or ""
    reason = get(get(raw, "incomplete_details"), "reason") or ""
    raw_finish = reason if status == "incomplete" and reason else status
    if refused:
        return "content_filter", raw_finish
    if status == "completed":
        return ("tool_calls" if has_calls else "stop"), raw_finish
    if status == "incomplete":
        return _INCOMPLETE.get(reason, "other"), raw_finish
    if status == "failed":
        return "error", raw_finish
    return "other", raw_finish


class ResponsesStreamAssembler:
    """Turns Responses stream events into deltas; argument fragments are keyed by `item_id`."""

    def __init__(self, *, model: str) -> None:
        self._model = model
        self._indexes: dict[str, int] = {}
        self._final: Any = None

    def feed(self, event: Any) -> list[TextDelta | ToolCallDelta]:
        """Deltas carried by one event.

        Raises:
            ModelError: the stream reported an `error` event.
        """
        kind = get(event, "type")
        if kind == "response.output_text.delta" and get(event, "delta"):
            return [TextDelta(text=get(event, "delta"))]
        if kind == "response.output_item.added":
            item = get(event, "item")
            if get(item, "type") == "function_call":
                return [
                    ToolCallDelta(
                        index=self._index(get(item, "id") or ""),
                        id=get(item, "call_id") or "",
                        name=get(item, "name") or "",
                        arguments_delta=get(item, "arguments") or "",
                    )
                ]
        elif kind == "response.function_call_arguments.delta":
            index = self._index(get(event, "item_id") or "")
            return [ToolCallDelta(index=index, arguments_delta=get(event, "delta") or "")]
        elif kind in _TERMINAL:
            self._final = get(event, "response")
        elif kind == "error":
            raise self._error(event)
        return []

    def _index(self, item_id: str) -> int:
        return self._indexes.setdefault(item_id, len(self._indexes))

    def _error(self, event: Any) -> ModelError:
        code = str(get(event, "code") or "")
        message = get(event, "message") or "stream error"
        return class_for_error_type(code)(
            f"openai {self._model}: {code or 'error'}: {message}",
            provider=PROVIDER,
            model=self._model,
        )

    def response(self) -> LLMResponse:
        """The final response from the terminal event.

        Raises:
            ModelUnavailable: the stream ended before a terminal event.
        """
        if self._final is None:
            raise ModelUnavailable(
                f"openai {self._model}: stream ended before the response finished",
                provider=PROVIDER,
                model=self._model,
            )
        return parse_response(self._final)
