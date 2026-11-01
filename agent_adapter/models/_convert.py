"""SDK-free helpers shared by the model ports: field access, JSON, structured output."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMResponse, Message, OutputSchema

logger = logging.getLogger(__name__)

_warned: set[tuple[str, str]] = set()


def get(obj: Any, *names: str) -> Any:
    """The first non-None field among `names` of a dict or an object; None when absent."""
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def count(obj: Any, *names: str) -> int:
    """A token count field as a non-negative int; missing or invalid counts as 0."""
    value = get(obj, *names)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def dump(obj: Any) -> dict[str, Any]:
    """A JSON-safe dict of an SDK model (None fields dropped) or a copy of a mapping."""
    if isinstance(obj, Mapping):
        return dict(obj)
    dumped: dict[str, Any] = obj.model_dump(mode="json", exclude_none=True)
    return dumped


def items(obj: Any, *names: str) -> list[Any]:
    """A list field (e.g. `choices`, `parts`); missing counts as empty."""
    value = get(obj, *names)
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []


def loads_arguments(text: str | None, *, provider: str, model: str) -> dict[str, Any]:
    """Tool-call arguments from a JSON string ("" means no arguments).

    Raises:
        ModelOutputInvalid: the text is not a JSON object.
    """
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        logger.warning("%s %s returned tool arguments that are not JSON", provider, model)
        raise ModelOutputInvalid(
            "tool-call arguments are not valid JSON", provider=provider, model=model
        ) from exc
    if not isinstance(value, dict):
        raise ModelOutputInvalid(
            "tool-call arguments are not a JSON object", provider=provider, model=model
        )
    return value


def dumps_arguments(arguments: Mapping[str, Any]) -> str:
    """Tool-call arguments as the compact JSON string providers expect."""
    return json.dumps(dict(arguments), separators=(",", ":"), ensure_ascii=False)


def with_parsed(
    response: LLMResponse, schema: OutputSchema | None, *, sdk_parsed: Any = None
) -> LLMResponse:
    """Fill `parsed` for a finished structured-output reply.

    Only a `stop` reply without tool calls is parsed; truncated or filtered replies keep
    `parsed=None` so the caller sees their finish reason instead.

    Raises:
        ModelOutputInvalid: the reply text is not valid JSON.
    """
    if schema is None or response.finish_reason != "stop" or response.tool_calls:
        return response
    if sdk_parsed is not None:
        return response.model_copy(update={"parsed": sdk_parsed})
    try:
        value = json.loads(response.text)
    except ValueError as exc:
        logger.warning(
            "%s %s structured output is not valid JSON (response %s)",
            response.provider,
            response.model,
            response.id,
        )
        raise ModelOutputInvalid(
            f"structured output for {schema.name!r} is not valid JSON",
            provider=response.provider,
            model=response.model,
            request_id=response.id,
        ) from exc
    return response.model_copy(update={"parsed": value})


def tool_payload(content: str) -> tuple[Any, bool]:
    """A tool message's content as JSON (else the text) and whether it reports an error.

    `run_tool_loop` sends failures as `{"error": {...}}`; that shape marks an error.
    """
    try:
        value = json.loads(content)
    except ValueError:
        return content, False
    return value, isinstance(value, dict) and set(value) == {"error"}


def tool_names(messages: Iterable[Message]) -> dict[str, str]:
    """Tool name per tool-call id, from the assistant messages of a history."""
    return {
        call.id: call.name
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
        if call.id
    }


def warn_unsupported(provider: str, field: str, detail: str) -> None:
    """Log once per process that a request field is not sent to this provider API."""
    if (provider, field) in _warned:
        return
    _warned.add((provider, field))
    logger.warning("%s: request field %r is not supported (%s); ignored", provider, field, detail)
