"""Parse and bound a `RunAgentInput`, and turn its messages into adapter `Message`s."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations.ag_ui._support import core

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_048_576
MAX_ID_CHARS = 128
MAX_RESUME = 64
MAX_MESSAGES = 1_000


class AGUIInputInvalid(ToolUsageError):
    """A `RunAgentInput` that is not valid JSON, fails the schema, or exceeds a bound."""

    kind = "ag-ui-input-invalid"


def parse_input(
    body: bytes | str | Mapping[str, Any] | RunAgentInput, *, max_bytes: int = MAX_BODY_BYTES
) -> RunAgentInput:
    """A validated, bounded `RunAgentInput` from a request body or a parsed mapping.

    Raises:
        AGUIInputInvalid: bad JSON, a schema error, or a bound exceeded (body size, id
            length, message or resume count, duplicate interrupt ids).
    """
    model = core().RunAgentInput
    if isinstance(body, model):
        return _bounded(body)
    if isinstance(body, bytes | str):
        size = len(body.encode("utf-8")) if isinstance(body, str) else len(body)
        if size > max_bytes:
            raise AGUIInputInvalid(f"request body is {size} bytes; the limit is {max_bytes}")
        try:
            body = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AGUIInputInvalid("request body is not JSON") from exc
    if not isinstance(body, Mapping):
        raise AGUIInputInvalid("request body must be a JSON object")
    try:
        parsed: RunAgentInput = model.model_validate(dict(body))
    except ValidationError as exc:
        fields = sorted({".".join(map(str, err["loc"])) for err in exc.errors()})
        raise AGUIInputInvalid(f"RunAgentInput is invalid at {fields[:5]}") from exc
    return _bounded(parsed)


def _bounded(run_input: RunAgentInput) -> RunAgentInput:
    for name in ("thread_id", "run_id"):
        _check_id(name, getattr(run_input, name))
    if len(run_input.messages) > MAX_MESSAGES:
        raise AGUIInputInvalid(f"at most {MAX_MESSAGES} messages, got {len(run_input.messages)}")
    resume = run_input.resume or []
    if len(resume) > MAX_RESUME:
        raise AGUIInputInvalid(f"at most {MAX_RESUME} resume entries, got {len(resume)}")
    ids = [entry.interrupt_id for entry in resume]
    for interrupt_id in ids:
        _check_id("resume interruptId", interrupt_id)
    if len(set(ids)) != len(ids):
        raise AGUIInputInvalid("resume answers the same interrupt more than once")
    return run_input


def _check_id(name: str, value: str) -> None:
    if not value or len(value) > MAX_ID_CHARS:
        raise AGUIInputInvalid(f"{name} must be 1 to {MAX_ID_CHARS} characters")


def to_messages(run_input: RunAgentInput) -> list[Message]:
    """The input's conversation as adapter `Message`s; text parts only, other roles skipped.

    Developer messages become system messages. Activity and reasoning messages are not
    sent to a model, so they are dropped.

    Raises:
        AGUIInputInvalid: an assistant tool call whose arguments are not a JSON object.
    """
    c = core()
    messages: list[Message] = []
    for message in run_input.messages:
        if isinstance(message, c.SystemMessage | c.DeveloperMessage):
            messages.append(Message(role="system", content=_text(message.content)))
        elif isinstance(message, c.UserMessage):
            messages.append(Message(role="user", content=_text(message.content)))
        elif isinstance(message, c.AssistantMessage):
            calls = tuple(_call(call) for call in message.tool_calls or ())
            messages.append(
                Message(role="assistant", content=message.content or "", tool_calls=calls)
            )
        elif isinstance(message, c.ToolMessage):
            content = _text(message.content)
            messages.append(
                Message(role="tool", content=content, tool_call_id=message.tool_call_id)
            )
        else:
            logger.debug("ag_ui: %s message %s not sent to the model", message.role, message.id)
    return messages


def _call(call: Any) -> ToolCall:
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise AGUIInputInvalid(f"tool call {call.id!r}: arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise AGUIInputInvalid(f"tool call {call.id!r}: arguments must be a JSON object")
    return ToolCall(name=call.function.name, arguments=arguments, id=call.id)


def _text(content: Any) -> str:
    """Plain text of a string or a list of content parts (non-text parts are left out)."""
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content or ())
