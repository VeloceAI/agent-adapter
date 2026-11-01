"""Tool-loop history helpers: a stable digest, the trailing tool turn, unanswered calls."""

from __future__ import annotations

from collections.abc import Sequence

from agent_adapter.cache.keys import digest
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall


def _history_digest(messages: Sequence[Message]) -> str:
    """Digest of the fields every history keeps (no `raw`, which hosts may drop)."""
    stable = [
        [
            m.role,
            m.content,
            m.tool_call_id,
            [[c.name, c.arguments, c.id] for c in m.tool_calls],
        ]
        for m in messages
    ]
    try:
        return digest(stable)[:12]
    except TypeError:  # arguments that JSON cannot hold
        return digest(repr(stable))[:12]


def _trailing_turn(messages: Sequence[Message]) -> tuple[int, tuple[ToolCall, ...]] | None:
    """The last assistant turn with tool calls, when only tool messages follow it."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "tool":
            continue
        if message.role == "assistant" and message.tool_calls:
            return index, message.tool_calls
        return None
    return None


def _unanswered(
    calls: Sequence[tuple[ToolCall, bool]], answered: Sequence[Message]
) -> list[ToolCall]:
    """Calls without a tool message, matched by id (given or synthetic).

    Tool messages without an id (older histories) answer id-less calls in order.
    """
    ids = {m.tool_call_id for m in answered if m.tool_call_id}
    anonymous = sum(1 for m in answered if not m.tool_call_id)
    todo: list[ToolCall] = []
    for call, given in calls:
        if call.id in ids:
            continue
        if not given and anonymous:
            anonymous -= 1
            continue
        todo.append(call)
    return todo
