"""Conversation structure for compression: atomic units and the messages no step may touch."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from agent_adapter.core.messages import Message

UnitKind = Literal["system", "user", "assistant", "tool_round", "tool_results"]

# Tool results recording an approval or a policy decision; kept verbatim with their round.
APPROVAL_ERROR_KINDS = frozenset(
    {
        "approval-denied",
        "approval-pending",
        "approval-not-found",
        "approval-already-decided",
        "guardrail-blocked",
        "budget-exceeded",
    }
)


# A native item or block `type` naming an approval: `mcp_approval_request`, `tool_approval`.
_APPROVAL_TYPE = re.compile(r"(?i)(?:^|[_.\-])(?:approval|confirmation)s?(?:[_.\-]|$)")
# Function names frameworks use to ask a human: ADK `adk_request_confirmation`.
_APPROVAL_CALL = re.compile(r"(?i)^(?:adk_)?request_(?:confirmation|approval)$")
_APPROVAL_KEYS = frozenset({"approval_request_id"})  # MCP approval responses
_MAX_DEPTH = 8
_MAX_NODES = 2000  # bounds the walk over one native item


@dataclass(frozen=True)
class Unit:
    """Messages that are kept, changed or removed together; `messages[start:stop]`."""

    start: int
    stop: int
    kind: UnitKind
    protected: bool
    messages: tuple[Message, ...]


def carries_approval_state(message: Message) -> bool:
    """True for a tool result recording an approval or policy decision, or approval raw items."""
    if message.raw is not None and has_approval_marker(message.raw.items):
        return True
    if message.role != "tool" or not message.content.startswith('{"error"'):
        return False
    try:
        body = json.loads(message.content)
    except ValueError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    return isinstance(error, dict) and error.get("kind") in APPROVAL_ERROR_KINDS


def has_approval_marker(native: object) -> bool:
    """True when a native message (any format) holds an approval or confirmation item.

    Looks for an item `type` naming an approval, an MCP `approval_request_id`, or a
    request-confirmation function name, in dicts, lists and plain or pydantic objects.
    """
    budget = [_MAX_NODES]
    return _marked(native, 0, budget, set())


def _marked(value: object, depth: int, budget: list[int], seen: set[int]) -> bool:
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_DEPTH or value is None:
        return False
    if isinstance(value, str | bytes | int | float | bool):
        return False
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, Mapping):
        fields: Mapping[object, object] = value
    elif isinstance(value, list | tuple | set | frozenset):
        return any(_marked(item, depth + 1, budget, seen) for item in value)
    else:
        found = getattr(value, "__dict__", None)
        if not isinstance(found, dict) or isinstance(value, type):
            return False
        fields = found
    if any(key in _APPROVAL_KEYS for key in fields):
        return True
    kind, name = fields.get("type"), fields.get("name")
    if isinstance(kind, str) and _APPROVAL_TYPE.search(kind):
        return True
    if isinstance(name, str) and _APPROVAL_CALL.match(name):
        return True
    return any(_marked(item, depth + 1, budget, seen) for item in fields.values())


def _groups(messages: Sequence[Message]) -> list[tuple[int, int, UnitKind]]:
    """(start, stop, kind) per unit: an assistant turn with calls owns the tool results after it."""
    groups: list[tuple[int, int, UnitKind]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        stop = index + 1
        kind: UnitKind
        if message.role == "assistant" and message.tool_calls:
            kind = "tool_round"
            while stop < len(messages) and messages[stop].role == "tool":
                stop += 1
        elif message.role == "tool":
            kind = "tool_results"  # results without their call (an old or foreign history)
            while stop < len(messages) and messages[stop].role == "tool":
                stop += 1
        elif message.role == "system":
            kind = "system"
        elif message.role == "user":
            kind = "user"
        else:
            kind = "assistant"
        groups.append((index, stop, kind))
        index = stop
    return groups


class Protection:
    """Which messages no step may change or remove, fixed on the input of one compression.

    Protected: system messages, pinned messages, rounds carrying approval state, the latest
    user message and the last unit (the newest turn, including a pending tool round). A unit
    with one protected message is protected whole.
    """

    def __init__(self, messages: Sequence[Message]) -> None:
        groups = _groups(messages)
        flagged: list[tuple[int, int]] = []
        last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=None)
        for number, (start, stop, _) in enumerate(groups):
            members = messages[start:stop]
            if (
                number == len(groups) - 1
                or (last_user is not None and start <= last_user < stop)
                or any(_intrinsic(m) or carries_approval_state(m) for m in members)
            ):
                flagged.append((start, stop))
        # Kept alive with their ids, so an id is never reused while this object lives.
        self._kept = tuple(m for start, stop in flagged for m in messages[start:stop])
        self._ids = frozenset(id(m) for m in self._kept)

    def is_protected(self, message: Message) -> bool:
        """True when no step may change or remove `message`."""
        return _intrinsic(message) or id(message) in self._ids


def _intrinsic(message: Message) -> bool:
    return message.role == "system" or message.pinned


def split_units(messages: Sequence[Message], protection: Protection | None = None) -> list[Unit]:
    """`messages` as units in order; `protected` follows `protection` (default: of `messages`)."""
    guard = protection if protection is not None else Protection(messages)
    units = []
    for start, stop, kind in _groups(messages):
        members = tuple(messages[start:stop])
        protected = any(guard.is_protected(m) for m in members)
        units.append(Unit(start, stop, kind, protected, members))
    return units
