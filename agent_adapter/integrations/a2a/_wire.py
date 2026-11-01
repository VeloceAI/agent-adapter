"""The A2A wire format of a tool call: data-part payloads, bounds and decisions (pure Python)."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.errors import MissingDependency

logger = logging.getLogger(__name__)

TARGET = "a2a"
TOOLS_EXTENSION = "urn:agent-adapter:a2a:tools:v1"  # card extension: per-skill input schemas
JSON_MEDIA = "application/json"
TOOL_KEY = "tool"
ARGUMENTS_KEY = "arguments"
APPROVAL_KIND = "approval-required"
MAX_PARTS = 16  # parts read from one incoming message
MAX_DATA_BYTES = 256 * 1024  # JSON size of one call's data part
MAX_DEPTH = 32  # nesting of the call's arguments
MAX_TEXT_CHARS = 4000  # remote status text kept in a ToolError


class WireError(ValueError):
    """A message that is not a well-formed tool call or decision; the text is client-safe."""


@dataclass(frozen=True)
class Call:
    """One requested tool call."""

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    """A client's approval answer (counts only with `client_approval=True`)."""

    approval_id: str
    approved: bool
    reason: str


def require_a2a() -> None:
    """Raise MissingDependency when the `a2a-sdk` package is not installed."""
    try:
        import a2a  # noqa: F401
    except ImportError as exc:
        logger.debug("a2a is not importable: %s", exc)
        raise MissingDependency(
            'the A2A integration needs a2a-sdk: pip install "agent-adapter[a2a]"'
        ) from exc


def whole_numbers(value: Any) -> Any:
    """`value` with whole floats as ints (protobuf `Value` stores every number as a double)."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {key: whole_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [whole_numbers(item) for item in value]
    return value


def check_bounds(value: Any, *, limit: int = MAX_DATA_BYTES, depth: int = MAX_DEPTH) -> None:
    """Refuse a payload over `limit` JSON bytes or nested deeper than `depth`.

    Raises:
        WireError: too large, too deep or not JSON.
    """
    if _depth(value, depth) > depth:
        raise WireError(f"the call is nested deeper than {depth} levels")
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise WireError("the call is not plain JSON") from exc
    if size > limit:
        raise WireError(f"the call is {size} bytes; the limit is {limit}")


def _depth(value: Any, cap: int, level: int = 0) -> int:
    if level > cap:
        return level
    if isinstance(value, Mapping):
        return max((_depth(v, cap, level + 1) for v in value.values()), default=level + 1)
    if isinstance(value, list):
        return max((_depth(v, cap, level + 1) for v in value), default=level + 1)
    return level


def parse_call(data_parts: Sequence[Any], *, limit: int = MAX_DATA_BYTES) -> Call:
    """The call in the first data part `{"tool": str, "arguments": object}`.

    Raises:
        WireError: no such part, a bad shape, or a payload over the bounds.
    """
    payload = _first_object(data_parts, TOOL_KEY)
    if payload is None:
        raise WireError('send a data part {"tool": "<name>", "arguments": {...}}')
    check_bounds(payload, limit=limit)
    tool, arguments = payload.get(TOOL_KEY), payload.get(ARGUMENTS_KEY, {})
    if not isinstance(tool, str) or not tool:
        raise WireError('"tool" must be a non-empty string')
    if not isinstance(arguments, Mapping):
        raise WireError('"arguments" must be an object')
    return Call(tool, whole_numbers(dict(arguments)))


def parse_decision(data_parts: Sequence[Any]) -> Decision | None:
    """The first `{"approval_id": str, "approved": bool}` data part; None when absent or bad."""
    payload = _first_object(data_parts, "approval_id")
    if payload is None:
        return None
    approval_id, approved = payload.get("approval_id"), payload.get("approved")
    reason = payload.get("reason")
    if not isinstance(approval_id, str) or not approval_id or not isinstance(approved, bool):
        return None
    return Decision(approval_id, approved, reason[:500] if isinstance(reason, str) else "")


def _first_object(data_parts: Sequence[Any], key: str) -> Mapping[str, Any] | None:
    for item in list(data_parts)[:MAX_PARTS]:
        if isinstance(item, Mapping) and key in item:
            return item
    return None


def approval_payload(approval_id: str, tool: str, *, client: bool) -> dict[str, Any]:
    """The data part of an `input-required` status: what is pending and how to answer."""
    answer = (
        {"approval_id": approval_id, "approved": "<true|false>", "reason": "<optional>"}
        if client
        else "decide it through the operator's approval API, then send any message"
    )
    return {"kind": APPROVAL_KIND, "approval_id": approval_id, "tool": tool, "answer": answer}


def clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """`text` cut to `limit` characters."""
    return text if len(text) <= limit else text[:limit] + " ..."
