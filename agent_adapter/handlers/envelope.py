"""The envelope a queue message or serverless event carries: one tool call, or one resume."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime.ids import new_id, scoped_id

MAX_ENVELOPE_BYTES = 1 << 20  # larger payloads are refused before parsing
MAX_KEY_CHARS = 256
CALL_KEYS = frozenset({"tool", "arguments", "run_id", "call_id"})
RESUME_KEYS = frozenset({"approval_id"})


@dataclass(frozen=True)
class Envelope:
    """A parsed envelope: a call (`tool` set) or a resume (`approval_id` set)."""

    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    call_id: str = ""
    approval_id: str = ""

    @property
    def is_resume(self) -> bool:
        """True when this envelope resumes a decided approval."""
        return bool(self.approval_id)


def parse_envelope(
    payload: Mapping[str, Any] | str | bytes, *, message_id: str = "", source: str = ""
) -> Envelope:
    """Validate `payload`; a call without ids gets ids derived from `message_id`, else fresh.

    A redelivered message keeps its `message_id`, so derived ids name the same call again.
    Decisions never travel in an envelope: only an `ApprovalStore` holds them.

    Raises:
        ToolUsageError: not a JSON object, too large, unknown or missing keys, bad values.
    """
    data = _as_mapping(payload)
    keys = set(data)
    if "approval_id" in keys:
        _only(keys, RESUME_KEYS, "a resume envelope")
        return Envelope(approval_id=_key(data, "approval_id"))
    _only(keys, CALL_KEYS, "a call envelope")
    tool = data.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ToolUsageError("envelope needs a non-empty string 'tool' (or an 'approval_id')")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ToolUsageError(f"envelope 'arguments' must be an object, got {_kind(arguments)}")
    run_id = _key(data, "run_id") if "run_id" in data else _derived("run_", source, message_id)
    call_id = _key(data, "call_id") if "call_id" in data else _derived("call_", source, message_id)
    return Envelope(tool=tool, arguments=dict(arguments), run_id=run_id, call_id=call_id)


def _as_mapping(payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if not isinstance(payload, str | bytes | bytearray):
        raise ToolUsageError(f"envelope must be a JSON object, got {_kind(payload)}")
    size = len(payload.encode("utf-8", "surrogatepass") if isinstance(payload, str) else payload)
    if size > MAX_ENVELOPE_BYTES:
        raise ToolUsageError(f"envelope is {size} B; the limit is {MAX_ENVELOPE_BYTES} B")
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ToolUsageError("envelope is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ToolUsageError(f"envelope must be a JSON object, got {_kind(data)}")
    return data


def _only(keys: set[str], allowed: frozenset[str], what: str) -> None:
    unknown = sorted(keys - allowed)
    if unknown:  # e.g. "approved": a decision in a message must never count
        raise ToolUsageError(f"{what} accepts only {sorted(allowed)}; unknown keys {unknown}")


def _key(data: Mapping[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str) or not 0 < len(value) <= MAX_KEY_CHARS:
        raise ToolUsageError(f"envelope {name!r} must be a string of 1-{MAX_KEY_CHARS} chars")
    if not value.isprintable():
        raise ToolUsageError(f"envelope {name!r} must be printable")
    return value


def _derived(prefix: str, source: str, message_id: str) -> str:
    return scoped_id(prefix, source, message_id) if message_id else new_id(prefix)


def _kind(value: object) -> str:
    return type(value).__name__
