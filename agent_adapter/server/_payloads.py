"""Request bodies in, JSON payloads out; parsing fails closed with a 400 problem."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from agent_adapter.approval.store import ApprovalRecord, check_json
from agent_adapter.core.errors import AdapterError, ToolUsageError
from agent_adapter.core.registry import exported_to
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.anthropic import anthropic_tool
from agent_adapter.integrations.gemini import function_declaration
from agent_adapter.integrations.openai import chat_tool
from agent_adapter.server.problems import Problem, problem

MAX_KEY_LENGTH = 256  # client-chosen ids end up in stores, caches and logs
MAX_REASON_LENGTH = 2_000
SHAPES = ("neutral", "openai", "anthropic", "gemini")
_RENDERERS = {"openai": chat_tool, "anthropic": anthropic_tool, "gemini": function_declaration}


class Refused(AdapterError):
    """A request the server answers with `problem` (internal: never leaves `ToolService`)."""

    kind = "bad-request"

    def __init__(self, found: Problem) -> None:
        self.problem = found
        super().__init__(found.detail)


def refuse(kind: str, detail: str, **extensions: Any) -> NoReturn:
    raise Refused(problem(kind, detail, **extensions))


@dataclass(frozen=True)
class InvokeBody:
    arguments: dict[str, Any]
    run_key: str
    thread_key: str


@dataclass(frozen=True)
class DecideBody:
    approved: bool
    edited_arguments: dict[str, Any] | None
    reason: str


def parse_json(body: bytes) -> Any:
    """Strict JSON: UTF-8, no NaN / Infinity, nesting within the approval depth cap."""
    try:
        value = json.loads(body.decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        refuse("bad-request", "the body is not valid JSON")
    try:
        check_json(value)
    except ToolUsageError:
        refuse("bad-request", "the body is nested too deeply or holds non-JSON values")
    return value


def invoke_body(body: bytes, headers_run: str | None, headers_thread: str | None) -> InvokeBody:
    """`{arguments, run_id?, thread_id?}`; body ids beat the `X-Run-Id` / `X-Thread-Id` headers."""
    data = _object(parse_json(body), "the body", {"arguments", "run_id", "thread_id"})
    if "arguments" not in data:
        refuse("bad-request", "the body needs an 'arguments' object")
    arguments = _object(data["arguments"], "'arguments'", None)
    run_key = client_key(data.get("run_id", headers_run), "run_id")
    thread_key = client_key(data.get("thread_id", headers_thread), "thread_id")
    return InvokeBody(arguments=arguments, run_key=run_key, thread_key=thread_key)


def decide_body(body: bytes) -> DecideBody:
    """`{approved: bool, edited_arguments?: object, reason?: str}`."""
    data = _object(parse_json(body), "the body", {"approved", "edited_arguments", "reason"})
    approved = data.get("approved")
    if not isinstance(approved, bool):
        refuse("bad-request", "'approved' must be true or false")
    edited = data.get("edited_arguments")
    if edited is not None:
        edited = _object(edited, "'edited_arguments'", None)
    reason = data.get("reason", "")
    if not isinstance(reason, str) or len(reason) > MAX_REASON_LENGTH:
        refuse("bad-request", f"'reason' must be a string of at most {MAX_REASON_LENGTH} chars")
    return DecideBody(approved=approved, edited_arguments=edited, reason=reason)


def client_key(value: Any, name: str) -> str:
    """A client-chosen id: "" when absent, else 1-256 printable characters without "/"."""
    if value is None or value == "":
        return ""
    if (
        isinstance(value, str)
        and len(value) <= MAX_KEY_LENGTH
        and value.isprintable()
        and "/" not in value
    ):
        return value
    refuse("bad-request", f"{name} must be 1-{MAX_KEY_LENGTH} printable characters, no '/'")


def csv(value: str | None) -> list[str] | None:
    """`a,b` -> ["a", "b"]; None when the parameter is absent."""
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def shape_of(query: Mapping[str, str]) -> str:
    shape = query.get("shape", "neutral") or "neutral"
    if shape not in SHAPES:
        refuse("bad-request", f"shape must be one of {', '.join(SHAPES)}")
    return shape


def renders(spec: ToolSpec, shape: str) -> bool:
    """True when `spec` has a provider rendering for `shape` (neutral always does)."""
    return shape == "neutral" or exported_to(spec, shape)


def descriptor(spec: ToolSpec, shape: str) -> dict[str, Any]:
    """The ToolDescriptor of `spec`; `provider` only for a provider shape."""
    approval = (
        "never" if spec.confirm is False else "always" if spec.confirm is True else "conditional"
    )
    item: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.json_schema(),
        "output_schema": spec.result_schema(),
        "kind": spec.kind,
        "tags": sorted(spec.tags),
        "effects": spec.effects,
        "idempotent": spec.idempotent,
        "approval": approval,
        "requires_approval": approval != "never",
        "parallel_safe": spec.parallel_safe,
        "timeout": spec.timeout,
        "retries": spec.retries,
        "cache_ttl": spec.cache_ttl,
        "prompt_hint": spec.prompt_hint,
    }
    if shape != "neutral":
        item["provider"] = _RENDERERS[shape](spec)
    return item


def approval_payload(record: ApprovalRecord) -> dict[str, Any]:
    """The Approval payload: client run ids, never the executor's scoped ones."""
    request = record.request
    decision = record.decision
    return {
        "approval_id": record.approval_id,
        "status": record.status,
        "tool": request.tool,
        "arguments": request.arguments,
        "run_id": run_key_of(record),
        "call_id": request.call_id,
        "hint": request.hint,
        "created_at": request.created_at,
        "decision": decision.model_dump(mode="json") if decision is not None else None,
    }


def run_key_of(record: ApprovalRecord) -> str:
    """The client's run id of a record (the stored run id when another surface made it)."""
    key = record.request.metadata.get("run_key")
    return key if isinstance(key, str) and key else record.request.run_id


def _object(value: Any, what: str, allowed: set[str] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        refuse("bad-request", f"{what} must be a JSON object")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            refuse("bad-request", f"{what} has {len(unknown)} unknown field(s)")
    return value


def _no_constant(name: str) -> float:
    raise ValueError(f"{name} is not JSON")
