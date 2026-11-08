"""Value types shared by every layer: tool calls and results, events, usage, ledger, approval."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, TypeAdapter, model_validator

JSONValue = Any  # documentation alias: a JSON-serialisable value

_FROZEN = ConfigDict(frozen=True)
# Bytes as base64: arbitrary (non-UTF-8) bytes must not break JSON-safe output.
_ANY: TypeAdapter[Any] = TypeAdapter(Any, config=ConfigDict(ser_json_bytes="base64"))

CacheScope = Literal["run", "thread", "global"]  # who shares a cached tool result


def _errors() -> Any:
    """The errors module, imported on use: errors stays free of package imports."""
    from agent_adapter.core import errors

    return errors


def to_jsonable(value: Any) -> JSONValue:
    """`value` in JSON mode; bytes become base64, unknown objects `str(obj)`."""
    return _ANY.dump_python(value, mode="json", fallback=str)


class ToolCall(BaseModel):
    """One tool call requested by a model or a host."""

    model_config = _FROZEN

    name: str
    arguments: dict[str, Any] = {}
    id: str = ""  # provider call id; the executor fills a new id when ""


class ErrorInfo(BaseModel):
    """JSON-safe description of a failure; never holds input values."""

    model_config = _FROZEN

    kind: str
    message: str
    exception: str = ""
    retryable: bool = False
    details: dict[str, Any] = {}

    @classmethod
    def from_exception(cls, exc: BaseException) -> ErrorInfo:
        """Describe `exc`; foreign exceptions expose only their class name as the message."""
        if not isinstance(exc, _errors().AdapterError):
            return cls(
                kind="internal-error",
                message=type(exc).__name__,
                exception=_errors().qualified_name(type(exc)),
                retryable=isinstance(exc, Exception),
            )
        kind = type(exc).kind
        return cls(
            kind=kind,
            message=str(exc),
            exception=_errors().qualified_name(type(exc)),
            retryable=type(exc).retryable,
            details=to_jsonable(_details(kind, exc)),
        )


def _details(kind: str, exc: BaseException) -> dict[str, Any]:
    if kind == "tool-args-invalid":
        return {"errors": list(getattr(exc, "errors", []))}
    if kind == "approval-pending":
        return {"approval_id": getattr(exc, "approval_id", "")}
    if kind == "tool-failed":
        return {"cause_type": getattr(exc, "cause_type", "")}
    if kind == "tool-error":
        return dict(getattr(exc, "details", {}))
    return {}


ToolStatus = Literal["ok", "error", "pending"]


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call."""

    call_id: str
    name: str
    status: ToolStatus
    value: Any = None
    error: ErrorInfo | None = None
    exception: BaseException | None = field(default=None, repr=False, compare=False)
    attempts: int = 0  # 0 when cached or short-circuited
    cached: bool = False
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the call succeeded."""
        return self.status == "ok"

    def unwrap(self) -> Any:
        """The value; else the kept exception, else ApprovalPending (pending) or ToolFailed."""
        if self.ok:
            return self.value
        if self.exception is not None:
            raise self.exception
        errors = _errors()
        if self.status == "pending":
            approval_id = str(self.error.details.get("approval_id", "")) if self.error else ""
            raise errors.ApprovalPending(approval_id, tool=self.name, call_id=self.call_id)
        raise errors.ToolFailed(self.name, call_id=self.call_id, attempts=self.attempts)

    def to_dict(self) -> dict[str, JSONValue]:
        """JSON-safe dict without the exception object."""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "status": self.status,
            "value": to_jsonable(self.value),
            "error": self.error.model_dump(mode="json") if self.error is not None else None,
            "attempts": self.attempts,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class Event:
    """A trace event (see `tracing.events` for the catalogue)."""

    type: str
    run_id: str = ""
    call_id: str = ""
    parent_id: str = ""
    ts: float = field(default_factory=time.time)  # epoch seconds
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """JSON-safe dict."""
        return {
            "type": self.type,
            "run_id": self.run_id,
            "call_id": self.call_id,
            "parent_id": self.parent_id,
            "ts": self.ts,
            "data": to_jsonable(dict(self.data)),
        }


class Usage(BaseModel):
    """Normalised token usage of one or more LLM calls."""

    model_config = _FROZEN

    input_tokens: NonNegativeInt = 0  # uncached, non-write input
    cached_input_tokens: NonNegativeInt = 0
    cache_write_input_tokens: NonNegativeInt = 0
    cache_write_by_ttl: dict[str, NonNegativeInt] = {}  # e.g. {"5m": 1200, "1h": 0}
    output_tokens: NonNegativeInt = 0  # includes reasoning
    reasoning_tokens: NonNegativeInt = 0  # informational
    units: float = Field(default=0.0, ge=0)  # non-token billing units

    def __add__(self, other: object) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        by_ttl = dict(self.cache_write_by_ttl)
        for ttl, count in other.cache_write_by_ttl.items():
            by_ttl[ttl] = by_ttl.get(ttl, 0) + count
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_input_tokens=self.cache_write_input_tokens + other.cache_write_input_tokens,
            cache_write_by_ttl=by_ttl,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            units=self.units + other.units,
        )

    @property
    def prompt_tokens(self) -> int:
        """Every input token: billable, cached and cache-write."""
        return self.input_tokens + self.cached_input_tokens + self.cache_write_input_tokens


LedgerKind = Literal["llm", "tool", "api"]
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "cache_write_by_ttl",
    "output_tokens",
    "reasoning_tokens",
    "units",
)


class LedgerEntry(BaseModel):
    """One billable call; idempotent on (run_id, entry_id)."""

    model_config = _FROZEN

    run_id: str
    call_id: str
    entry_id: str = ""  # idempotency key; defaults to call_id
    kind: LedgerKind
    stage: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_write_by_ttl: dict[str, int] = {}
    output_tokens: int = 0
    reasoning_tokens: int = 0
    units: float = 0.0
    cost_usd: float | None = None  # None: unknown price
    latency_ms: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    ok: bool = True
    error_kind: str = ""
    cache_hit: bool = False
    attempts: int = 1
    thread_id: str = ""
    price_version: str = ""
    metadata: dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def _default_entry_id(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and not data.get("entry_id"):
            return {**data, "entry_id": data.get("call_id", "")}
        return data

    @classmethod
    def from_usage(cls, usage: Usage, **fields: Any) -> LedgerEntry:
        """An entry whose token counts come from `usage`; `fields` sets the rest."""
        counts = {name: getattr(usage, name) for name in _USAGE_FIELDS}
        return cls(**{**counts, **fields})


class ApprovalRequest(BaseModel):
    """A call (or flow node) waiting for a human decision."""

    model_config = _FROZEN

    approval_id: str
    run_id: str
    call_id: str
    tool: str  # tool name, or "node:<flow>.<node>" for flow pauses
    arguments: dict[str, Any] = {}  # LLM-facing names, JSON mode
    kind: Literal["tool", "node"] = "tool"
    hint: str = ""
    metadata: dict[str, Any] = {}
    created_at: float = 0.0


class ApprovalDecision(BaseModel):
    """A reviewer's decision; `edited_args` replaces the arguments when set."""

    model_config = _FROZEN

    approved: bool
    edited_args: dict[str, Any] | None = None
    reason: str = ""
    decided_by: str = ""
    decided_at: float = 0.0


Decision = ApprovalDecision


class ModelCaps(BaseModel):
    """Capability flags of one model; the defaults disable every gated feature."""

    model_config = _FROZEN

    prompt_cache_key: bool = False  # OpenAI prompt_cache_key
    prompt_cache_retention_24h: bool = False
    prompt_cache_options: bool = False  # OpenAI gpt-5.6+ prompt_cache_options
    explicit_cache: bool = False  # Gemini caches.create
    cache_control: bool = False  # Anthropic cache_control
    min_cache_tokens: int = 1024
    structured_output: bool = True
    reasoning: bool = False
    context_window: int | None = Field(default=None, gt=0)  # tokens, prompt plus reply
    max_output_tokens: int | None = Field(default=None, gt=0)  # tokens; one reply's ceiling

    @model_validator(mode="after")
    def _check_window(self) -> ModelCaps:
        window, output = self.context_window, self.max_output_tokens
        if window is not None and output is not None and output >= window:
            raise ValueError("max_output_tokens must be less than context_window")
        return self

    def prompt_window(self) -> int | None:
        """Tokens a prompt may use: `context_window` less `max_output_tokens`; None if unknown."""
        if self.context_window is None:
            return None
        return self.context_window - (self.max_output_tokens or 0)
