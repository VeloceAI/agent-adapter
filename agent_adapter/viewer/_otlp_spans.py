"""One OTLP span as the start and end events the adapter would have recorded for it."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from agent_adapter.tracing import semconv as sc

_ERROR_CODES = {2, "2", "STATUS_CODE_ERROR"}
MAX_ID = 128  # characters kept of a span or parent id
_LAST_NS = 10**13 * 10**6  # run file names hold 13-digit millisecond starts
_INFERENCE = {sc.OP_CHAT, sc.OP_GENERATE_CONTENT, "text_completion"}
_PROVIDERS = {name: adapter for adapter, name in sc.PROVIDER_NAMES.items()}

Event = dict[str, Any]
Data = dict[str, Any]


def span_events(span: Mapping[str, Any], attrs: Mapping[str, Any], run_id: str) -> list[Event]:
    """`[started, finished-or-failed]` for a GenAI, flow or node span; `[]` for any other."""
    name = str(span.get("name") or "")
    op = str(attrs.get(sc.GEN_AI_OPERATION_NAME) or "")
    family = _family(op, name)
    if family is None:
        return []
    start, end = _seconds(span.get("startTimeUnixNano")), _seconds(span.get("endTimeUnixNano"))
    end = max(end, start)
    error = _error(span, attrs)
    build = _BUILDERS[family]
    begin, finish = build(attrs, name, (end - start) * 1000, error)
    kind = "failed" if error else "finished"
    if family == "flow":
        kind = "finished"  # a failed flow says so in its status
    base = {
        "run_id": run_id,
        "call_id": str(span.get("spanId") or "")[:MAX_ID],
        "parent_id": str(span.get("parentSpanId") or "")[:MAX_ID],
    }
    return [
        {"type": f"{family}.started", **base, "ts": start, "data": begin},
        {"type": f"{family}.{kind}", **base, "ts": end, "data": finish},
    ]


def _family(op: str, name: str) -> str | None:
    if op in _INFERENCE:
        return "llm"
    if op == sc.OP_EXECUTE_TOOL:
        return "tool"
    if op == sc.OP_EMBEDDINGS:
        return "embedding"
    if op == sc.OP_INVOKE_WORKFLOW:
        return "flow"
    if name.startswith(f"{sc.OP_NODE} "):
        return "node"
    return None


def _seconds(nanos: Any) -> float:
    """Epoch seconds; a missing, bad, negative or too-late time counts as 0."""
    try:
        value = int(nanos)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return value / 1e9 if 0 <= value < _LAST_NS else 0.0


def _error(span: Mapping[str, Any], attrs: Mapping[str, Any]) -> str:
    """The error kind: `error.type`, else `error` for an error status, else ""."""
    status = span.get("status")
    failed = isinstance(status, Mapping) and status.get("code") in _ERROR_CODES
    kind = attrs.get(sc.ERROR_TYPE)
    if isinstance(kind, str) and kind:
        return kind
    return "error" if failed else ""


def _content(data: Data, key: str, value: Any) -> None:
    """Captured content, parsed when it is JSON text (so secret-named keys get masked)."""
    if value is None:
        return
    if isinstance(value, str) and value[:1] in "[{":
        with contextlib.suppress(ValueError, RecursionError):  # else kept as the text it is
            value = json.loads(value)
    data[key] = value


def _int(attrs: Mapping[str, Any], key: str) -> int:
    value = attrs.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _llm(attrs: Mapping[str, Any], name: str, ms: float, error: str) -> tuple[Data, Data]:
    provider = str(attrs.get(sc.GEN_AI_PROVIDER_NAME) or "otel")
    provider = _PROVIDERS.get(provider, provider)
    model = str(attrs.get(sc.GEN_AI_REQUEST_MODEL) or name.partition(" ")[2] or "model")
    stage = str(attrs.get(sc.AA_STAGE) or "")
    begin: Data = {"provider": provider, "model": model, "stage": stage}
    begin["stream"] = bool(attrs.get(sc.GEN_AI_REQUEST_STREAM, False))
    _content(begin, "input_messages", attrs.get(sc.GEN_AI_INPUT_MESSAGES))
    finish: Data = {"provider": provider, "model": model, "stage": stage, "latency_ms": ms}
    if error:
        return begin, {**finish, "error_kind": error}
    cached = _int(attrs, sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
    written = _int(attrs, sc.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS)
    total = _int(attrs, sc.GEN_AI_USAGE_INPUT_TOKENS)
    finish["usage"] = {
        "input_tokens": max(total - cached - written, 0),  # OTel counts cache use inside input
        "cached_input_tokens": cached,
        "cache_write_input_tokens": written,
        "output_tokens": _int(attrs, sc.GEN_AI_USAGE_OUTPUT_TOKENS),
        "reasoning_tokens": _int(attrs, sc.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS),
    }
    reasons = attrs.get(sc.GEN_AI_RESPONSE_FINISH_REASONS)
    finish["finish_reason"] = str(reasons[0]) if isinstance(reasons, list) and reasons else "stop"
    finish["cost_usd"] = _cost(attrs)
    finish["response_model"] = str(attrs.get(sc.GEN_AI_RESPONSE_MODEL) or model)
    _content(finish, "output_messages", attrs.get(sc.GEN_AI_OUTPUT_MESSAGES))
    return begin, finish


def _cost(attrs: Mapping[str, Any]) -> float | None:
    cost = attrs.get(sc.AA_COST_USD)
    return float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None


def _tool(attrs: Mapping[str, Any], name: str, ms: float, error: str) -> tuple[Data, Data]:
    tool = str(attrs.get(sc.GEN_AI_TOOL_NAME) or name.partition(" ")[2] or "tool")
    begin: Data = {"tool": tool, "kind": str(attrs.get(sc.AA_TOOL_KIND) or "")}
    _content(begin, "arguments", attrs.get(sc.GEN_AI_TOOL_CALL_ARGUMENTS))
    attempts = _int(attrs, sc.AA_ATTEMPTS) or 1
    finish: Data = {"tool": tool, "latency_ms": ms, "attempts": attempts}
    if error:
        return begin, {**finish, "error_kind": error}
    finish["cached"] = attrs.get(sc.AA_CACHE_HIT) is True
    _content(finish, "result", attrs.get(sc.GEN_AI_TOOL_CALL_RESULT))
    return begin, finish


def _embedding(attrs: Mapping[str, Any], name: str, ms: float, error: str) -> tuple[Data, Data]:
    provider = str(attrs.get(sc.GEN_AI_PROVIDER_NAME) or "otel")
    provider = _PROVIDERS.get(provider, provider)
    model = str(attrs.get(sc.GEN_AI_REQUEST_MODEL) or name.partition(" ")[2] or "model")
    begin: Data = {"provider": provider, "model": model, "stage": str(attrs.get(sc.AA_STAGE) or "")}
    begin["texts"] = _int(attrs, sc.AA_EMBEDDING_TEXTS)
    finish: Data = {**begin, "latency_ms": ms}
    if error:
        return begin, {**finish, "error_kind": error}
    finish["usage"] = {"input_tokens": _int(attrs, sc.GEN_AI_USAGE_INPUT_TOKENS)}
    finish["cost_usd"] = _cost(attrs)
    finish["batches"] = _int(attrs, sc.AA_EMBEDDING_BATCHES)
    finish["cache_hits"] = _int(attrs, sc.AA_CACHE_HITS)
    return begin, finish


def _flow(attrs: Mapping[str, Any], name: str, ms: float, error: str) -> tuple[Data, Data]:
    flow = str(attrs.get(sc.GEN_AI_WORKFLOW_NAME) or name.partition(" ")[2] or "flow")
    status = str(attrs.get(sc.AA_FLOW_STATUS) or ("failed" if error else "done"))
    finish: Data = {"flow": flow, "status": status, "steps": _int(attrs, sc.AA_FLOW_STEPS)}
    if error:
        finish["error_kind"] = error
    return {"flow": flow}, finish


def _node(attrs: Mapping[str, Any], name: str, ms: float, error: str) -> tuple[Data, Data]:
    flow, _, node = name.partition(" ")[2].partition(".")
    data: Data = {
        "flow": str(attrs.get(sc.GEN_AI_WORKFLOW_NAME) or flow),
        "node": str(attrs.get(sc.AA_FLOW_NODE) or node),
        "visit": _int(attrs, sc.AA_FLOW_VISIT),
    }
    return data, ({**data, "error_kind": error} if error else dict(data))


_BUILDERS: dict[str, Callable[[Mapping[str, Any], str, float, str], tuple[Data, Data]]] = {
    "llm": _llm,
    "tool": _tool,
    "embedding": _embedding,
    "flow": _flow,
    "node": _node,
}
