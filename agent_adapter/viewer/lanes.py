"""Timeline spans from recorded events, and one lane per concurrent call."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SPAN_FAMILIES = ("tool", "llm", "embedding", "rerank", "node", "flow")
END_TYPES = {
    "tool.finished": "ok",
    "tool.failed": "error",
    "tool.cached": "cached",
    "tool.pending": "pending",
    "llm.finished": "ok",
    "llm.cached": "cached",
    "llm.failed": "error",
    "embedding.finished": "ok",
    "embedding.failed": "error",
    "rerank.finished": "ok",
    "rerank.failed": "error",
    "node.finished": "ok",
    "node.failed": "error",
    "node.paused": "paused",
    "flow.finished": "ok",
}
# Events that end at their own timestamp and carry their duration.
_POINT_SPANS = {"context.compressed": "ok", "context.compress_failed": "error"}
_MARKS = {"tool.retry", "approval.requested", "approval.decided", "llm.fallback"}


def build_spans(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One span per call (tool, LLM, node, flow, compression), in start order, with lanes."""
    spans: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, event in enumerate(events):
        kind = str(event.get("type", ""))
        data = event.get("data") or {}
        ts = float(event.get("ts") or 0.0)
        family, key = span_identity(kind, str(event.get("call_id") or ""), data)
        if kind in _POINT_SPANS:
            latency = _number(data.get("latency_ms")) or 0.0
            span = _new(key, "context", ts - latency / 1000.0, event, index)
            _close(span, _POINT_SPANS[kind], ts, data, index)
            spans[key] = span
            order.append(key)
            continue
        if family not in SPAN_FAMILIES:
            continue
        current = spans.get(key)
        if kind.endswith(".started"):
            if current is None:
                spans[key] = _new(key, family, ts, event, index)
                order.append(key)
            continue
        if current is None:  # an end or mark whose start was not recorded
            current = spans[key] = _new(key, family, ts, event, index)
            order.append(key)
        if kind in _MARKS:
            current["marks"].append({"type": kind, "ts": ts, "event": index})
            if kind == "tool.retry":
                current["retries"] += 1
        elif kind in END_TYPES:
            _close(current, END_TYPES[kind], ts, data, index)
    found = [spans[key] for key in order]
    last = max((float(event.get("ts") or 0.0) for event in events), default=0.0)
    for span in found:
        if span["status"] == "running":
            span["end"] = max(span["start"], last)  # still open when the recording ends
    _attach_nodes(found)
    assign_lanes(found)
    return found


def assign_lanes(spans: Iterable[dict[str, Any]]) -> int:
    """Give each span the lowest free lane at its start (greedy interval colouring)."""
    busy: list[tuple[float, int]] = []  # (end, lane)
    free: list[int] = []
    lanes = 0
    for span in sorted(spans, key=lambda s: (s["start"], -s["end"])):  # enclosing first
        while busy and busy[0][0] <= span["start"]:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if free:
            lane = heapq.heappop(free)
        else:
            lane, lanes = lanes, lanes + 1
        span["lane"] = lane
        heapq.heappush(busy, (span["end"], lane))
    return lanes


def span_identity(kind: str, call_id: str, data: Mapping[str, Any]) -> tuple[str, str]:
    """(family, span key) of an event type; approvals belong to the tool call they gate."""
    family = kind.split(".", 1)[0]
    if family == "approval":
        family = "tool"
    if call_id:
        return family, f"{family}:{call_id}"
    if family == "flow":
        return family, f"flow:{data.get('flow', '')}"
    if family == "node":
        return family, f"node:{data.get('flow', '')}.{data.get('node', '')}.{data.get('visit', '')}"
    return family, f"{family}:{data.get('tool') or data.get('model') or ''}"


def _new(key: str, family: str, ts: float, event: Mapping[str, Any], index: int) -> dict[str, Any]:
    data = event.get("data") or {}
    return {
        "id": key,
        "family": family,
        "call_id": str(event.get("call_id") or ""),
        "parent_id": str(event.get("parent_id") or ""),
        "name": _name(family, data),
        "start": ts,
        "end": ts,
        "status": "running",
        "events": [index],
        "marks": [],
        "retries": 0,
        "cached": False,
        "cost_usd": None,
        "error_kind": "",
        "timeout": False,
        "node": str(data.get("node", "")) if family == "node" else "",
        "tokens": None,
        "lane": 0,
    }


def _close(
    span: dict[str, Any], status: str, ts: float, data: Mapping[str, Any], index: int
) -> None:
    span["end"] = max(ts, span["start"])
    span["status"] = status
    if index not in span["events"]:
        span["events"].append(index)
    span["cached"] = status == "cached" or bool(data.get("cached"))
    cost = _number(data.get("cost_usd"))
    if cost is None:
        cost = _number(data.get("summary_cost_usd"))
    span["cost_usd"] = cost
    kind = str(data.get("error_kind") or "")
    span["error_kind"] = kind
    span["timeout"] = "timeout" in kind
    if "tokens_before" in data:
        span["tokens"] = {"before": data.get("tokens_before"), "after": data.get("tokens_after")}


def _attach_nodes(spans: list[dict[str, Any]]) -> None:
    """Give every span the flow node it ran under, through its parent chain."""
    by_call = {span["call_id"]: span for span in spans if span["call_id"]}
    for span in spans:
        seen: set[str] = set()
        current = span
        while not current["node"] and current["parent_id"] and current["parent_id"] not in seen:
            seen.add(current["parent_id"])
            parent = by_call.get(current["parent_id"])
            if parent is None:
                break
            current = parent
        span["node"] = current["node"]


def _name(family: str, data: Mapping[str, Any]) -> str:
    if family == "tool":
        return str(data.get("tool", ""))
    if family in ("llm", "embedding", "rerank", "context"):
        return f"{data.get('provider', '')}/{data.get('model', '')}"
    if family == "node":
        return str(data.get("node", ""))
    if family == "flow":
        return str(data.get("flow", ""))
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
