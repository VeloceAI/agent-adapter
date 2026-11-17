"""How a run's parallel and sequential work went: overlap, waits, claims and cost per agent."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent_adapter.core.types import Event
from agent_adapter.tracing._content import content_data
from agent_adapter.viewer.lanes import build_spans

MAIN = "main"  # calls outside every agent tool and flow node
_CACHED = ("tool.cached", "llm.cached")
_SHARED = "inflight.shared"


class _Everything:
    wants_content = True


def analyse_run(events: Iterable[Mapping[str, Any] | Event]) -> dict[str, Any]:
    """A run's overlap report from its events (recorded dicts or `Event`s), JSON-ready.

    Keys: `calls` (started tool and model calls), `duplicates` (same tool and arguments;
    arguments are only in events recorded with content), `saved` (calls answered by the
    cache or by an identical call in flight), `tokens` (model tokens used, and saved by
    the cache and by in-flight sharing), `joins` (parallel branches: how long each waited
    for the slowest), `parallel_ms` and `waited_ms` (their totals), `critical_path` (the
    slowest branch of each join, in order), `claims` (contested board claims), `agents`
    (calls, tokens, cost, busy time and board activity per agent or node) and `board`.
    """
    rows = [_row(event) for event in events]
    spans = build_spans(rows)
    kinds = _tool_kinds(rows)
    owner = _owners(spans, kinds)
    joins = _joins(spans, kinds, _flows(rows))
    return {
        "calls": {
            "tool": sum(1 for r in rows if r["type"] == "tool.started"),
            "llm": sum(1 for r in rows if r["type"] == "llm.started"),
        },
        **_duplicates(rows, owner),
        "saved": _saved(rows),
        "tokens": _tokens(rows),
        "joins": joins,
        "parallel_ms": round(sum(j["wall_ms"] for j in joins), 3),
        "waited_ms": round(sum(j["waited_ms"] for j in joins), 3),
        "critical_path": [j["slowest"] for j in joins],
        "claims": _claims(rows),
        "agents": _agents(rows, spans, kinds, owner),
        "board": [_board_point(r) for r in rows if r["type"].startswith("board.")],
    }


def _row(event: Mapping[str, Any] | Event) -> dict[str, Any]:
    if isinstance(event, Event):
        row: dict[str, Any] = event.to_dict()
        row["data"] = dict(content_data(event, _Everything()))
        return row
    data = event.get("data")
    return {
        **event,
        "type": str(event.get("type", "")),
        "data": data if isinstance(data, dict) else {},
    }


def _tool_kinds(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(r.get("call_id") or ""): str(r["data"].get("kind", ""))
        for r in rows
        if r["type"] == "tool.started"
    }


def _agent_of(span: Mapping[str, Any], kinds: Mapping[str, str]) -> bool:
    return span["family"] == "tool" and kinds.get(span["call_id"]) == "agent"


def _owners(spans: Sequence[dict[str, Any]], kinds: Mapping[str, str]) -> dict[str, str]:
    """call id -> the innermost agent around it (itself excluded), else its node, else MAIN."""
    by_call = {s["call_id"]: s for s in spans if s["call_id"]}
    owners: dict[str, str] = {}
    for span in spans:
        if not span["call_id"]:
            continue
        seen: set[str] = set()
        parent = by_call.get(span["parent_id"])
        found = ""
        while parent is not None and parent["call_id"] not in seen:
            seen.add(parent["call_id"])
            if _agent_of(parent, kinds):
                found = parent["name"]
                break
            parent = by_call.get(parent["parent_id"])
        owners[span["call_id"]] = found or span["node"] or MAIN
    return owners


def _duplicates(rows: Sequence[Mapping[str, Any]], owner: Mapping[str, str]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    recorded = False
    for row in rows:
        if row["type"] != "tool.finished" or "arguments" not in row["data"]:
            continue
        recorded = True
        args = json.dumps(row["data"]["arguments"], sort_keys=True, default=str)
        groups[(str(row["data"].get("tool", "")), args)].append(str(row.get("call_id") or ""))
    found: list[dict[str, Any]] = [
        {
            "tool": tool,
            "count": len(ids),
            "call_ids": ids,
            "agents": sorted({owner.get(i, MAIN) for i in ids}),
        }
        for (tool, _), ids in groups.items()
        if len(ids) > 1
    ]
    found.sort(key=lambda d: (-d["count"], d["tool"]))
    return {
        "duplicates": found,
        "duplicate_calls": sum(d["count"] - 1 for d in found),
        "arguments_recorded": recorded,
    }


def _saved(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shared = [r for r in rows if r["type"] == _SHARED]
    cached = sum(1 for r in rows if r["type"] in _CACHED)
    waited = sum(_number(r["data"].get("waited_ms")) for r in shared)
    return {
        "cache": max(0, cached - len(shared)),  # a waiter also emits its own `*.cached`
        "inflight": len(shared),
        "inflight_waited_ms": round(waited, 3),
    }


def _usage(usage: Any) -> tuple[int, int]:
    """(input, output) tokens of a usage dict; input counts cached and cache-write input."""
    if not isinstance(usage, Mapping):
        return 0, 0
    keys = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens")
    return int(sum(_number(usage.get(k)) for k in keys)), int(_number(usage.get("output_tokens")))


def _tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Model tokens used, and the tokens of replies the cache or an identical call gave."""
    shared = {
        str(r.get("call_id") or "")
        for r in rows
        if r["type"] == _SHARED and r["data"].get("kind") == "llm"
    }
    totals = dict.fromkeys(("input", "output", "saved_by_cache", "saved_by_inflight"), 0)
    for row in rows:
        tokens_in, tokens_out = _usage(row["data"].get("usage"))
        if row["type"] == "llm.finished" and not row["data"].get("nested"):
            totals["input"] += tokens_in
            totals["output"] += tokens_out
        elif row["type"] == "llm.cached":
            where = "inflight" if row.get("call_id") in shared else "cache"
            totals[f"saved_by_{where}"] += tokens_in + tokens_out
    return totals


def _flows(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(r.get("call_id") or ""): str(r["data"].get("flow", ""))
        for r in rows
        if r["type"] == "node.started"
    }


def _joins(
    spans: Sequence[dict[str, Any]], kinds: Mapping[str, str], flows: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Groups of branches that ran side by side and were joined: flow supersteps, agent fan-outs."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        if span["family"] == "node":
            flow = flows.get(span["call_id"], "")
            groups[("flow", f"{span['parent_id']}|{flow}")].append(span)
        elif _agent_of(span, kinds):
            groups[("agents", span["parent_id"])].append(span)
    found = []
    for (kind, key), members in groups.items():
        where = key.split("|", 1)[1] if kind == "flow" else key or MAIN
        for group in _overlapping(members):
            if len(group) > 1:
                found.append(_join(kind, where, group))
    found.sort(key=lambda j: j["start"])
    return found


def _overlapping(spans: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    end = float("-inf")
    for span in sorted(spans, key=lambda s: s["start"]):
        if not groups or span["start"] >= end:
            groups.append([span])
            end = span["end"]
        else:
            groups[-1].append(span)
            end = max(end, span["end"])
    return groups


def _join(kind: str, where: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    start = min(s["start"] for s in group)
    finish = max(s["end"] for s in group)
    slowest = max(group, key=lambda s: s["end"] - s["start"])
    branches = [
        {
            "name": s["name"],
            "duration_ms": _ms(s["end"] - s["start"]),
            "waited_ms": _ms(finish - s["end"]),
            "status": s["status"],
        }
        for s in sorted(group, key=lambda s: s["start"])
    ]
    return {
        "kind": kind,
        "where": where,
        "start": start,
        "wall_ms": _ms(finish - start),
        "slowest": slowest["name"],
        "waited_ms": round(sum(b["waited_ms"] for b in branches), 3),
        "branches": branches,
    }


def _claims(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    losers: dict[str, dict[str, Any]] = {}
    total = 0
    for row in rows:
        if row["type"] != "board.claimed":
            continue
        total += 1
        data = row["data"]
        if data.get("won") is False:
            entry = losers.setdefault(
                str(data.get("key", "")),
                {
                    "key": str(data.get("key", "")),
                    "holder": str(data.get("holder", "")),
                    "lost": [],
                },
            )
            entry["lost"].append(str(data.get("by", "")))
    return {"total": total, "contested": list(losers.values())}


def _agents(
    rows: Sequence[Mapping[str, Any]],
    spans: Sequence[dict[str, Any]],
    kinds: Mapping[str, str],
    owner: Mapping[str, str],
) -> list[dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}

    def of(name: str) -> dict[str, Any]:
        return table.setdefault(
            name,
            {
                "agent": name,
                "tool_calls": 0,
                "llm_calls": 0,
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "busy_ms": 0.0,
                "posts": 0,
                "claims_won": 0,
                "claims_lost": 0,
            },
        )

    for span in spans:
        if _agent_of(span, kinds):
            of(span["name"])["busy_ms"] += _ms(span["end"] - span["start"])
        if span["family"] not in ("tool", "llm") or not span["call_id"]:
            continue
        row = of(owner.get(span["call_id"], MAIN))
        row["tool_calls" if span["family"] == "tool" else "llm_calls"] += 1
        row["cost_usd"] += span["cost_usd"] or 0.0
    for event in rows:
        data = event["data"]
        if event["type"] == "llm.finished" and not data.get("nested"):
            row = of(owner.get(str(event.get("call_id") or ""), MAIN))
            tokens_in, tokens_out = _usage(data.get("usage"))
            row["input_tokens"] += tokens_in
            row["output_tokens"] += tokens_out
        elif event["type"] == "board.posted":
            of(str(data.get("by") or MAIN))["posts"] += 1
        elif event["type"] == "board.claimed":
            won = "claims_won" if data.get("won") else "claims_lost"
            of(str(data.get("by") or MAIN))[won] += 1
    for row in table.values():
        row["cost_usd"] = round(row["cost_usd"], 8)
        row["busy_ms"] = round(row["busy_ms"], 3)
    return sorted(table.values(), key=lambda r: (r["agent"] == MAIN, r["agent"]))


def _board_point(row: Mapping[str, Any]) -> dict[str, Any]:
    data = row["data"]
    point = {
        "type": row["type"],
        "ts": _number(row.get("ts")),
        "by": str(data.get("by", "")),
        "key": str(data.get("key", "")),
    }
    if row["type"] == "board.posted":
        point["kind"] = str(data.get("kind", ""))
        point["seq"] = data.get("seq")
    else:
        point["won"] = bool(data.get("won"))
        point["holder"] = str(data.get("holder", ""))
    return point


def _ms(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000, 3)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
