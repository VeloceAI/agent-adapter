"""A flow graph as JSON-able nodes and edges, and as a mermaid flowchart."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.flow.actions import Call, Join, Map
from agent_adapter.flow.graph import FlowGraph, NodeSpec

_SHAPES: dict[str, tuple[str, str]] = {
    "fn": ("[", "]"),
    "call": ("([", "])"),
    "map": ("[[", "]]"),
    "join": ("{{", "}}"),
}


_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_SIZE = 64
_CACHE_LOCK = threading.Lock()


def cached_flow_graph(graph: FlowGraph, digest: str) -> dict[str, Any]:
    """`flow_graph(graph)`, built once per graph digest (callers must not change it)."""
    with _CACHE_LOCK:
        found = _CACHE.get(digest)
        if found is not None:
            _CACHE.move_to_end(digest)
            return found
    built = flow_graph(graph)
    with _CACHE_LOCK:
        _CACHE[digest] = built
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return built


def flow_graph(graph: FlowGraph) -> dict[str, Any]:
    """Nodes and edges of `graph`, marking fan-out, fan-in, routes, loops and pause gates."""
    back = back_edges(graph)
    edges: list[dict[str, Any]] = []
    for source in graph.nodes:
        static = graph.edges.get(source, ())
        parallel = len(static) > 1
        for target in static:
            loop = (source, target) in back
            edges.append(_edge(source, target, "edge", "", parallel=parallel, loop=loop))
        route = graph.routes.get(source)
        if route is None:
            continue
        labels: dict[str, list[str]] = {}
        for label, target in route.targets.items():
            labels.setdefault(target, [])
            if label != target:
                labels[target].append(label)
        for target, names in labels.items():
            edge = _edge(source, target, "route", ", ".join(names), loop=(source, target) in back)
            edge["max_loops"] = route.max_loops
            edge["on_exhausted"] = route.on_exhausted == target
            edges.append(edge)
    return {
        "flow": graph.name,
        "start": graph.start,
        "nodes": [_node(graph, spec) for spec in graph.nodes.values()],
        "edges": edges,
    }


def flow_mermaid(graph: FlowGraph) -> str:
    """`graph` as a mermaid `flowchart TD`; fan-out, fan-in, loops and pause gates are marked."""
    data = flow_graph(graph)
    lines = ["flowchart TD", f"  aa_start((start)) --> {_id(graph.start)}"]
    for node in data["nodes"]:
        opening, closing = _SHAPES[node["kind"]]
        text = _quote("<br/>".join(_node_lines(node)))
        style = ":::pause" if node["pause"] else ""
        lines.append(f"  {_id(node['id'])}{opening}{text}{closing}{style}")
    for edge in data["edges"]:
        lines.append(f"  {_id(edge['source'])} {_arrow(edge)} {_id(edge['target'])}")
    lines.extend(f"  {_id(name)} --> aa_end((end))" for name in graph.nodes if name in graph.ends)
    lines.append("  classDef pause stroke-dasharray: 5 5")
    return "\n".join(lines) + "\n"


def back_edges(graph: FlowGraph) -> set[tuple[str, str]]:
    """Edges that lead back to a node still on the depth-first path from the start."""
    found: set[tuple[str, str]] = set()
    state: dict[str, int] = {}  # 1 = on the path, 2 = done
    if graph.start not in graph.nodes:
        return found
    stack: list[tuple[str, list[str]]] = [(graph.start, list(graph.successors(graph.start)))]
    state[graph.start] = 1
    while stack:
        node, todo = stack[-1]
        if not todo:
            state[node] = 2
            stack.pop()
            continue
        nxt = todo.pop(0)
        if nxt not in graph.nodes:
            continue
        if state.get(nxt) == 1:
            found.add((node, nxt))
        elif nxt not in state:
            state[nxt] = 1
            stack.append((nxt, list(graph.successors(nxt))))
    return found


def _edge(source: str, target: str, kind: str, label: str, **flags: bool) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "kind": kind,
        "label": label,
        "parallel": flags.get("parallel", False),
        "loop": flags.get("loop", False),
        "max_loops": None,
        "on_exhausted": False,
    }


def _node(graph: FlowGraph, spec: NodeSpec) -> dict[str, Any]:
    action = spec.action
    pause = spec.pause
    return {
        "id": spec.name,
        "kind": spec.kind,
        "start": spec.name == graph.start,
        "end": spec.name in graph.ends,
        "fan_out": isinstance(action, Map),
        "fan_in": isinstance(action, Join),
        "pause": "always" if pause is True else ("rule" if callable(pause) else None),
        "tool": _tool(action),
        "over": action.over if isinstance(action, Map) else None,
        "sources": list(action.sources) if isinstance(action, Join) else [],
        "concurrency": spec.concurrency,
        "timeout": spec.timeout,
        "retries": spec.retries,
        "has_prompt": spec.prompt is not None,
    }


def _tool(action: Any) -> str | None:
    if isinstance(action, Call):
        return _tool_name(action.tool)
    if isinstance(action, Map):
        if action.tool is not None:
            return _tool_name(action.tool)
        if action.tool_from is not None:
            return f"item.{action.tool_from}"
        return _callable_name(action.fn)
    return None


def _tool_name(tool: str | ToolSpec) -> str:
    return tool.name if isinstance(tool, ToolSpec) else str(tool)


def _callable_name(fn: Callable[..., Any] | None) -> str | None:
    return getattr(fn, "__name__", None) if fn is not None else None


def _node_lines(node: dict[str, Any]) -> list[str]:
    lines = [str(node["id"])]
    if node["fan_out"]:
        lines.append(f"fan-out over {node['over']}")
    if node["fan_in"]:
        lines.append(f"fan-in: {', '.join(node['sources'])}")
    if node["tool"] and node["kind"] in ("call", "map"):
        lines.append(f"tool: {node['tool']}")
    if node["pause"]:
        lines.append("pause gate")
    return lines


def _arrow(edge: dict[str, Any]) -> str:
    text = edge["label"]
    if edge["max_loops"] is not None and edge["loop"]:
        text = f"{text} (max {edge['max_loops']})".strip()
    if edge["loop"]:
        arrow = "-.->"
    elif edge["parallel"]:
        arrow = "==>"
    else:
        arrow = "-->"
    return f"{arrow}|{_quote(text)}|" if text else arrow


def _id(name: str) -> str:
    # Prefixed so a node named `end` or `graph` never clashes with mermaid keywords.
    return f"n_{name}"


def _quote(text: str) -> str:
    escaped = text.replace("&", "#amp;").replace('"', "#quot;")
    escaped = escaped.replace("<br/>", "\0").replace("<", "#lt;").replace(">", "#gt;")
    return '"' + escaped.replace("\0", "<br/>") + '"'
