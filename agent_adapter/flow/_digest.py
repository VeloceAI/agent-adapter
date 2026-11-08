"""The graph digest a checkpoint carries: the flow's shape and the code identity of its parts."""

from __future__ import annotations

import functools
import hashlib
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec._canon import NotSealable, canonical_bytes, encode, type_name
from agent_adapter.flow.actions import Call, Join, Map, SubFlow
from agent_adapter.flow.graph import FlowGraph, NodeSpec


def graph_digest(graph: FlowGraph) -> str:
    """SHA-256 of the graph: names, kinds, actions, gates, edges, routes, start and ends.

    Functions count by identity (`module:qualname`), so renaming or moving one changes it; a
    changed body under the same name does not. A `Call` / `Map` counts by its tool, static
    arguments and options; a `SubFlow` by its child graph's digest and its field mapping.
    """
    shape = {
        "name": graph.name,
        "state": type_name(graph.state_model),
        "start": graph.start,
        "nodes": [[n, s.kind, _gate(s.pause), action_shape(s)] for n, s in graph.nodes.items()],
        "edges": {n: list(targets) for n, targets in graph.edges.items()},
        "routes": {
            n: [dict(r.targets), r.max_loops, r.on_exhausted, identity(r.fn)]
            for n, r in graph.routes.items()
        },
        "ends": sorted(graph.ends),
    }
    return hashlib.sha256(canonical_bytes(shape)).hexdigest()


def _gate(pause: Any) -> str:
    if isinstance(pause, bool):
        return "pause" if pause else ""
    return f"edit {identity(pause)}"


def action_shape(node: NodeSpec) -> list[Any]:
    """A node action's identity: its kind, tools, function identities and static options."""
    action = node.action
    if isinstance(action, Call):
        args = _maybe(action.args) if callable(action.args) else _static(action.args)
        return ["call", _tool(action.tool), action.into, args]
    if isinstance(action, Map):
        options = {
            "over": action.over,
            "tool": _tool(action.tool),
            "tool_from": action.tool_from,
            "args_from": action.args_from,
            "args_fn": _maybe(action.args_fn),
            "fn": _maybe(action.fn),
            "into": action.into,
            "on_error": action.on_error,
        }
        return ["map", options]
    if isinstance(action, Join):
        return ["join", list(action.sources), _maybe(action.fn)]
    if isinstance(action, SubFlow):
        child = graph_digest(node.child) if node.child is not None else None
        return ["subflow", child, _fields(action.input), _fields(action.into)]
    return ["fn", identity(action)]


def _fields(spec: Any) -> Any:
    """A SubFlow mapping in canonical form: a field name, a field table, or a function identity."""
    if spec is None or isinstance(spec, str):
        return spec
    if callable(spec):
        return identity(spec)
    return {"fields": {str(k): str(v) for k, v in dict(spec).items()}}


def _tool(tool: str | ToolSpec | None) -> str | None:
    return tool.name if isinstance(tool, ToolSpec) else tool


def _static(args: Any) -> Any:
    """Static `Call.args` in canonical form; unsealable ones by their type names only."""
    try:
        return {"args": encode(dict(args))}
    except NotSealable:
        return {"unsealable": sorted({type_name(type(v)) for v in dict(args).values()})}


def _maybe(fn: Any) -> str | None:
    return identity(fn) if callable(fn) else None


def identity(fn: Any) -> str:
    """A callable's stable identity: `module:qualname` (a partial: of its function)."""
    if isinstance(fn, functools.partial):
        return f"partial({identity(fn.func)})"
    target = fn if hasattr(fn, "__qualname__") else type(fn)
    module = getattr(target, "__module__", None) or "?"
    return f"{module}:{target.__qualname__}"
