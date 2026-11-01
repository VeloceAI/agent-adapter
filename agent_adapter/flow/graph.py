"""The validated flow IR (`FlowGraph`) that the local runner and the compilers consume."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow.actions import Call, Join, Map
from agent_adapter.flow.state import MergeRule, merge_rules

NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
_NAME = re.compile(NAME_PATTERN)
LOOP_FIELD = "aa_loop_counts"  # reserved: compilers keep route loop counters there
VISITS_FIELD = "aa_visits"  # reserved: compilers keep node visit counts there
RESERVED_FIELDS = (LOOP_FIELD, VISITS_FIELD)

NodeKind = Literal["fn", "call", "map", "join"]


@dataclass(frozen=True)
class NodeSpec:
    """One node: its action and run options."""

    name: str
    kind: NodeKind
    action: Any
    concurrency: int | None
    timeout: float | None
    retries: int
    pause: bool | Callable[..., Mapping[str, Any]]
    prompt: str | None


@dataclass(frozen=True)
class RouteSpec:
    """A conditional exit: `fn(state)` returns labels; `targets` maps label -> node."""

    source: str
    fn: Callable[..., Any]
    targets: Mapping[str, str]
    max_loops: int | None
    on_exhausted: str | None


@dataclass(frozen=True)
class FlowGraph:
    """A validated flow: nodes in declaration order, static edges, routes, ends and merge rules."""

    name: str
    state_model: type[BaseModel]
    start: str
    nodes: Mapping[str, NodeSpec]
    edges: Mapping[str, tuple[str, ...]]
    routes: Mapping[str, RouteSpec]
    ends: frozenset[str]
    rules: Mapping[str, MergeRule]

    def successors(self, node: str) -> tuple[str, ...]:
        """Every node `node` can lead to: static edges, then route targets."""
        found = list(self.edges.get(node, ()))
        route = self.routes.get(node)
        if route is not None:
            found.extend(route.targets.values())
        return tuple(dict.fromkeys(found))

    def predecessors(self, node: str) -> tuple[str, ...]:
        """Every node that can lead to `node`, in declaration order."""
        return tuple(name for name in self.nodes if node in self.successors(name))


def node_kind(action: Any) -> NodeKind | None:
    """The kind of a node action, or None when it is not one."""
    if isinstance(action, Call):
        return "call"
    if isinstance(action, Map):
        return "map"
    if isinstance(action, Join):
        return "join"
    return "fn" if callable(action) else None


def build_graph(
    name: str,
    state_model: type[BaseModel],
    *,
    nodes: Sequence[NodeSpec],
    edges: Sequence[tuple[str, str]],
    routes: Sequence[RouteSpec],
    start: str | None,
    ends: Sequence[str],
    problems: Sequence[str] = (),
) -> FlowGraph:
    """Check a flow definition and return its graph.

    Raises:
        FlowDefinitionError: every problem found, listed together.
    """
    check = _Checker(name, state_model)
    check.state()
    check.found.extend(problems)
    table = check.nodes(nodes)
    first = start if start is not None else next(iter(table), None)
    check.endpoints(table, first, ends, edges)
    route_table = check.routes(table, routes)
    static = {node: tuple(dict.fromkeys(d for s, d in edges if s == node)) for node in table}
    graph = FlowGraph(
        name=name,
        state_model=state_model,
        start=first or "",
        nodes=MappingProxyType(table),
        edges=MappingProxyType(static),
        routes=MappingProxyType(route_table),
        ends=frozenset(ends),
        rules=MappingProxyType(check.rules),
    )
    if first in table:
        check.shape(graph)
    if check.found:
        raise FlowDefinitionError(check.found, flow=name)
    return graph


class _Checker:
    """Collects definition problems in a fixed order."""

    def __init__(self, name: str, state_model: type[BaseModel]) -> None:
        self.flow = name
        self.state_model = state_model
        self.fields = state_model.model_fields
        self.rules: dict[str, MergeRule] = {}
        self.found: list[str] = []
        if not _NAME.match(name):
            self.found.append(f"flow name {name!r} is not an identifier ({NAME_PATTERN})")

    def state(self) -> None:
        for reserved in RESERVED_FIELDS:
            if reserved in self.fields:
                self.found.append(f"state field {reserved!r} is reserved")
        try:
            self.rules = merge_rules(self.state_model)
        except FlowDefinitionError as exc:
            self.found.extend(exc.problems)

    def nodes(self, nodes: Sequence[NodeSpec]) -> dict[str, NodeSpec]:
        table: dict[str, NodeSpec] = {}
        for node in nodes:
            if node.name in table:
                continue  # the builder reported the duplicate
            table[node.name] = node
            if not _NAME.match(node.name):
                self.found.append(f"node name {node.name!r} is not an identifier ({NAME_PATTERN})")
            self._action(node)
            self._options(node)
        if not table:
            self.found.append("flow has no nodes")
        return table

    def _action(self, node: NodeSpec) -> None:
        action, where = node.action, f"node {node.name!r}"
        if node_kind(action) is None:
            self.found.append(
                f"{where}: action must be a function, Call, Map or Join, "
                f"got {type(action).__name__}"
            )
        elif isinstance(action, Map):
            if action.over not in self.fields:
                self.found.append(f"{where}: Map.over {action.over!r} is not a state field")
            into = action.into or node.name
            if into not in self.fields:
                self.found.append(f"{where}: Map writes {into!r}, which is not a state field")
            if action.on_error not in ("fail", "skip"):
                self.found.append(
                    f"{where}: Map.on_error must be 'fail' or 'skip', got {action.on_error!r}"
                )
        elif isinstance(action, Call) and action.into not in (None, *self.fields):
            self.found.append(f"{where}: Call.into {action.into!r} is not a state field")

    def _options(self, node: NodeSpec) -> None:
        where = f"node {node.name!r}"
        concurrency, timeout, retries = node.concurrency, node.timeout, node.retries
        if concurrency is not None and (not _is_int(concurrency) or concurrency < 1):
            self.found.append(f"{where}: concurrency must be an int >= 1, got {concurrency!r}")
        if timeout is not None and not _is_positive(timeout):
            self.found.append(f"{where}: timeout must be a positive number, got {timeout!r}")
        if not _is_int(retries) or retries < 0:
            self.found.append(f"{where}: retries must be an int >= 0, got {retries!r}")
        if not isinstance(node.pause, bool) and not callable(node.pause):
            self.found.append(f"{where}: pause must be a bool or a callable, got {node.pause!r}")
        if node.prompt is not None and not isinstance(node.prompt, str):
            self.found.append(f"{where}: prompt must be a string, got {node.prompt!r}")

    def endpoints(
        self,
        table: Mapping[str, NodeSpec],
        start: str | None,
        ends: Sequence[str],
        edges: Sequence[tuple[str, str]],
    ) -> None:
        if start is not None and start not in table:
            self.found.append(f"start: unknown node {start!r}")
        self.found.extend(f"end: unknown node {end!r}" for end in ends if end not in table)
        for src, dst in edges:
            for end in dict.fromkeys((src, dst)):
                if end not in table:
                    self.found.append(f"edge {src} -> {dst}: unknown node {end!r}")

    def routes(
        self, table: Mapping[str, NodeSpec], routes: Sequence[RouteSpec]
    ) -> dict[str, RouteSpec]:
        kept: dict[str, RouteSpec] = {}
        for route in routes:
            where = f"route from {route.source}"
            if route.source not in table:
                self.found.append(f"{where}: unknown node {route.source!r}")
            elif route.source in kept:
                self.found.append(f"node {route.source!r} has more than one route")
                continue
            kept[route.source] = route
            for target in dict.fromkeys(route.targets.values()):
                if target not in table:
                    self.found.append(f"{where}: unknown target {target!r}")
            loops = route.max_loops
            if loops is not None and (not _is_int(loops) or loops < 0):
                self.found.append(f"{where}: max_loops must be an int >= 0, got {loops!r}")
            if route.on_exhausted is not None:
                if loops is None:
                    self.found.append(f"{where}: on_exhausted needs max_loops")
                if route.on_exhausted not in route.targets.values():
                    self.found.append(
                        f"{where}: on_exhausted {route.on_exhausted!r} is not one of its targets"
                    )
        return {source: route for source, route in kept.items() if source in table}

    def shape(self, graph: FlowGraph) -> None:
        for name in graph.nodes:
            has_edges, has_route = bool(graph.edges[name]), name in graph.routes
            if has_edges and has_route:
                self.found.append(f"node {name!r} has static edges and a route; use one")
            exits = has_edges or has_route
            if name in graph.ends and exits:
                self.found.append(f"end node {name!r} has an exit")
            elif name not in graph.ends and not exits:
                self.found.append(f"node {name!r} has no exit and is not an end node")
        reached = _reachable(graph)
        self.found.extend(
            f"node {name!r} is not reachable from {graph.start!r}"
            for name in graph.nodes
            if name not in reached
        )
        for name, node in graph.nodes.items():
            if isinstance(node.action, Join):
                self._join(graph, name, node.action)
        self.found.extend(_uncapped_cycles(graph))

    def _join(self, graph: FlowGraph, name: str, join: Join) -> None:
        if not join.sources:
            self.found.append(f"join {name!r} has no sources")
        before = graph.predecessors(name)
        for source in join.sources:
            if source not in graph.nodes:
                self.found.append(f"join {name!r}: unknown source {source!r}")
            elif source not in before:
                self.found.append(f"join {name!r}: source {source!r} is not a predecessor")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and value > 0


def _reachable(graph: FlowGraph) -> set[str]:
    seen = {graph.start}
    todo = [graph.start]
    while todo:
        for nxt in graph.successors(todo.pop()):
            if nxt in graph.nodes and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _uncapped_cycles(graph: FlowGraph) -> list[str]:
    """One problem per cycle that no route with `max_loops` breaks."""

    def loop_edges(node: str) -> list[str]:
        found = list(graph.edges[node])
        route = graph.routes.get(node)
        if route is not None and route.max_loops is None:
            found.extend(route.targets.values())
        return [n for n in dict.fromkeys(found) if n in graph.nodes]

    problems: list[str] = []
    seen_cycles: set[frozenset[str]] = set()
    state: dict[str, int] = {}  # 1 = on the current path, 2 = done
    path: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        path.append(node)
        for nxt in loop_edges(node):
            if state.get(nxt) == 1:
                cycle = path[path.index(nxt) :]
                if frozenset(cycle) not in seen_cycles:
                    seen_cycles.add(frozenset(cycle))
                    text = " -> ".join([*cycle, nxt])
                    problems.append(f"cycle {text} does not pass through a route with max_loops")
            elif nxt not in state:
                visit(nxt)
        path.pop()
        state[node] = 2

    for name in graph.nodes:
        if name not in state:
            visit(name)
    return problems
