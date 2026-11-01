"""The `Flow` builder: nodes, edges, routes, start and ends; `validate()` returns the graph."""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow.actions import Call, Join, Map, NodeFn, Route
from agent_adapter.flow.diagram import flow_graph, flow_mermaid
from agent_adapter.flow.graph import FlowGraph, NodeSpec, RouteSpec, build_graph, node_kind

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=BaseModel)


class Flow(Generic[S]):
    """A portable graph flow over a pydantic state model; every builder method returns `self`."""

    def __init__(self, name: str, *, state: type[S], description: str = "") -> None:
        if not (isinstance(state, type) and issubclass(state, BaseModel)):
            raise FlowDefinitionError(
                [f"state must be a pydantic BaseModel subclass, got {state!r}"], flow=str(name)
            )
        self.name = name
        self.state_model: type[S] = state
        self.description = description
        self._nodes: list[NodeSpec] = []
        self._edges: list[tuple[str, str]] = []
        self._routes: list[RouteSpec] = []
        self._start: str | None = None
        self._ends: list[str] = []
        self._problems: list[str] = []

    def node(
        self,
        name: str,
        action: NodeFn[S] | Call | Map | Join,
        *,
        concurrency: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        pause: bool | Callable[[S], Mapping[str, Any]] = False,
        prompt: str | None = None,
    ) -> Flow[S]:
        """Add a node: a function `fn(state)` / `fn(state, ctx)`, or a `Call`, `Map` or `Join`."""
        if any(node.name == name for node in self._nodes):
            self._problems.append(f"duplicate node {name!r}")
        self._nodes.append(
            NodeSpec(
                name=name,
                kind=node_kind(action) or "fn",
                action=action,
                concurrency=concurrency,
                timeout=timeout,
                retries=retries,
                pause=pause,
                prompt=prompt,
            )
        )
        return self

    def edge(self, src: str, dst: str | Sequence[str]) -> Flow[S]:
        """Add static edges; several destinations run as parallel branches in the next superstep."""
        targets = [dst] if isinstance(dst, str) else list(dst)
        for target in targets:
            if (src, target) not in self._edges:
                self._edges.append((src, target))
        return self

    def route(
        self,
        src: str,
        route: Route[S],
        *,
        max_loops: int | None = None,
        on_exhausted: str | None = None,
    ) -> Flow[S]:
        """Add a conditional exit from `src`; `max_loops` caps how often it may revisit nodes."""
        targets = route.targets
        if isinstance(targets, Mapping):
            table = dict(targets)
        elif isinstance(targets, str) or not isinstance(targets, Sequence):
            self._problems.append(
                f"route from {src}: targets must be node names or a label -> node mapping"
            )
            table = {}
        else:
            table = {target: target for target in targets}
        self._routes.append(
            RouteSpec(
                source=src,
                fn=route.fn,
                targets=table,
                max_loops=max_loops,
                on_exhausted=on_exhausted,
            )
        )
        return self

    def start(self, name: str) -> Flow[S]:
        """Set the start node (default: the first node added)."""
        self._start = name
        return self

    def end(self, *names: str) -> Flow[S]:
        """Mark end nodes: they have no exit, and the flow is done once nothing else is pending."""
        self._ends.extend(name for name in names if name not in self._ends)
        return self

    def validate(self) -> FlowGraph:
        """Check the definition and return its graph.

        Raises:
            FlowDefinitionError: every problem found, listed together.
        """
        return build_graph(
            self.name,
            self.state_model,
            nodes=self._nodes,
            edges=self._edges,
            routes=self._routes,
            start=self._start,
            ends=self._ends,
            problems=self._problems,
        )

    def graph(self) -> dict[str, Any]:
        """The validated graph as JSON-able nodes and edges (fan-out, fan-in, routes, loops, gates).

        Raises:
            FlowDefinitionError: the flow is invalid.
        """
        return flow_graph(self.validate())

    def to_mermaid(self) -> str:
        """The validated graph as a mermaid flowchart.

        Raises:
            FlowDefinitionError: the flow is invalid.
        """
        return flow_mermaid(self.validate())

    def prompts(self) -> dict[str, str]:
        """The prompt of every node that has one, by node name."""
        return {node.name: node.prompt for node in self._nodes if isinstance(node.prompt, str)}

    def with_prompts(self, prompts: Mapping[str, str]) -> Flow[S]:
        """A copy whose named nodes carry the given prompts; this flow is unchanged.

        Raises:
            FlowDefinitionError: a name is not a node of this flow.
        """
        known = {node.name for node in self._nodes}
        unknown = [name for name in prompts if name not in known]
        if unknown:
            raise FlowDefinitionError(
                [f"unknown node {name!r}" for name in unknown], flow=self.name
            )
        clone = copy.copy(self)
        clone._nodes = [
            _with_prompt(node, prompts[node.name]) if node.name in prompts else node
            for node in self._nodes
        ]
        clone._edges = list(self._edges)
        clone._routes = list(self._routes)
        clone._ends = list(self._ends)
        clone._problems = list(self._problems)
        logger.debug("flow %s: copy with %d replaced prompts", self.name, len(prompts))
        return clone

    def __repr__(self) -> str:
        return f"Flow({self.name!r}, nodes={[node.name for node in self._nodes]})"


def _with_prompt(node: NodeSpec, prompt: str) -> NodeSpec:
    return dataclasses.replace(node, prompt=prompt)
