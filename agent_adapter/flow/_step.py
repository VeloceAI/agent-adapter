"""One superstep's bookkeeping: gate edits, merging node updates, and route targets."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowError, FlowLoopExceeded, log_failure
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.flow._invoke import describe_error
from agent_adapter.flow._nodes import NodeOutcome
from agent_adapter.flow.graph import FlowGraph, RouteSpec
from agent_adapter.flow.state import apply_updates

logger = logging.getLogger(__name__)


def apply_edits(
    graph: FlowGraph,
    state: BaseModel,
    ready: Sequence[str],
    decisions: Mapping[str, ApprovalDecision],
) -> BaseModel:
    """Apply each granted gate's `edited_args` as a state update, in node-name order."""
    for name in sorted(ready):
        decision = decisions.get(name)
        if decision is None or decision.edited_args is None:
            continue
        try:
            state = apply_updates(state, [decision.edited_args], graph.rules)
        except FlowError as exc:
            raise FlowError(
                f"flow {graph.name!r} node {name!r}: edited state is invalid: {exc}",
                flow=graph.name,
                node=name,
            ) from exc
    return state


def merge_outcomes(
    graph: FlowGraph, state: BaseModel, outcomes: Sequence[NodeOutcome], step: int
) -> BaseModel:
    """Merge every node's updates: node-name order (LangGraph task order), then Map item order."""
    ordered = sorted(outcomes, key=lambda outcome: outcome.node)
    updates = [update for outcome in ordered for update in outcome.updates]
    try:
        return apply_updates(state, updates, graph.rules)
    except FlowError as exc:
        nodes = [outcome.node for outcome in ordered if outcome.updates]
        log_failure(
            logger, logging.WARNING, exc, "flow %s superstep %d: merge failed", graph.name, step
        )
        raise FlowError(
            f"flow {graph.name!r} superstep {step} (nodes {nodes}): {exc}", flow=graph.name
        ) from exc


def route_state(graph: FlowGraph, state: BaseModel, outcome: NodeOutcome) -> BaseModel:
    """What a route sees: the superstep's input state plus its own node's updates only."""
    try:
        return apply_updates(state, outcome.updates, graph.rules)
    except FlowError as exc:
        raise FlowError(
            f"flow {graph.name!r} route from {outcome.node!r}: {exc}",
            flow=graph.name,
            node=outcome.node,
        ) from exc


def next_frontier(
    graph: FlowGraph,
    frontier: Sequence[str],
    ready: Sequence[str],
    routed: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """The next frontier in declaration order: unrun nodes, `ready`'s edges and route picks."""
    following = [name for name in frontier if name not in ready]
    for name in ready:
        following.extend(graph.edges[name])
        following.extend(routed.get(name, ()))
    chosen = set(following)
    return tuple(name for name in graph.nodes if name in chosen)


async def route_targets(
    flow: str,
    route: RouteSpec,
    state: BaseModel,
    visits_before: Mapping[str, int],
    loops: dict[str, int],
) -> list[str]:
    """The nodes a route picks; picking the source or an earlier-visited node counts a loop."""
    labels = await call_route(flow, route, state)
    return pick_targets(flow, route, labels, visits_before, loops)


async def call_route(flow: str, route: RouteSpec, state: BaseModel) -> list[Any]:
    """The labels `route.fn(state)` returns (sync or async), as a list.

    Raises:
        FlowError: the route function failed.
    """
    try:
        labels = route.fn(state)
        if inspect.isawaitable(labels):
            labels = await labels
    except Exception as exc:
        if is_control_flow(exc):
            raise
        raise FlowError(
            f"flow {flow!r} route from {route.source!r} failed: {describe_error(exc)}",
            flow=flow,
            node=route.source,
        ) from exc
    return [labels] if isinstance(labels, str) else list(labels)


def pick_targets(
    flow: str,
    route: RouteSpec,
    labels: Sequence[Any],
    visits_before: Mapping[str, int],
    loops: dict[str, int],
) -> list[str]:
    """The nodes `labels` name; picking the source or an earlier-visited node counts a loop.

    Raises:
        FlowError: a label the route does not know.
        FlowLoopExceeded: `max_loops` passed and no `on_exhausted`.
    """
    source = route.source
    targets: list[str] = []
    for label in labels:
        if not isinstance(label, str) or label not in route.targets:
            raise FlowError(
                f"flow {flow!r} route from {source!r} returned unknown label {label!r}",
                flow=flow,
                node=source,
            )
        target = route.targets[label]
        if target == source or visits_before.get(target, 0) > 0:
            loops[source] = loops.get(source, 0) + 1
            if route.max_loops is not None and loops[source] > route.max_loops:
                if route.on_exhausted is None:
                    raise FlowLoopExceeded(flow, source, route.max_loops)
                logger.info(
                    "flow %s route from %s exhausted %d loops; going to %s",
                    flow,
                    source,
                    route.max_loops,
                    route.on_exhausted,
                )
                target = route.on_exhausted
        targets.append(target)
    return targets
