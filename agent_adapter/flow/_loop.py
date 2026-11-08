"""The superstep loop the runner and sub-flows share: gates, node visits, merges and routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import FlowLoopExceeded
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._resume import ResumeAnswer, _make_answer
from agent_adapter.flow._gates import check_gates
from agent_adapter.flow._nodes import NodeEnv, NodeOutcome, run_node
from agent_adapter.flow._pause import Cursor
from agent_adapter.flow._step import (
    apply_edits,
    merge_outcomes,
    next_frontier,
    route_state,
    route_targets,
)
from agent_adapter.flow._withdraw import SUPERSEDED, withdraw
from agent_adapter.flow.graph import FlowGraph, NodeSpec
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Steps:
    """What one graph's loop needs: the graph, its context and node env, the gate port, caps."""

    graph: FlowGraph
    ctx: ToolContext
    env: NodeEnv
    port: ApprovalPort | None
    max_steps: int
    concurrency: int | None
    on_step: Callable[[Cursor], Awaitable[None]] | None = None  # a durable run's checkpoint


@dataclass(frozen=True)
class Stop:
    """Why the loop stopped early: a gate or a node waits, or a gate was denied."""

    gate: ApprovalRequest | None = None
    decisions: Mapping[str, ApprovalDecision] = field(default_factory=dict)
    outcomes: tuple[NodeOutcome, ...] = ()  # the paused superstep's outcomes, waiting included
    denied: str | None = None  # the denied gate's node, or `node/child path` below a SubFlow


async def run_steps(steps: Steps, cursor: Cursor) -> Stop | None:
    """Run supersteps from `cursor` until the frontier is empty (None) or the run must stop.

    Raises:
        FlowLoopExceeded: `max_steps`, or a route past `max_loops` without `on_exhausted`.
        FlowError: a node, route or merge failed.
    """
    graph = steps.graph
    while cursor.frontier:
        if cursor.steps >= steps.max_steps:
            logger.warning("flow %s stopped after max_steps=%d", graph.name, steps.max_steps)
            raise FlowLoopExceeded(graph.name, cursor.frontier[0], steps.max_steps)
        nodes = graph.nodes
        ready = [n for n in cursor.frontier if nodes[n].kind != "join"] or list(cursor.frontier)
        gates = await check_gates(
            flow=graph.name,
            nodes=[nodes[name] for name in ready],
            state=cursor.state,
            ctx=steps.ctx,
            visits=cursor.visits,
            port=steps.port,
            granted=cursor.granted,
            encoder=steps.env.encoder,
            scope=steps.env.scope,
        )
        if gates.pending is not None:
            return Stop(gate=gates.pending, decisions=gates.decisions)
        if gates.denied is not None:
            return Stop(denied=gates.denied)
        state = apply_edits(graph, cursor.state, ready, gates.decisions)
        visits = dict(cursor.visits)
        for name in ready:
            visits[name] = visits.get(name, 0) + 1
        to_run = [nodes[name] for name in ready if name not in cursor.writes]
        ran = await _superstep(steps, to_run, state, visits, gates.decisions, cursor)
        outcomes = {**cursor.writes, **{outcome.node: outcome for outcome in ran}}
        refused = sorted(o.denied for o in ran if o.denied is not None)
        if refused:
            await _drop_waiting(steps, ran)
            return Stop(denied=refused[0])
        if any(o.pending is not None for o in ran):
            return Stop(decisions=gates.decisions, outcomes=tuple(outcomes.values()))
        frontier, loops = await _next(graph, cursor, ready, state, outcomes)
        merged = merge_outcomes(graph, state, list(outcomes.values()), cursor.steps + 1)
        cursor.state, cursor.frontier, cursor.visits = merged, frontier, visits
        cursor.loops, cursor.steps = loops, cursor.steps + 1
        cursor.granted, cursor.writes, cursor.answers = {}, {}, {}
        cursor.items, cursor.journal, cursor.children = {}, {}, {}
        if steps.on_step is not None:
            await steps.on_step(cursor)
    return None


async def _superstep(
    steps: Steps,
    specs: Sequence[NodeSpec],
    state: BaseModel,
    visits: Mapping[str, int],
    decisions: Mapping[str, ApprovalDecision],
    cursor: Cursor,
) -> list[NodeOutcome]:
    def work(spec: NodeSpec) -> Any:
        return lambda: run_node(
            steps.env,
            spec,
            state,
            steps.ctx,
            visit=visits[spec.name],
            approval=decisions.get(spec.name),
            answer=_answer_for(cursor, spec.name),
            items=cursor.items.get(spec.name),
            journal=cursor.journal.get(spec.name),
            child=cursor.children.get(spec.name),
        )

    if len(specs) <= 1:
        return [await work(spec)() for spec in specs]
    return await steps.env.executor.parallel(
        *[work(spec) for spec in specs], concurrency=steps.concurrency or len(specs)
    )


async def _drop_waiting(steps: Steps, ran: Sequence[NodeOutcome]) -> None:
    """A denied superstep never resumes: withdraw what its other nodes left pending."""
    waiting = [made for outcome in ran for made in outcome.pendings]
    if waiting:
        await withdraw(steps.graph.name, steps.ctx, waiting, SUPERSEDED)


async def _next(
    graph: FlowGraph,
    cursor: Cursor,
    ready: Sequence[str],
    state: BaseModel,
    outcomes: Mapping[str, NodeOutcome],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """The next frontier (declaration order) and the loop counters after this superstep."""
    loops = dict(cursor.loops)
    routed: dict[str, list[str]] = {}
    for name in ready:
        route = graph.routes.get(name)
        if route is not None:
            seen = route_state(graph, state, outcomes[name])
            routed[name] = await route_targets(graph.name, route, seen, cursor.visits, loops)
    return next_frontier(graph, cursor.frontier, ready, routed), loops


def paused_event(ctx: ToolContext, flow: str, pending: ApprovalRequest) -> None:
    """`node.paused` for a gate that waits for its decision."""
    node = str(pending.metadata.get("node", ""))
    event = make_event(
        EventType.NODE_PAUSED,
        ctx,
        call_id=pending.call_id,
        flow=flow,
        node=node,
        approval_id=pending.approval_id,
    )
    emit(ctx.sink, event)
    logger.info("flow %s node %s waits for approval %s", flow, node, pending.approval_id)


def _answer_for(cursor: Cursor, node: str) -> ResumeAnswer | None:
    entries = cursor.answers.get(node)
    return _make_answer(entries) if entries else None
