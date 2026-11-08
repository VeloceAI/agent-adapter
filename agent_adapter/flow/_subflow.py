"""Run a `SubFlow` node: map the parent state in, run the child graph's supersteps, map back."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowError
from agent_adapter.exec._approve import approval_store
from agent_adapter.flow._digest import graph_digest
from agent_adapter.flow._invoke import describe_error
from agent_adapter.flow._loop import Steps, paused_event, run_steps
from agent_adapter.flow._nested import SubFlowDenied, SubflowHost, SubFlowPaused
from agent_adapter.flow._nodes import NodeEnv
from agent_adapter.flow._pause import Cursor, RunInfo, child_paused, cursor_from
from agent_adapter.flow._result import FlowCheckpoint
from agent_adapter.flow._seal import allowed_classes
from agent_adapter.flow._subcheck import input_pairs, output_pairs
from agent_adapter.flow.actions import SubFlow
from agent_adapter.flow.graph import FlowGraph, NodeSpec
from agent_adapter.flow.state import merge_rules, validated_state
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


async def run_subflow(
    env: NodeEnv,
    node: NodeSpec,
    state: BaseModel,
    ctx: ToolContext,
    checkpoint: FlowCheckpoint | None,
) -> Mapping[str, Any]:
    """The parent update from one SubFlow visit; a paused child continues from `checkpoint`.

    Raises:
        SubFlowPaused: the child paused (a gate or an approval inside it).
        SubFlowDenied: a gate inside the child was denied.
        FlowError: the mapping failed, or the child failed (as a flow run would).
    """
    host, child = env.subflows, node.child
    if host is None or child is None or not isinstance(node.action, SubFlow):
        raise FlowError(f"flow {env.flow!r} node {node.name!r}: not a runnable SubFlow node")
    action, at = node.action, _At(env.flow, node.name)
    initial = _child_input(action, state, child, at)
    cursor = cursor_from(checkpoint) if checkpoint is not None else Cursor(initial, (child.start,))
    child_env = dataclasses.replace(
        env, flow=child.name, allowed=allowed_classes(child.state_model), scope=f"{ctx.call_id}/"
    )
    steps = Steps(child, ctx, child_env, host.port, host.max_steps, host.concurrency)
    event_id = f"{ctx.call_id}/{child.name}"
    origin = {"parent_flow": env.flow, "parent_node": node.name}
    emit(
        ctx.sink,
        make_event(EventType.FLOW_STARTED, ctx, call_id=event_id, flow=child.name, **origin),
    )
    status = "failed"
    try:
        stop = await run_steps(steps, cursor)
        if stop is None:
            status = "done"
            return _child_output(action, initial, cursor.state, type(state), at)
        if stop.denied is not None:
            status = "denied"
            logger.info("%s: sub-flow %s denied at %s", at, child.name, stop.denied)
            raise SubFlowDenied(env.flow, stop.denied)
        raise await _paused(host, child, steps, cursor, stop)
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except SubFlowPaused:
        status = "paused"
        raise
    finally:
        steps_run = cursor.steps
        event = make_event(
            EventType.FLOW_FINISHED,
            ctx,
            call_id=event_id,
            flow=child.name,
            status=status,
            steps=steps_run,
            **origin,
        )
        emit(ctx.sink, event)


async def _paused(
    host: SubflowHost, child: FlowGraph, steps: Steps, cursor: Cursor, stop: Any
) -> SubFlowPaused:
    info = RunInfo(child.name, False, graph_digest(child), steps.ctx, approval_store(host.port))
    checkpoint, kept = await child_paused(info, cursor, stop.gate, stop.decisions, stop.outcomes)
    if stop.gate is not None:
        paused_event(steps.ctx, child.name, stop.gate)
    return SubFlowPaused(checkpoint, tuple(kept))


@dataclasses.dataclass(frozen=True)
class _At:
    """The SubFlow node, for error messages."""

    flow: str
    node: str

    def __str__(self) -> str:
        return f"flow {self.flow!r} node {self.node!r}"

    def error(self, message: str) -> FlowError:
        return FlowError(f"{self}: {message}", flow=self.flow, node=self.node)


def _child_input(action: SubFlow, state: BaseModel, child: FlowGraph, at: _At) -> BaseModel:
    model = child.state_model
    if callable(action.input):
        data = _call(action.input, state, at, "input")
        if isinstance(data, model):
            return data
        if not isinstance(data, Mapping):
            raise at.error(f"SubFlow input returned {type(data).__name__}, not a mapping")
    else:
        pairs = input_pairs(action, type(state), model, merge_rules(type(state)))
        data = {target: getattr(state, source) for target, source in pairs}
    try:
        return validated_state(model, data, what="sub-flow input")
    except FlowError as exc:
        raise at.error(str(exc)) from exc


def _child_output(
    action: SubFlow, initial: BaseModel, final: BaseModel, parent: type[BaseModel], at: _At
) -> Mapping[str, Any]:
    into = action.into
    if isinstance(into, str):
        return {into: final}
    if callable(into):
        update = _call(into, final, at, "into")
        if not isinstance(update, Mapping):
            raise at.error(f"SubFlow into returned {type(update).__name__}, not a mapping")
        return dict(update)
    pairs = output_pairs(action, parent, type(final))
    return {
        target: getattr(final, source)
        for target, source in pairs
        if getattr(final, source) != getattr(initial, source)  # only what the child wrote
    }


def _call(fn: Any, state: BaseModel, at: _At, label: str) -> Any:
    try:
        return fn(state)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        raise at.error(f"SubFlow {label} failed: {describe_error(exc)}") from exc
