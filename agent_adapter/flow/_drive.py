"""Driving a started flow run: supersteps to an end, its start and finish events, content."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import CheckpointConflict, CheckpointStoreCorrupt
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.exec._approve import approval_store
from agent_adapter.exec._sink_hub import offer_flow_graph
from agent_adapter.flow import _recover
from agent_adapter.flow._durable import DurableRun
from agent_adapter.flow._loop import Steps, paused_event, run_steps
from agent_adapter.flow._nodes import NodeEnv
from agent_adapter.flow._pause import Cursor, RunInfo, denied, paused
from agent_adapter.flow._result import FlowResult, frozen
from agent_adapter.flow.diagram import cached_flow_graph
from agent_adapter.flow.graph import FlowGraph
from agent_adapter.hooks.builtin._content import safe_value
from agent_adapter.tracing._content import emit_with_content
from agent_adapter.tracing.events import EventType, emit, make_event

if TYPE_CHECKING:
    from agent_adapter.flow.runner import FlowRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    """One run() / resume() call: graph, context, node env, port and checkpoints."""

    graph: FlowGraph
    ctx: ToolContext
    env: NodeEnv
    port: ApprovalPort | None
    owned: bool
    digest: str  # the graph's shape, sealed into its checkpoints
    durable: DurableRun | None = None  # set when the runner has a checkpoint store

    @property
    def info(self) -> RunInfo:
        return RunInfo(
            self.graph.name, self.owned, self.digest, self.ctx, approval_store(self.port)
        )

    def steps(self, runner: FlowRunner) -> Steps:
        caps = (runner.max_steps, runner.concurrency)
        on_step: Callable[[Cursor], Awaitable[None]] | None = None
        if self.durable is not None:
            on_step = self.durable.commit
        hook = runner.on_superstep
        if on_step is not None and hook is not None:
            on_step = _recover.then_hook(on_step, hook)
        return Steps(self.graph, self.ctx, self.env, self.port, *caps, on_step=on_step)


async def drive(runner: FlowRunner, run: Run, cursor: Cursor) -> FlowResult[Any]:
    """Run a started flow to its end, pause or failure, with its start and finish events."""
    # Only graph-taking sinks (the run viewer) get the graph; events stay small.
    offer_flow_graph(
        run.ctx.sink,
        run.ctx.run_id,
        run.graph.name,
        lambda: cached_flow_graph(run.graph, run.digest),
    )
    traced: dict[str, Any] = {"trace_state": True} if run.env.trace_state else {}
    start = make_event(EventType.FLOW_STARTED, run.ctx, flow=run.graph.name, **traced)
    emit_with_content(run.ctx.sink, start, lambda: {"input": safe_value(cursor.state)})
    try:
        result = await _result(runner, run, cursor)
    except asyncio.CancelledError:
        # Ends the flow (and a run it owns) for every sink; nodes cut short emit nothing.
        _flow_finished(run, "cancelled", cursor.steps)
        if run.owned:
            _end_run(run, ok=False)
        logger.info("flow %s cancelled after %d steps", run.graph.name, cursor.steps)
        raise
    except Exception as exc:
        if is_control_flow(exc):
            raise
        if run.durable is not None and not _store_trouble(exc):
            await run.durable.failed(cursor)
        _flow_finished(run, "failed", cursor.steps)
        if run.owned:
            _end_run(run, ok=False)
        raise
    _flow_finished(run, result.status, result.steps, result.state)
    if run.owned and result.status != "paused":
        _end_run(run, ok=result.status == "done")
    logger.debug("flow %s %s after %d steps", run.graph.name, result.status, result.steps)
    return result


async def _result(runner: FlowRunner, run: Run, cursor: Cursor) -> FlowResult[Any]:
    stop = await run_steps(run.steps(runner), cursor)
    if stop is None:
        result: FlowResult[Any] = FlowResult(
            state=cursor.state, status="done", steps=cursor.steps, visits=frozen(cursor.visits)
        )
    elif stop.denied is not None:
        result = denied(cursor, stop.denied)
    else:
        result = await paused(
            runner._sealer, run.info, cursor, stop.gate, stop.decisions, stop.outcomes
        )
    if run.durable is not None:
        await run.durable.finish(cursor, result)
    if stop is not None and stop.gate is not None:
        paused_event(run.ctx, run.graph.name, stop.gate)
    return result


def _flow_finished(run: Run, status: str, steps: int, state: Any = None) -> None:
    """`flow.finished`; a content sink also gets the final state of a finished (done) flow."""
    event = make_event(
        EventType.FLOW_FINISHED, run.ctx, flow=run.graph.name, status=status, steps=steps
    )
    done = status == "done" and state is not None
    emit_with_content(run.ctx.sink, event, lambda: {"output": safe_value(state)} if done else {})


def _store_trouble(exc: BaseException) -> bool:
    """A failure caused by the checkpoint store (fenced or corrupt): leave the run as stored."""
    seen: set[int] = set()
    found: BaseException | None = exc
    while found is not None and id(found) not in seen:
        if isinstance(found, CheckpointConflict | CheckpointStoreCorrupt):
            return True
        seen.add(id(found))
        found = found.__cause__ or found.__context__
    return False


def _end_run(run: Run, *, ok: bool) -> None:
    emit(run.ctx.sink, make_event(EventType.RUN_FINISHED, run.ctx, ok=ok))
    run.env.executor.end_run(run.ctx.run_id)
