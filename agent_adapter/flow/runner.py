"""The local flow runner: LangGraph-style supersteps on the executor, with pause and resume."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowLoopExceeded, ToolUsageError
from agent_adapter.core.ports import ApprovalPort, EventSink, LLMPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._approve import approval_store
from agent_adapter.exec._resume import ResumeAnswer, _make_answer
from agent_adapter.exec._sink_hub import merge_sinks, offer_flow_graph
from agent_adapter.exec.executor import Executor
from agent_adapter.flow._digest import graph_digest
from agent_adapter.flow._gates import (
    check_gates,
    check_resume,
)
from agent_adapter.flow._nodes import NodeEnv, NodeOutcome, run_node
from agent_adapter.flow._pause import Cursor, RunInfo, denied, paused
from agent_adapter.flow._result import (
    FlowCheckpoint,
    FlowResult,
    FlowStatus,
    frozen,
)
from agent_adapter.flow._resume import check_answers, resume_entry, spend_once
from agent_adapter.flow._seal import (
    Sealer,
    allowed_classes,
    check_graph,
    check_seal,
    dump_json,
    load_json,
    runner_key,
)
from agent_adapter.flow._step import (
    apply_edits,
    merge_outcomes,
    next_frontier,
    route_state,
    route_targets,
)
from agent_adapter.flow.diagram import cached_flow_graph
from agent_adapter.flow.graph import FlowGraph, NodeSpec
from agent_adapter.flow.spec import Flow
from agent_adapter.flow.state import validated_state
from agent_adapter.hooks.base import Hook
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=BaseModel)


@dataclass(frozen=True)
class _Run:
    """One run() / resume() call."""

    graph: FlowGraph
    ctx: ToolContext
    env: NodeEnv
    port: ApprovalPort | None
    owned: bool
    digest: str  # the graph's shape, sealed into its checkpoints

    @property
    def info(self) -> RunInfo:
        return RunInfo(
            self.graph.name, self.owned, self.digest, self.ctx, approval_store(self.port)
        )


class FlowRunner:
    """Runs a `Flow` locally: nodes on the executor, supersteps, merges, routes and pauses."""

    def __init__(
        self,
        executor: Executor,
        *,
        approval: ApprovalPort | None = None,
        hooks: Sequence[Hook] = (),
        sink: EventSink | None = None,
        max_steps: int = 100,
        concurrency: int | None = None,
        model: LLMPort | None = None,
        checkpoint_key: bytes | None = None,
        seal_encoder: Callable[[object], object] | None = None,
        seal_decoder: Callable[[str, object], object] | None = None,
        trace_state: bool = False,
    ) -> None:
        """`trace_state=True` puts each node's updated values on `node.finished` (`updates`).

        Raises:
            ToolUsageError: a bad `max_steps`, `concurrency`, codec, hook or `trace_state`.
        """
        if not isinstance(trace_state, bool):
            raise ToolUsageError(f"trace_state must be a bool, got {trace_state!r}")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ToolUsageError(f"max_steps must be an int >= 1, got {max_steps!r}")
        if concurrency is not None and (
            not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1
        ):
            raise ToolUsageError(f"concurrency must be an int >= 1 or None, got {concurrency!r}")
        for label, fn in (("seal_encoder", seal_encoder), ("seal_decoder", seal_decoder)):
            if fn is not None and not callable(fn):
                raise ToolUsageError(f"{label} must be callable or None, got {fn!r}")
        bad_hooks = [hook for hook in hooks if not isinstance(hook, Hook)]
        if bad_hooks:
            raise ToolUsageError(f"hooks must be Hook instances, got {bad_hooks!r}")
        self.executor = executor
        self.approval = approval
        self.hooks = tuple(hooks)
        self.sink = sink
        self.max_steps = max_steps
        self.concurrency = concurrency
        self.model = model
        self.trace_state = trace_state
        # signs checkpoints; the default per-process key resumes only in this process
        self._sealer = Sealer(runner_key(checkpoint_key), seal_encoder, seal_decoder)

    async def run(
        self, flow: Flow[S], state: S | Mapping[str, Any], ctx: ToolContext | None = None
    ) -> FlowResult[S]:
        """Run `flow` from its start node until it is done, paused or denied.

        Raises:
            FlowDefinitionError: the flow is invalid.
            FlowError: a node, route or merge failed (`node` names the node when known).
            FlowLoopExceeded: a route without `on_exhausted` passed `max_loops`, or `max_steps`.
            FlowError: `state-not-sealable`: the run would pause but some part has no
                canonical form; the pending approval records are withdrawn first.
        """
        graph = flow.validate()
        cursor = Cursor(state=self._initial(graph, state), frontier=(graph.start,))
        owned = ctx is None or not ctx.run_id
        run = self._start(graph, ctx, run_id="", executor=self.executor, owned=owned)
        if run.owned:
            emit(run.ctx.sink, make_event(EventType.RUN_STARTED, run.ctx))
        return cast(FlowResult[S], await self._drive(run, cursor))

    async def resume(
        self,
        flow: Flow[S],
        checkpoint: FlowCheckpoint,
        decision: ApprovalDecision,
        ctx: ToolContext | None = None,
    ) -> FlowResult[S]:
        """Continue a paused run; `decision` answers `checkpoint.pending` (the port is not asked).

        Only the paused node(s) of the paused superstep run again. A denied tool decision
        raises `ApprovalDenied` inside the node (a `FlowError`, or a skipped `Map` item).

        Raises:
            FlowError: the checkpoint is from another flow or graph, has nothing pending, holds
                another state model, or is not sealed by this runner's key
                (`checkpoint-tampered`); as `run`.
            ApprovalAlreadyDecided: the checkpoint was already resumed.
        """
        graph = flow.validate()
        pending = check_resume(graph, checkpoint, decision)
        check_graph(graph, checkpoint)
        check_seal(self._sealer, checkpoint)
        tool_store = approval_store(self.executor.approval)
        if tool_store is not None:
            await check_answers(tool_store, checkpoint)
        node = str(pending.metadata.get("node", ""))
        granted = dict(checkpoint.decisions)
        answers = dict(checkpoint._answers)
        if pending.kind == "node":
            granted[node] = decision
            port = self.approval if self.approval is not None else self.executor.approval
        else:
            answers[node] = (*answers.get(node, ()), resume_entry(pending, decision))
            port = self.executor.approval
        await spend_once(pending, decision, port)
        owned = checkpoint._owned if ctx is None else not ctx.run_id
        run = self._start(graph, ctx, run_id=pending.run_id, executor=self.executor, owned=owned)
        cursor = Cursor(
            state=checkpoint.state,
            frontier=checkpoint.frontier,
            visits=dict(checkpoint.visits),
            loops=dict(checkpoint.loops),
            steps=checkpoint.steps,
            granted=granted,
            writes={n: NodeOutcome(n, updates=u) for n, u in checkpoint._writes.items()},
            answers=answers,
            items=dict(checkpoint._items),
            journal=dict(checkpoint._journal),
        )
        return cast(FlowResult[S], await self._drive(run, cursor))

    def dump_checkpoint(self, checkpoint: FlowCheckpoint) -> str:
        """The checkpoint as sealed JSON text, for storage outside this process.

        Raises:
            FlowError: the checkpoint is not sealed by this runner (`checkpoint-tampered`).
        """
        return dump_json(self._sealer, checkpoint)

    def load_checkpoint(self, flow: Flow[Any], data: str | bytes) -> FlowCheckpoint:
        """A checkpoint from `dump_checkpoint` text; its MAC is checked before anything is built.

        Raises:
            FlowError: not a checkpoint, a wrong MAC (`checkpoint-tampered`), another graph,
                or a value that cannot be loaded (e.g. a custom type and no `seal_decoder`).
        """
        return load_json(self._sealer, flow.validate(), data)

    def _initial(self, graph: FlowGraph, state: Any) -> BaseModel:
        model = graph.state_model
        if isinstance(state, model):
            return state
        if isinstance(state, Mapping):
            return validated_state(model, state, what=f"initial state of flow {graph.name!r}")
        raise ToolUsageError(
            f"flow {graph.name!r}: state must be a {model.__name__} or a mapping, "
            f"got {type(state).__name__}"
        )

    def _start(
        self,
        graph: FlowGraph,
        ctx: ToolContext | None,
        *,
        run_id: str,
        executor: Executor,
        owned: bool,
    ) -> _Run:
        if ctx is None:
            ctx = executor.context(run_id=run_id or None)
        elif not ctx.run_id:
            ctx = dataclasses.replace(ctx, run_id=run_id or new_id("run_"))
        chosen = first_sink(ctx.sink, executor.sink)
        # The runner's own sink adds to the run's sink, never hides it, never doubles it.
        sink = merge_sinks(chosen, self.sink)
        changes: dict[str, Any] = {"sink": sink}
        if ctx.ledger is None:
            changes["ledger"] = executor.ledger
        if ctx.cache is None:
            changes["cache"] = executor.cache
        ctx = dataclasses.replace(ctx, **changes)
        hooks = tuple({id(hook): hook for hook in (*executor.hooks, *self.hooks)}.values())
        sealer = self._sealer
        env = NodeEnv(
            graph.name,
            executor,
            hooks,
            self.model,
            sealer.encoder,
            sealer.decoder,
            allowed_classes(graph.state_model),
            self.trace_state,
        )
        port = self.approval if self.approval is not None else executor.approval
        return _Run(graph, ctx, env, port, owned, graph_digest(graph))

    async def _drive(self, run: _Run, cursor: Cursor) -> FlowResult[Any]:
        # Only graph-taking sinks (the run viewer) get the graph; events stay small.
        offer_flow_graph(
            run.ctx.sink,
            run.ctx.run_id,
            run.graph.name,
            lambda: cached_flow_graph(run.graph, run.digest),
        )
        traced: dict[str, Any] = {"trace_state": True} if run.env.trace_state else {}
        emit(
            run.ctx.sink,
            make_event(EventType.FLOW_STARTED, run.ctx, flow=run.graph.name, **traced),
        )
        try:
            result = await self._loop(run, cursor)
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
            _flow_finished(run, "failed", cursor.steps)
            if run.owned:
                _end_run(run, ok=False)
            raise
        _flow_finished(run, result.status, result.steps)
        if run.owned and result.status != "paused":
            _end_run(run, ok=result.status == "done")
        logger.debug("flow %s %s after %d steps", run.graph.name, result.status, result.steps)
        return result

    async def _loop(self, run: _Run, cursor: Cursor) -> FlowResult[Any]:
        graph = run.graph
        while cursor.frontier:
            if cursor.steps >= self.max_steps:
                logger.warning("flow %s stopped after max_steps=%d", graph.name, self.max_steps)
                raise FlowLoopExceeded(graph.name, cursor.frontier[0], self.max_steps)
            nodes = graph.nodes
            ready = [n for n in cursor.frontier if nodes[n].kind != "join"] or list(cursor.frontier)
            gates = await check_gates(
                flow=graph.name,
                nodes=[nodes[name] for name in ready],
                state=cursor.state,
                ctx=run.ctx,
                visits=cursor.visits,
                port=run.port,
                granted=cursor.granted,
                encoder=self._sealer.encoder,
            )
            if gates.pending is not None:
                info = run.info
                result = await paused(
                    self._sealer, info, cursor, gates.pending, gates.decisions, []
                )
                _paused_event(run, gates.pending)
                return result
            if gates.denied is not None:
                return denied(cursor, gates.denied)
            state = apply_edits(graph, cursor.state, ready, gates.decisions)
            visits = dict(cursor.visits)
            for name in ready:
                visits[name] = visits.get(name, 0) + 1
            to_run = [nodes[name] for name in ready if name not in cursor.writes]
            ran = await self._superstep(run, to_run, state, visits, gates.decisions, cursor)
            outcomes = {**cursor.writes, **{outcome.node: outcome for outcome in ran}}
            if any(o.pending is not None for o in ran):
                waiting = list(outcomes.values())
                return await paused(self._sealer, run.info, cursor, None, gates.decisions, waiting)
            frontier, loops = await _next(run, cursor, ready, state, outcomes)
            merged = merge_outcomes(graph, state, list(outcomes.values()), cursor.steps + 1)
            cursor.state, cursor.frontier, cursor.visits = merged, frontier, visits
            cursor.loops, cursor.steps = loops, cursor.steps + 1
            cursor.granted, cursor.writes, cursor.answers = {}, {}, {}
            cursor.items, cursor.journal = {}, {}
        return FlowResult(
            state=cursor.state, status="done", steps=cursor.steps, visits=frozen(cursor.visits)
        )

    async def _superstep(
        self,
        run: _Run,
        specs: Sequence[NodeSpec],
        state: BaseModel,
        visits: Mapping[str, int],
        decisions: Mapping[str, ApprovalDecision],
        cursor: Cursor,
    ) -> list[NodeOutcome]:
        def work(spec: NodeSpec) -> Any:
            return lambda: run_node(
                run.env,
                spec,
                state,
                run.ctx,
                visit=visits[spec.name],
                approval=decisions.get(spec.name),
                answer=_answer_for(cursor, spec.name),
                items=cursor.items.get(spec.name),
                journal=cursor.journal.get(spec.name),
            )

        if len(specs) <= 1:
            return [await work(spec)() for spec in specs]
        return await run.env.executor.parallel(
            *[work(spec) for spec in specs], concurrency=self.concurrency or len(specs)
        )


async def _next(
    run: _Run,
    cursor: Cursor,
    ready: Sequence[str],
    state: BaseModel,
    outcomes: Mapping[str, NodeOutcome],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """The next frontier (declaration order) and the loop counters after this superstep."""
    graph = run.graph
    loops = dict(cursor.loops)
    routed: dict[str, list[str]] = {}
    for name in ready:
        route = graph.routes.get(name)
        if route is not None:
            seen = route_state(graph, state, outcomes[name])
            routed[name] = await route_targets(graph.name, route, seen, cursor.visits, loops)
    return next_frontier(graph, cursor.frontier, ready, routed), loops


def _flow_finished(run: _Run, status: str, steps: int) -> None:
    event = make_event(
        EventType.FLOW_FINISHED, run.ctx, flow=run.graph.name, status=status, steps=steps
    )
    emit(run.ctx.sink, event)


def _end_run(run: _Run, *, ok: bool) -> None:
    emit(run.ctx.sink, make_event(EventType.RUN_FINISHED, run.ctx, ok=ok))
    run.env.executor.end_run(run.ctx.run_id)


def _paused_event(run: _Run, pending: ApprovalRequest) -> None:
    node = str(pending.metadata.get("node", ""))
    event = make_event(
        EventType.NODE_PAUSED,
        run.ctx,
        call_id=pending.call_id,
        flow=run.graph.name,
        node=node,
        approval_id=pending.approval_id,
    )
    emit(run.ctx.sink, event)
    logger.info("flow %s node %s waits for approval %s", run.graph.name, node, pending.approval_id)


def _answer_for(cursor: Cursor, node: str) -> ResumeAnswer | None:
    entries = cursor.answers.get(node)
    return _make_answer(entries) if entries else None


__all__ = ["FlowCheckpoint", "FlowResult", "FlowRunner", "FlowStatus"]
