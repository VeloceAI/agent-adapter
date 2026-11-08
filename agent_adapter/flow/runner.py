"""The local flow runner: LangGraph-style supersteps on the executor, with pause and resume."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalAlreadyDecided,
    FlowDefinitionError,
    ToolNotFound,
    ToolUsageError,
)
from agent_adapter.core.ports import ApprovalPort, EventSink, LLMPort
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec._approve import approval_store
from agent_adapter.exec._durable import CallLog, durable_key, new_segment
from agent_adapter.exec._durable_seal import identity_of
from agent_adapter.exec._sink_hub import merge_sinks
from agent_adapter.exec.checkpoints import CheckpointStatus, CheckpointStore
from agent_adapter.exec.executor import Executor
from agent_adapter.flow import _recover
from agent_adapter.flow._digest import graph_digest
from agent_adapter.flow._drive import Run, drive
from agent_adapter.flow._durable import DurableRun, check_latest
from agent_adapter.flow._gates import check_resume
from agent_adapter.flow._nested import SubflowHost
from agent_adapter.flow._nodes import NodeEnv
from agent_adapter.flow._pause import Cursor, cursor_from
from agent_adapter.flow._recover import restored_ctx, settle_spend, spend_record
from agent_adapter.flow._result import (
    FlowCheckpoint,
    FlowResult,
    FlowStatus,
)
from agent_adapter.flow._resume import answer_child, check_answers, resume_entry, spend_once
from agent_adapter.flow._seal import (
    Sealer,
    allowed_classes,
    check_graph,
    check_seal,
    dump_json,
    load_json,
    runner_key,
)
from agent_adapter.flow._subcheck import tool_names
from agent_adapter.flow._subflow import run_subflow
from agent_adapter.flow.graph import FlowGraph
from agent_adapter.flow.spec import Flow
from agent_adapter.flow.state import validated_state
from agent_adapter.hooks.base import Hook
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=BaseModel)


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
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        """`trace_state=True` puts each node's updated values on `node.finished` (`updates`).

        With `checkpoints`, every superstep ends with a sealed checkpoint in the store and
        each tool call is logged, so `recover` continues a crashed run without re-running a
        finished call.

        Raises:
            ToolUsageError: a bad `max_steps`, `concurrency`, codec, hook, `trace_state` or
                `checkpoints`.
            ConfigError: `checkpoints` other than a `MemoryCheckpointStore` without a
                `checkpoint_key`.
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
        if checkpoints is not None:
            checkpoint_key = durable_key(checkpoints, checkpoint_key, "FlowRunner")
        self.checkpoints = checkpoints
        # awaited with the superstep count after each checkpoint a durable run goes on from
        self.on_superstep: Callable[[int], Awaitable[None]] | None = None
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
        graph = self._checked(flow)
        cursor = Cursor(state=self._initial(graph, state), frontier=(graph.start,))
        owned = ctx is None or not ctx.run_id
        run = self._start(graph, ctx, run_id="", executor=self.executor, owned=owned)
        if run.durable is not None:  # step 0: a run id the store already holds is refused
            await run.durable.save(cursor, "running")
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
        graph = self._checked(flow)
        pending = check_resume(graph, checkpoint, decision)
        check_graph(graph, checkpoint)
        check_seal(self._sealer, checkpoint)
        tool_store = approval_store(self.executor.approval)
        if tool_store is not None:
            await check_answers(tool_store, checkpoint)
        node = str(pending.metadata.get("node", ""))
        granted = dict(checkpoint.decisions)
        answers = dict(checkpoint._answers)
        children = dict(checkpoint._children)
        gate_port = self.approval if self.approval is not None else self.executor.approval
        port = gate_port if pending.kind == "node" else self.executor.approval
        if node in children:  # the pause is inside this SubFlow node's child run
            children[node] = answer_child(children[node], decision)
        elif pending.kind == "node":
            granted[node] = decision
        else:
            answers[node] = (*answers.get(node, ()), resume_entry(pending, decision))
        owned = checkpoint._owned if ctx is None else not ctx.run_id
        cursor = cursor_from(checkpoint, granted=granted, answers=answers, children=children)
        if self.checkpoints is None:
            await spend_once(pending, decision, port)
            run = self._start(
                graph, ctx, run_id=pending.run_id, executor=self.executor, owned=owned
            )
            return cast(FlowResult[S], await self._drive(run, cursor))
        key = self._sealer.key
        step, identity = await check_latest(self.checkpoints, key, checkpoint, pending.run_id)
        bound = restored_ctx(self.executor, pending.run_id, identity, ctx)
        run = self._start(
            graph, bound, run_id=pending.run_id, executor=self.executor, owned=owned, step=step
        )
        durable = cast(DurableRun, run.durable)
        # Stored first (one resume wins the step), spent second: a crash between the two
        # recovers by spending it then.
        spend = spend_record(pending, decision)
        await durable.save(cursor, "running", spend=spend)
        try:
            await settle_spend(spend, port)
        except ApprovalAlreadyDecided:
            await durable.failed(cursor)
            raise
        return cast(FlowResult[S], await self._drive(run, cursor))

    async def recover(
        self,
        flow: Flow[S],
        run_id: str,
        ctx: ToolContext | None = None,
        *,
        rerun_in_doubt: bool = False,
    ) -> FlowResult[S]:
        """Continue a crashed run from its last checkpoint; finished tool calls replay.

        A call that started but logged no result runs again only if its tool is
        `idempotent=True`, or with `rerun_in_doubt=True` after you checked it. A run that
        was paused comes back paused (resume it as usual).

        Raises:
            FlowError: no `checkpoints=` store, or the flow's graph changed since the save.
            CheckpointNotFound: nothing is stored for `run_id`.
            CheckpointConflict: another worker took the run first, it already ended, or the
                checkpoint is for another flow.
            CheckpointStoreCorrupt: the stored checkpoint or its signature does not check out.
            RecoveryNeedsDecision: a call may have run and its tool is not idempotent.
        """
        return cast(
            FlowResult[S],
            await _recover.recover(self, flow, run_id, ctx, rerun=rerun_in_doubt),
        )

    async def recover_all(
        self, flow: Flow[Any], *, rerun_in_doubt: bool = False, limit: int = 100
    ) -> dict[str, FlowResult[Any] | AdapterError]:
        """Recover every unfinished run of `flow` (a worker on startup), one at a time.

        Each run id maps to its result, or to the error that stopped its recovery (another
        worker took it, a call needs a decision, a node failed); the other runs still go on.

        Raises:
            FlowError: no `checkpoints=` store.
        """
        return await _recover.recover_all(self, flow, rerun=rerun_in_doubt, limit=limit)

    async def inspect(
        self, flow: Flow[Any], run_id: str, ctx: ToolContext | None = None
    ) -> tuple[CheckpointStatus, FlowCheckpoint]:
        """A durable run's stored status and checkpoint, checked, without saving anything.

        Raises:
            FlowError: no `checkpoints=` store, or the flow's graph changed.
            CheckpointNotFound, CheckpointConflict, CheckpointStoreCorrupt: as `recover`.
            ToolUsageError: `ctx` is for another run, user, thread or principal.
        """
        return await _recover.inspect(self, flow, run_id, ctx)

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

    def _checked(self, flow: Flow[Any]) -> FlowGraph:
        """The validated graph; cached nodes must not name a tool that needs approval.

        Raises:
            FlowDefinitionError: the flow is invalid, or a cached node's tool needs approval.
        """
        graph = flow.validate()
        registry = self.executor.registry
        problems = []
        for flow_name, node, tool in tool_names(graph) if registry is not None else ():
            try:
                confirm = registry.get(tool).confirm
            except ToolNotFound:
                continue  # the node fails when it runs, as without a cache
            if confirm:
                problems.append(
                    f"flow {flow_name!r} node {node!r}: cache cannot be used with tool {tool!r}, "
                    "which needs approval"
                )
        if problems:
            raise FlowDefinitionError(problems, flow=graph.name)
        return graph

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
        step: int = -1,
    ) -> Run:
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
        durable = None
        if self.checkpoints is not None:
            log = CallLog(self.checkpoints, ctx.run_id, new_segment(), sealer.key)
            digest = graph_digest(graph)
            durable = DurableRun(
                self.checkpoints,
                graph.name,
                sealer,
                digest,
                owned,
                ctx,
                step,
                log,
                identity_of(ctx),
            )
        env = NodeEnv(
            graph.name,
            executor,
            hooks,
            self.model,
            sealer.encoder,
            sealer.decoder,
            allowed_classes(graph.state_model),
            self.trace_state,
            subflows=SubflowHost(
                run_subflow, port_for(self, executor), self.max_steps, self.concurrency
            ),
            cache_secret=sealer.key,
            calls=durable.log if durable is not None else None,
        )
        port = port_for(self, executor)
        return Run(graph, ctx, env, port, owned, graph_digest(graph), durable)

    async def _drive(self, run: Run, cursor: Cursor) -> FlowResult[Any]:
        return await drive(self, run, cursor)


def port_for(runner: FlowRunner, executor: Executor) -> ApprovalPort | None:
    """The port pause gates ask: the runner's, else the executor's."""
    return runner.approval if runner.approval is not None else executor.approval


__all__ = ["FlowCheckpoint", "FlowResult", "FlowRunner", "FlowStatus"]
