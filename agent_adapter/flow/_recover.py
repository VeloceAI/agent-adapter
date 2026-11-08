"""`FlowRunner.recover` and durable `resume`: claim, identity, and the approval spend."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalAlreadyDecided,
    CheckpointConflict,
    ToolUsageError,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._approve import approval_store, same_decision
from agent_adapter.exec._durable_seal import check_identity
from agent_adapter.exec.checkpoints import CheckpointStatus
from agent_adapter.exec.executor import Executor
from agent_adapter.flow._durable import (
    DurableRun,
    check_calls,
    ended,
    load_recovery,
    not_durable,
    paused_result,
    recovered_event,
)
from agent_adapter.flow._pause import Cursor, cursor_from
from agent_adapter.flow._result import FlowCheckpoint, FlowResult
from agent_adapter.flow._resume import spend_once
from agent_adapter.flow.spec import Flow
from agent_adapter.tracing.events import EventType, emit, make_event

if TYPE_CHECKING:
    from agent_adapter.flow.runner import FlowRunner

logger = logging.getLogger(__name__)


def restored_ctx(
    executor: Executor, run_id: str, identity: Mapping[str, str], ctx: ToolContext | None
) -> ToolContext | None:
    """The context a recovered run continues in: the caller's (checked), or its own identity.

    Raises:
        ToolUsageError: `ctx` is for another run, user, thread or principal.
    """
    if ctx is not None and ctx.run_id not in ("", run_id):
        raise ToolUsageError(f"ctx.run_id {ctx.run_id!r} is not the recovered run {run_id!r}")
    check_identity(run_id, identity, ctx)
    if ctx is not None:
        return ctx
    return executor.context(
        run_id=run_id, user_id=identity.get("user_id", ""), thread_id=identity.get("thread_id", "")
    )


def spend_record(pending: ApprovalRequest, decision: ApprovalDecision) -> dict[str, Any]:
    """The approval spend a resume stores (signed) before it spends the decision."""
    return {
        "request": pending.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
    }


async def settle_spend(spend: Mapping[str, Any], port: ApprovalPort | None) -> None:
    """Spend a stored decision; one already spent with this same decision counts as done.

    Raises:
        ApprovalAlreadyDecided: the store holds another decision for the request.
    """
    request = ApprovalRequest.model_validate(spend["request"])
    decision = ApprovalDecision.model_validate(spend["decision"])
    try:
        await spend_once(request, decision, port)
    except ApprovalAlreadyDecided:
        store = approval_store(port)
        if store is not None:
            record = await store.get(request.approval_id)
            if not (record.consumed and same_decision(record.decision, decision)):
                raise
        logger.info("approval %s was spent before the crash", request.approval_id)


def spend_port(runner: FlowRunner, request: Mapping[str, Any]) -> ApprovalPort | None:
    """The port a stored spend settles in: the gate port for a node gate, else the tool port."""
    executor = runner.executor
    if request.get("kind") == "node" and runner.approval is not None:
        return runner.approval
    return executor.approval


async def recover(
    runner: FlowRunner, flow: Flow[Any], run_id: str, ctx: ToolContext | None, *, rerun: bool
) -> FlowResult[Any]:
    """The body of `FlowRunner.recover` (see there)."""
    graph = runner._checked(flow)
    store = runner.checkpoints
    if store is None:
        raise not_durable(graph.name)
    found = await load_recovery(store, runner._sealer, graph, run_id)
    bound = restored_ctx(runner.executor, run_id, found.envelope.identity, ctx)
    status = found.stored.checkpoint.status
    if status == "paused":
        return paused_result(found.checkpoint)
    if status != "running":
        raise ended(run_id, status)
    rerun_count = check_calls(run_id, graph, runner.executor.registry, found, rerun=rerun)
    owned = found.checkpoint._owned if ctx is None else not ctx.run_id
    run = runner._start(
        graph, bound, run_id=run_id, executor=runner.executor, owned=owned, step=found.stored.step
    )
    durable = cast(DurableRun, run.durable)
    cursor = cursor_from(found.checkpoint)
    spend = found.envelope.spend
    # The claim: a new segment, so a worker cut off mid-step cannot log (or start) a call.
    await durable.save(cursor, "running", found.checkpoint, carried=found.calls, spend=spend)
    if spend is not None:
        await settle_spend(spend, spend_port(runner, spend["request"]))
    if run.owned:
        emit(run.ctx.sink, make_event(EventType.RUN_STARTED, run.ctx))
    recovered_event(durable, found, rerun_count)
    return await runner._drive(run, cursor)


async def inspect(
    runner: FlowRunner, flow: Flow[Any], run_id: str, ctx: ToolContext | None
) -> tuple[CheckpointStatus, FlowCheckpoint]:
    """The body of `FlowRunner.inspect` (see there)."""
    graph = runner._checked(flow)
    if runner.checkpoints is None:
        raise not_durable(graph.name)
    found = await load_recovery(runner.checkpoints, runner._sealer, graph, run_id)
    restored_ctx(runner.executor, run_id, found.envelope.identity, ctx)
    return found.stored.checkpoint.status, found.checkpoint


async def recover_all(
    runner: FlowRunner, flow: Flow[Any], *, rerun: bool, limit: int
) -> dict[str, FlowResult[Any] | AdapterError]:
    """The body of `FlowRunner.recover_all` (see there)."""
    graph = runner._checked(flow)
    if runner.checkpoints is None:
        raise not_durable(graph.name)
    out: dict[str, FlowResult[Any] | AdapterError] = {}
    for saved in await runner.checkpoints.list_unfinished(name=graph.name, limit=limit):
        try:
            out[saved.run_id] = await recover(runner, flow, saved.run_id, None, rerun=rerun)
        except CheckpointConflict as exc:
            logger.info("flow %s run %s: taken by another worker", graph.name, saved.run_id)
            out[saved.run_id] = exc
        except AdapterError as exc:
            logger.warning("flow %s run %s not recovered (%s)", graph.name, saved.run_id, exc.kind)
            out[saved.run_id] = exc
    return out


def then_hook(
    first: Callable[[Cursor], Awaitable[None]], hook: Callable[[int], Awaitable[None]]
) -> Callable[[Cursor], Awaitable[None]]:
    """`first`, then `hook(steps)` when the run goes on (a durable engine ends its step there)."""

    async def both(cursor: Cursor) -> None:
        await first(cursor)
        if cursor.frontier:
            await hook(cursor.steps)

    return both
