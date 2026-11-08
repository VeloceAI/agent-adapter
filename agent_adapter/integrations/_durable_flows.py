"""Engine-neutral durable flow steps: one superstep per call, on a checkpointed FlowRunner."""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_adapter.approval import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    CheckpointConflict,
    CheckpointNotFound,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.checkpoints import CheckpointStore
from agent_adapter.flow import Flow, FlowCheckpoint, FlowResult, FlowRunner
from agent_adapter.integrations._durable_slice import SliceEnd, SliceSink
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import FanoutSink, members

logger = logging.getLogger(__name__)

StepStatus = Literal["running", "paused", "done", "denied", "failed"]
_FROZEN = ConfigDict(frozen=True, extra="forbid")
_TERMINAL = frozenset({"done", "denied", "failed"})


class PendingApproval(BaseModel):
    """What a paused run waits for; the tool arguments stay out of engine history."""

    model_config = _FROZEN

    approval_id: str
    kind: Literal["tool", "node"]
    tool: str
    node: str = ""
    hint: str = ""


class FlowStepReport(BaseModel):
    """Where a durable run stands after one step; `state` is set once it is done or denied."""

    model_config = _FROZEN

    run_id: str
    flow: str
    status: StepStatus
    steps: int = 0
    state: dict[str, Any] | None = None
    pending: PendingApproval | None = None
    denied_at: str | None = None


class ApprovalNotice(BaseModel):
    """A wake-up for a paused run: re-check the approval store for this approval's decision.

    It carries no verdict. Only a decided record in your approval store resumes the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    approval_id: str = Field(min_length=1)


TIMEOUT_DECISION = ApprovalDecision(
    approved=False, reason="approval timed out", decided_by="timeout"
)


class RunIdentity(BaseModel):
    """Who a durable run acts for; sealed into its checkpoints, so every step must match."""

    model_config = _FROZEN

    user_id: str = ""
    thread_id: str = ""
    principal: str = ""  # ctx.extras["principal"]: per-principal budgets and approvals

    @classmethod
    def of(cls, raw: object) -> RunIdentity:
        """An identity from engine input (None is the empty identity).

        Raises:
            ToolUsageError: not an identity.
        """
        if raw is None:
            return cls()
        try:
            return cls.model_validate(raw)
        except ValidationError:
            raise ToolUsageError("the run identity is not valid") from None


class DurableFlows:
    """Runs registered flows one superstep per call, for a durable engine's activities or steps.

    Every call goes through `runner`, so its executor's approvals, hooks, budgets, cache,
    ledger and events still apply. The runner's checkpoint store is the source of truth: a
    call that is retried after a crash continues the run, and finished tool calls replay.
    """

    def __init__(self, runner: FlowRunner, flows: Sequence[Flow[Any]]) -> None:
        """Wrap a checkpointed runner and the flows it may run.

        Raises:
            ToolUsageError: `runner` has no `checkpoints=` store, `flows` is empty, or two
                flows share a name.
        """
        if not isinstance(runner, FlowRunner):
            raise ToolUsageError(f"runner must be a FlowRunner, got {type(runner).__name__}")
        if runner.checkpoints is None:
            raise ToolUsageError("DurableFlows needs a FlowRunner with a checkpoints= store")
        named: dict[str, Flow[Any]] = {}
        for flow in flows:
            if not isinstance(flow, Flow):
                raise ToolUsageError(f"flows must be Flow objects, got {type(flow).__name__}")
            name = flow.validate().name
            if name in named:
                raise ToolUsageError(f"two flows are named {name!r}")
            named[name] = flow
        if not named:
            raise ToolUsageError("DurableFlows needs at least one flow")
        self.runner = runner
        self.store: CheckpointStore = runner.checkpoints
        self._flows = named

    @property
    def names(self) -> tuple[str, ...]:
        """The registered flow names."""
        return tuple(self._flows)

    def flow(self, name: str) -> Flow[Any]:
        """The registered flow called `name`.

        Raises:
            ToolUsageError: no flow has that name.
        """
        try:
            return self._flows[name]
        except (KeyError, TypeError):
            raise ToolUsageError(f"no durable flow is registered as {name!r}") from None

    async def advance(
        self,
        flow: str,
        run_id: str,
        state: Mapping[str, Any] | None = None,
        identity: RunIdentity | None = None,
    ) -> FlowStepReport:
        """Start the run (with `state`) or continue it by one superstep; idempotent per position.

        Raises:
            ToolUsageError: unknown flow, a new run without `state`, a run id held by another
                flow, or an identity other than the run's.
            CheckpointConflict: another worker holds the run (retry).
            RecoveryNeedsDecision: a call may have run before a crash and is not idempotent.
            FlowError: a node, route or merge failed (the run is marked failed).
        """
        spec, ctx = self.flow(flow), self._context(run_id, identity)
        found = await self._inspect(spec, run_id, ctx)
        if found is None:
            if state is None:
                raise ToolUsageError(f"run {run_id!r} is new: advance needs its initial state")
            return await self._step(spec, ctx, first=True, state=state)
        status, checkpoint = found
        if status == "running":
            return await self._step(spec, ctx)
        return _stored_report(run_id, status, checkpoint)

    async def resume(
        self,
        flow: str,
        run_id: str,
        approval_id: str,
        *,
        timed_out: bool = False,
        identity: RunIdentity | None = None,
    ) -> FlowStepReport:
        """Re-check the approval store; a decided record resumes the paused superstep once.

        A record still pending (or no record: a forged notice) leaves the run paused. With
        `timed_out`, a still-pending record is first decided as denied in the store. A retry
        after the decision was used continues the stored run instead.

        Raises:
            ToolUsageError: unknown flow, the approval's port has no store, or as `advance`.
            CheckpointNotFound: nothing is stored for `run_id`.
            ApprovalAlreadyDecided: the store record was already used.
            CheckpointConflict, RecoveryNeedsDecision, FlowError: as `advance`.
        """
        if not isinstance(approval_id, str) or not approval_id:
            raise ToolUsageError("approval_id must be a non-empty str")
        spec, ctx = self.flow(flow), self._context(run_id, identity)
        found = await self._inspect(spec, run_id, ctx)
        if found is None:
            raise CheckpointNotFound(run_id)
        status, checkpoint = found
        if status == "running":
            return await self._step(spec, ctx)
        report = _stored_report(run_id, status, checkpoint)
        pending = checkpoint.pending
        if status != "paused" or pending is None:
            return report
        if pending.approval_id != approval_id:
            logger.warning("flow %s run %s: notice for another approval ignored", flow, run_id)
            return report
        decision = await self._stored_decision(pending, timed_out=timed_out)
        if decision is None:
            return report  # nobody decided in the store: keep waiting
        return await self._step(spec, ctx, checkpoint=checkpoint, decision=decision)

    async def _stored_decision(
        self, pending: ApprovalRequest, *, timed_out: bool
    ) -> ApprovalDecision | None:
        """The store's decision for `pending`, or None while it is undecided (or unknown)."""
        store = self._store_for(pending)
        try:
            record = await store.get(pending.approval_id)
        except ApprovalNotFound:
            logger.warning("approval %s has no store record: still waiting", pending.approval_id)
            return None
        if record.consumed:
            raise ApprovalAlreadyDecided(pending.approval_id, status="consumed")
        if record.status == "pending":
            if not timed_out:
                return None
            try:
                record = await store.decide(pending.approval_id, TIMEOUT_DECISION)
            except ApprovalAlreadyDecided:  # a reviewer decided first: theirs counts
                record = await store.get(pending.approval_id)
        return record.decision

    def _store_for(self, pending: ApprovalRequest) -> ApprovalStore:
        """The approval store of the port that asked: engine approvals need one.

        Raises:
            ToolUsageError: the port has no store (a `DeferredApproval` is needed).
        """
        runner = self.runner
        port = runner.executor.approval
        if pending.kind == "node" and runner.approval is not None:
            port = runner.approval
        store = getattr(port, "store", None)
        if not isinstance(store, ApprovalStore):
            raise ToolUsageError(
                "engine approvals need a DeferredApproval port with an approval store"
            )
        return store

    async def pending_request(
        self, flow: str, run_id: str, identity: RunIdentity | None = None
    ) -> ApprovalRequest | None:
        """The full pending request (arguments included) of a paused run, for a reviewer.

        Raises:
            ToolUsageError: unknown flow, a run id held by another flow, or another identity.
        """
        spec = self.flow(flow)
        found = await self._inspect(spec, run_id, self._context(run_id, identity))
        if found is None or found[0] != "paused":
            return None
        return found[1].pending

    async def _inspect(
        self, spec: Flow[Any], run_id: str, ctx: ToolContext
    ) -> tuple[str, FlowCheckpoint] | None:
        """The run's checked status and checkpoint, or None for a new run."""
        try:
            return await self.runner.inspect(spec, run_id, ctx)
        except CheckpointNotFound:
            return None
        except CheckpointConflict as exc:  # held by another flow or a tool loop: never retry
            logger.warning("run %s is not a run of flow %s", run_id, spec.validate().name)
            raise ToolUsageError(f"run {run_id!r} belongs to another flow or a tool loop") from exc

    async def _step(
        self,
        spec: Flow[Any],
        ctx: ToolContext,
        *,
        first: bool = False,
        state: Mapping[str, Any] | None = None,
        checkpoint: FlowCheckpoint | None = None,
        decision: ApprovalDecision | None = None,
    ) -> FlowStepReport:
        runner = copy.copy(self.runner)
        runner.on_superstep = _end_step
        runner.sink = None  # folded into the context's sink
        ctx = dataclasses.replace(ctx, sink=self._sink(ctx, first=first))
        name, run_id = spec.validate().name, ctx.run_id
        if first:
            emit(ctx.sink, make_event(EventType.RUN_STARTED, ctx))
        try:
            if checkpoint is not None and decision is not None:
                result = await runner.resume(spec, checkpoint, decision, ctx)
            elif state is not None:
                result = await runner.run(spec, state, ctx)
            else:
                result = await runner.recover(spec, run_id, ctx)
        except SliceEnd as end:
            return FlowStepReport(run_id=run_id, flow=name, status="running", steps=end.steps)
        except Exception:
            await self._end_if_failed(spec, ctx)
            raise
        report = _report(run_id, name, result)
        if report.status in _TERMINAL:
            self._end(ctx, ok=report.status == "done")
        return report

    def _context(self, run_id: str, identity: RunIdentity | None) -> ToolContext:
        if not isinstance(run_id, str) or not run_id:
            raise ToolUsageError("run_id must be a non-empty str")
        who = identity if identity is not None else RunIdentity()
        if not isinstance(who, RunIdentity):
            raise ToolUsageError(f"identity must be a RunIdentity, got {type(who).__name__}")
        extras = {"principal": who.principal} if who.principal else None
        return self.runner.executor.context(
            run_id=run_id, user_id=who.user_id, thread_id=who.thread_id, extras=extras
        )

    def _sink(self, ctx: ToolContext, *, first: bool) -> EventSink | None:
        extra = self.runner.sink
        inner: EventSink | None = ctx.sink
        if extra is not None and all(extra is not sink for sink in members(ctx.sink)):
            inner = extra if inner is None else FanoutSink(inner, extra)
        return SliceSink(inner, first=first) if inner is not None else None

    async def _end_if_failed(self, spec: Flow[Any], ctx: ToolContext) -> None:
        """A step that failed the run (stored as failed) ends it; any other failure keeps it."""
        try:
            found = await self._inspect(spec, ctx.run_id, ctx)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "run %s: status not read back", ctx.run_id)
            return
        if found is not None and found[0] == "failed":
            self._end(ctx, ok=False)

    def _end(self, ctx: ToolContext, *, ok: bool) -> None:
        emit(ctx.sink, make_event(EventType.RUN_FINISHED, ctx, ok=ok))
        self.runner.executor.end_run(ctx.run_id)


async def _end_step(steps: int) -> None:
    """The runner's superstep hook: the checkpoint is saved, so this engine step ends here."""
    raise SliceEnd(steps)


def _stored_report(run_id: str, status: str, checkpoint: FlowCheckpoint) -> FlowStepReport:
    """A run that already paused or ended: its report, from the checked checkpoint."""
    ended = status in ("done", "denied")
    return FlowStepReport.model_validate(
        {
            "run_id": run_id,
            "flow": checkpoint.flow,
            "status": status,
            "steps": checkpoint.steps,
            "state": checkpoint.state.model_dump(mode="json") if ended else None,
            "pending": _pending(checkpoint.pending) if status == "paused" else None,
        }
    )


def _report(run_id: str, flow: str, result: FlowResult[Any]) -> FlowStepReport:
    paused = result.status == "paused"
    return FlowStepReport(
        run_id=run_id,
        flow=flow,
        status=result.status,
        steps=result.steps,
        state=None if paused else result.state.model_dump(mode="json"),
        pending=_pending(result.pending) if paused else None,
        denied_at=result.denied_at,
    )


def _pending(request: ApprovalRequest | None) -> PendingApproval | None:
    if request is None:
        return None
    return PendingApproval(
        approval_id=request.approval_id,
        kind=request.kind,
        tool=request.tool,
        node=str(request.metadata.get("node", "")),
        hint=request.hint,
    )


__all__ = [
    "ApprovalNotice",
    "DurableFlows",
    "FlowStepReport",
    "PendingApproval",
    "RunIdentity",
]
