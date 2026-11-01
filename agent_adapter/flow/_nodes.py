"""Run one flow node: hooks, timeout and retries, and its action (fn, Call, Map, Join)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalPending,
    FlowError,
    ToolNotFound,
    ToolTimeout,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._approve import approval_store, pending_store
from agent_adapter.exec._canon import Decoder, Encoder
from agent_adapter.exec._journal import Journal, JournalData, _make_journal
from agent_adapter.exec._resume import Pending, ResumeAnswer, _visit_scope, current_journal
from agent_adapter.exec.executor import Executor, bind_model
from agent_adapter.flow._invoke import describe_error, invoke_node_fn
from agent_adapter.flow._map import ItemOutcome, MapVisit, item_journal, with_ids
from agent_adapter.flow._trace import changed_fields, traced_updates
from agent_adapter.flow.actions import MODEL_KEY, PROMPT_KEY, Call, Join, Map
from agent_adapter.flow.graph import NodeSpec
from agent_adapter.hooks.base import Hook, NodeInvocation
from agent_adapter.hooks.dispatch import run_after, run_before, run_on_error
from agent_adapter.runtime.current import use_context
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeEnv:
    """What every node of one flow run shares."""

    flow: str
    executor: Executor
    hooks: tuple[Hook, ...]
    model: LLMPort | None = None
    encoder: Encoder | None = None  # FlowRunner(seal_encoder=)
    decoder: Decoder | None = None  # FlowRunner(seal_decoder=)
    allowed: Mapping[str, type] = field(default_factory=dict)  # classes a replay may build
    trace_state: bool = False  # FlowRunner(trace_state=): node.finished carries the values


@dataclass(frozen=True)
class NodeOutcome:
    """A node's updates, or the approval it is waiting for."""

    node: str
    updates: tuple[Mapping[str, Any], ...] = ()
    pending: ApprovalRequest | None = None
    items: Mapping[int, ItemOutcome] = field(default_factory=dict)  # finished Map items
    journal: JournalData = field(default_factory=dict)  # finished executor calls, for the replay
    pendings: tuple[Pending, ...] = ()  # every record the visit left pending, with its store
    visit_id: str = ""  # the visit's call id; flow-assigned `{visit_id}:{i}` ids repeat on resume


@dataclass
class _Updates:
    items: list[Mapping[str, Any]] = field(default_factory=list)
    view: Mapping[str, Any] | None = None  # what after_node hooks see


def node_call_id(flow: str, node: str, visit: int) -> str:
    """The id of one node visit (node events and the node's context)."""
    return f"{flow}:{node}:{visit}"


async def run_node(
    env: NodeEnv,
    node: NodeSpec,
    state: BaseModel,
    ctx: ToolContext,
    *,
    visit: int,
    approval: ApprovalDecision | None = None,
    answer: ResumeAnswer | None = None,
    items: Mapping[int, ItemOutcome] | None = None,
    journal: JournalData | None = None,
) -> NodeOutcome:
    """Run one node visit; failures raise FlowError, a pending approval is returned.

    Args:
        answer: Answers the tool approvals this visit resumes; spent when the visit ends.
        items: `Map` items finished before the pause (by index); they do not run again.
        journal: Executor calls this visit finished before the pause; they replay.
    """
    call_id = node_call_id(env.flow, node.name, visit)
    node_ctx = ctx.child(call_id=call_id, approval=approval)
    node_ctx.extras[PROMPT_KEY] = node.prompt
    if env.model is not None:
        node_ctx.extras[MODEL_KEY] = bind_model(env.executor, env.model, node_ctx)
    ids = {"flow": env.flow, "node": node.name, "visit": visit}
    emit(ctx.sink, make_event(EventType.NODE_STARTED, ctx, call_id=call_id, **ids))
    visit_run = _Node(env, node, state, node_ctx, items or {})
    replay = _visit_journal(env, node, call_id, journal)
    # every visit gets its own scope, so a nested flow never sees an outer answer
    with use_context(node_ctx), _visit_scope(answer, replay) as made:
        try:
            done = await visit_run.run()
        except ApprovalPending as exc:
            request = _pending_request(exc, node_ctx, ids)
            default_store = approval_store(env.executor.approval)
            emit(
                ctx.sink,
                make_event(
                    EventType.NODE_PAUSED,
                    ctx,
                    call_id=call_id,
                    flow=env.flow,
                    node=node.name,
                    approval_id=request.approval_id,
                ),
            )
            logger.info(
                "flow %s node %s waits for approval %s", env.flow, node.name, exc.approval_id
            )
            return NodeOutcome(
                node.name,
                pending=request,
                items=dict(visit_run.items),
                journal=replay.data(drop=[item_journal(i) for i in visit_run.items]),
                pendings=_pendings(made, request, pending_store(exc, default_store)),
                visit_id=call_id,
            )
        except Exception as exc:
            if is_control_flow(exc):
                raise
            error = _node_error(env.flow, node.name, exc)
            emit(
                ctx.sink,
                make_event(
                    EventType.NODE_FAILED, ctx, call_id=call_id, error_kind=error.kind, **ids
                ),
            )
            if error is exc:
                raise
            raise error from exc
    changed = changed_fields(done.items)
    state_data: dict[str, Any] = {"changed": changed}
    if env.trace_state:
        state_data["updates"] = traced_updates(done.view or {}, changed)
    emit(
        ctx.sink,
        make_event(EventType.NODE_FINISHED, ctx, call_id=call_id, **ids, **state_data),
    )
    return NodeOutcome(node.name, updates=tuple(done.items))


def _visit_journal(env: NodeEnv, node: NodeSpec, call_id: str, data: JournalData | None) -> Journal:
    """The visit's journal: a child of the enclosing one (a nested flow), else its own.

    Map tool calls are kept by item index instead (identical items would collide), so a Map
    visit's own journal records nothing; `Map(fn=...)` items get child journals.
    """
    calls = node.kind != "map"
    outer = current_journal()
    if outer is not None:
        return outer.child(call_id, data, calls=calls)
    return _make_journal(
        data, encoder=env.encoder, decoder=env.decoder, calls=calls, allowed=env.allowed
    )


class _Node:
    """One node visit: hooks around the action, with the node's timeout and retries."""

    def __init__(
        self,
        env: NodeEnv,
        node: NodeSpec,
        state: BaseModel,
        ctx: ToolContext,
        items: Mapping[int, ItemOutcome],
    ) -> None:
        self.env = env
        self.node = node
        self.state = state
        self.ctx = ctx
        self.items: dict[int, ItemOutcome] = dict(items)  # finished Map items by index
        self.where = f"flow {env.flow!r} node {node.name!r}"

    async def run(self) -> _Updates:
        hooks = self.env.hooks
        inv = NodeInvocation(self.env.flow, self.node.name, self.state, self.ctx)
        short = await run_before(hooks, "before_node", inv)
        if short is not None:
            done = self._as_updates(short.value)
        else:
            try:
                done = await self._attempts()
            except ApprovalPending:
                raise
            except Exception as exc:
                if is_control_flow(exc):
                    raise
                recovered = await run_on_error(hooks, "on_node_error", inv, exc)
                if recovered is None:
                    raise
                done = self._as_updates(recovered.value)
        replaced = await run_after(hooks, "after_node", inv, done.view)
        if replaced is not done.view:
            done = self._as_updates(replaced)
        return done

    def _as_updates(self, value: Any) -> _Updates:
        if value is None:
            return _Updates()
        if not isinstance(value, Mapping):
            raise FlowError(
                f"{self.where} returned {type(value).__name__}, expected a mapping or None",
                flow=self.env.flow,
                node=self.node.name,
            )
        update = dict(value)
        return _Updates([update], update)

    async def _attempts(self) -> _Updates:
        """The action under the node's timeout, retried with the executor's backoff."""
        node = self.node
        expired: list[ToolTimeout] = []  # our own timeouts, told apart from a tool's by identity

        async def attempt() -> _Updates:
            scope = asyncio.timeout(node.timeout)
            try:
                async with scope:
                    return await self._action()
            except TimeoutError as exc:
                if not scope.expired():
                    raise
                error = ToolTimeout(f"node:{self.env.flow}.{node.name}", node.timeout or 0.0)
                expired.append(error)
                raise error from exc

        try:
            if node.retries == 0:
                return await attempt()
            [done] = await self.env.executor.parallel(attempt, retries=node.retries)
        except ToolTimeout as exc:
            if not any(exc is mine for mine in expired):
                raise
            raise FlowError(
                f"{self.where} timed out after {node.timeout:g}s",
                flow=self.env.flow,
                node=node.name,
            ) from exc
        return done

    async def _action(self) -> _Updates:
        action = self.node.action
        if isinstance(action, Call):
            return await self._call(action)
        if isinstance(action, Map):
            visit = MapVisit(
                self.env.flow, self.node, self.env.executor, self.state, self.ctx, self.items
            )
            into, kept = await visit.run(action)
            return _Updates([{into: value} for value in kept], {into: kept})
        if isinstance(action, Join):
            if action.fn is None:
                return _Updates()
            return self._as_updates(await invoke_node_fn(action.fn, self.state, self.ctx))
        return self._as_updates(await invoke_node_fn(action, self.state, self.ctx))

    async def _call(self, action: Call) -> _Updates:
        executor = self.env.executor
        spec = action.tool if isinstance(action.tool, ToolSpec) else self._spec(action.tool)
        args = action.args(self.state) if callable(action.args) else action.args
        tool_call_id = f"{self.ctx.call_id}:0"
        try:
            value = await executor.run_spec(spec, dict(args), self.ctx, call_id=tool_call_id)
        except ApprovalPending as exc:
            filled = with_ids(exc, tool=spec.name, call_id=tool_call_id)
            if filled is exc:
                raise
            raise filled from exc
        if action.into is None:
            return _Updates()
        return self._as_updates({action.into: value})

    def _spec(self, name: str) -> ToolSpec:
        registry = self.env.executor.registry
        if registry is None:
            raise ToolNotFound(f"no tool named {name!r}: the executor has no registry")
        return registry.get(name)


def _pendings(
    made: list[Pending], surfaced: ApprovalRequest, store: ApprovalStore | None
) -> tuple[Pending, ...]:
    """The visit's pending records, once each by approval id; the surfaced one included."""
    found: dict[str, Pending] = {}
    for request, holder in made:
        found.setdefault(request.approval_id, (request, holder))
    found.setdefault(surfaced.approval_id, (surfaced, store))
    return tuple(found.values())


def _pending_request(
    exc: ApprovalPending, ctx: ToolContext, ids: Mapping[str, Any]
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=exc.approval_id,
        run_id=exc.run_id or ctx.run_id,
        call_id=exc.call_id or ctx.call_id,
        tool=exc.tool,
        arguments=dict(exc.arguments or {}),
        kind="tool",
        metadata=dict(ids),
        created_at=time.time(),
    )


def _node_error(flow: str, node: str, exc: Exception) -> FlowError:
    if isinstance(exc, FlowError) and exc.flow == flow and exc.node == node:
        error = exc
    else:
        error = FlowError(
            f"flow {flow!r} node {node!r} failed: {describe_error(exc)}", flow=flow, node=node
        )
    if isinstance(exc, AdapterError):
        logger.warning("%s", error)
    else:
        logger.error("%s", error)  # the exception type only: its text stays at DEBUG
        logger.debug("%s", error, exc_info=exc)
    return error
