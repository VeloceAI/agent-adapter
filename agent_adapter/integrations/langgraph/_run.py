"""`run_compiled`: run a compiled flow graph to a `FlowResult` (start, gates, resumes)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from agent_adapter.approval.batch import PreparedApproval
from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    FlowError,
    FlowLoopExceeded,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import approval_store, same_decision, settle_own
from agent_adapter.flow import FlowResult
from agent_adapter.flow.graph import LOOP_FIELD, VISITS_FIELD
from agent_adapter.flow.state import validated_state
from agent_adapter.flow.visit import frozen
from agent_adapter.integrations.langgraph._context import run_ids
from agent_adapter.integrations.langgraph._gate import GateAnswer, ask_gate, gate_request, spend
from agent_adapter.integrations.langgraph._node import CompiledFlow
from agent_adapter.integrations.langgraph.approval import (
    PAYLOAD_TYPE,
    binding,
    payload,
    resume_value,
)
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 100  # FlowRunner's max_steps default
_GATES = object()  # resume step: ask the superstep's gates again before anything runs


async def run_compiled(app: Any, state: Any, config: Mapping[str, Any]) -> FlowResult[Any]:
    """Run a graph compiled from `compile_flow` until it is done, paused or denied.

    `state` is the initial state (a model or mapping), an `ApprovalDecision` answering the
    pending approval (bound to it), a LangGraph `Command`, or None to continue. Gates whose
    port decides at once are passed without stopping; `recursion_limit` defaults to 100.

    Raises:
        ToolUsageError: `app` was not compiled from `compile_flow`, or a bad `state`.
        FlowError: a node, route or merge failed, or nothing waits for a decision.
        FlowLoopExceeded: a route passed `max_loops`, or the recursion limit.
    """
    info = compiled_flow(app)
    runner = _Runner(app, info, _config(app, config))
    return await runner.start(state)


def compiled_flow(app: Any) -> CompiledFlow:
    """The flow behind a graph compiled from `compile_flow`.

    Raises:
        ToolUsageError: `app` is not one.
    """
    info = getattr(getattr(app, "builder", None), "aa_flow", None)
    if not isinstance(info, CompiledFlow):
        raise ToolUsageError("run_compiled needs a graph compiled from compile_flow(...)")
    return info


def _config(app: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ToolUsageError(f"config must be a mapping, got {type(config).__name__}")
    out = dict(config)
    configurable = dict(out.get("configurable") or {})
    if app.checkpointer is None and not any(run_ids({"configurable": configurable})):
        configurable["run_id"] = new_id("run_")
    out["configurable"] = configurable
    out.setdefault("recursion_limit", DEFAULT_RECURSION_LIMIT)
    return out


class _Runner:
    def __init__(self, app: Any, info: CompiledFlow, config: dict[str, Any]) -> None:
        self.app, self.info, self.config = app, info, config
        self.graph = info.graph
        self.ctx = info.executor.context(run_id=run_ids(config)[0] or None)
        self.steps = 0

    async def start(self, state: Any) -> FlowResult[Any]:
        command = await self._input(state)
        self._emit(EventType.FLOW_STARTED)
        try:
            if command is _GATES:
                command = await self._gates(await self._snapshot())
            result = command if isinstance(command, FlowResult) else await self._drive(command)
        except Exception as exc:
            if not is_control_flow(exc):
                self._emit(EventType.FLOW_FINISHED, status="failed", steps=self.steps)
            raise
        self._emit(EventType.FLOW_FINISHED, status=result.status, steps=result.steps)
        return result

    def _emit(self, type_: str, **data: Any) -> None:
        emit(self.ctx.sink, make_event(type_, self.ctx, flow=self.graph.name, **data))

    async def _input(self, state: Any) -> Any:
        from langgraph.types import Command

        if isinstance(state, ApprovalDecision):
            return await self._resume(state)
        if state is None:
            return await self._continue()
        if isinstance(state, Command):
            return state
        model = self._initial(state)
        values = {name: getattr(model, name) for name in type(model).model_fields}
        return {**values, VISITS_FIELD: {}, LOOP_FIELD: {}}

    def _initial(self, state: Any) -> BaseModel:
        model = self.graph.state_model
        if isinstance(state, model):
            return state
        if isinstance(state, Mapping):
            what = f"initial state of flow {self.graph.name!r}"
            return validated_state(model, state, what=what)
        raise ToolUsageError(
            f"flow {self.graph.name!r}: state must be a {model.__name__}, a mapping, an "
            f"ApprovalDecision, a Command or None, got {type(state).__name__}"
        )

    async def _drive(self, command: Any) -> FlowResult[Any]:
        while True:
            outcome = await self._invoke(command)
            if isinstance(outcome, FlowResult):
                return outcome
            values, interrupts = outcome
            if interrupts:
                return await self._paused(_first_request(self.graph.nodes, interrupts))
            snapshot = await self._snapshot()
            if snapshot is None or not snapshot.next:
                return self._done(values)
            gates = await self._gates(snapshot)
            if isinstance(gates, FlowResult):
                return gates
            command = None  # every gate of the superstep approved at once: go on

    async def _invoke(self, command: Any) -> Any:
        from langgraph.errors import GraphRecursionError

        try:
            if self.app.checkpointer is None:
                stream = self.app.astream(command, self.config, stream_mode="values")
                chunks = [chunk async for chunk in stream]
                self.steps = max(0, len(chunks) - 1)
                return chunks[-1], ()
            out = await self.app.ainvoke(command, self.config, version="v2")
            return out.value, out.interrupts
        except GraphRecursionError as exc:
            limit = int(self.config["recursion_limit"])
            first = await self._first_next()
            logger.warning("flow %s stopped after recursion_limit=%d", self.graph.name, limit)
            raise FlowLoopExceeded(self.graph.name, first, limit) from exc
        except ApprovalDenied as exc:
            gated = self._gate_of(exc.tool)
            if gated is None:
                raise
            return await self._denied(gated)

    def _gate_of(self, tool: str) -> str | None:
        prefix = f"node:{self.graph.name}."
        node = tool.removeprefix(prefix) if tool.startswith(prefix) else None
        return node if node in self.info.gated else None

    async def _snapshot(self) -> Any:
        if self.app.checkpointer is None:
            return None
        snapshot = await self.app.aget_state(self.config)
        self.steps = int((snapshot.metadata or {}).get("step") or 0)
        return snapshot

    async def _first_next(self) -> str:
        snapshot = await self._snapshot()
        upcoming = set(snapshot.next) if snapshot is not None else set()
        return next((n for n in self.graph.nodes if n in upcoming), self.graph.start)

    async def _committed(self) -> dict[str, Any]:
        """The state the paused superstep started from (finished siblings' writes left out)."""
        snapshot = await self._snapshot()
        found = await self.app.checkpointer.aget_tuple(snapshot.config)
        return dict(found.checkpoint["channel_values"]) if found is not None else {}

    def _model(self, values: Mapping[str, Any]) -> BaseModel:
        fields = self.graph.state_model.model_fields
        data = {key: value for key, value in values.items() if key in fields}
        return validated_state(self.graph.state_model, data, what=f"flow {self.graph.name!r}")

    def _done(self, values: Mapping[str, Any]) -> FlowResult[Any]:
        visits = frozen(dict(values.get(VISITS_FIELD) or {}))
        return FlowResult(self._model(values), "done", self.steps, visits)

    async def _paused(self, pending: ApprovalRequest | None) -> FlowResult[Any]:
        values = await self._committed()
        visits = frozen(dict(values.get(VISITS_FIELD) or {}))
        if pending is not None:
            logger.info("flow %s waits for approval %s", self.graph.name, pending.approval_id)
        return FlowResult(self._model(values), "paused", self.steps, visits, pending=pending)

    async def _denied(self, node: str) -> FlowResult[Any]:
        values = await self._committed()
        visits = frozen(dict(values.get(VISITS_FIELD) or {}))
        model = self._model(values)
        return FlowResult(model, "denied", self.steps, visits, denied_at=node)

    async def _gate_answers(self, snapshot: Any) -> list[tuple[str, GateAnswer]]:
        """Ask the port about each gate of the waiting superstep, in order, until one waits."""
        values = await self._committed()
        state, visits = self._model(values), dict(values.get(VISITS_FIELD) or {})
        port = self.info.executor.approval
        tasks = {task.name: f"{task.name}:{task.id}" for task in snapshot.tasks}
        answers: list[tuple[str, GateAnswer]] = []
        for name in (n for n in self.info.gated if n in snapshot.next):
            spec, visit = self.graph.nodes[name], visits.get(name, 0) + 1
            request = gate_request(self.graph.name, spec, state, self.ctx, visit, tasks[name])
            answer = await ask_gate(self.graph.name, name, request, port, peek=True)
            answers.append((name, answer))
            if not answer.approved:
                break
        return answers

    async def _gates(self, snapshot: Any) -> FlowResult[Any] | None:
        """Paused or denied at the first gate that is not approved; None when all are."""
        answers = await self._gate_answers(snapshot)
        if not answers:
            return await self._paused(None)  # stopped before nodes that have no gate
        name, answer = answers[-1]
        if answer.pending is not None:
            return await self._paused(answer.pending)
        if not answer.approved:
            await spend(answer, self.info.executor.approval)
            return await self._denied(name)
        return None

    async def _continue(self) -> Any:
        """None continues; at a gate pause the gates are asked again before anything runs."""
        snapshot = await self._snapshot()
        if snapshot is None or _ours(snapshot.interrupts):
            return None
        if not any(n in snapshot.next for n in self.info.gated):
            return None
        self._check_gates(snapshot)
        return _GATES

    def _check_gates(self, snapshot: Any) -> None:
        """ToolUsageError when several gates of one superstep wait without a store-backed port."""
        waiting = [n for n in self.info.gated if n in snapshot.next]
        if len(waiting) > 1 and approval_store(self.info.executor.approval) is None:
            raise ToolUsageError("several gates in one superstep need a store-backed approval port")

    async def _resume(self, decision: ApprovalDecision) -> Any:
        from langgraph.types import Command

        if self.app.checkpointer is None:
            raise ToolUsageError("resuming a compiled flow needs a checkpointer")
        snapshot = await self._snapshot()
        port = self.info.executor.approval
        store = approval_store(port)
        ours = _ours(snapshot.interrupts)
        if ours:
            first = _ordered(self.graph.nodes, ours)[0]
            pending = ApprovalRequest.model_validate(first.value["request"])
            if store is not None and await _holds(store, pending):  # the store decides
                await _decide(store, pending, decision)
                return None
            value = resume_value(self.info.mac_key, first, decision)
            return Command(resume={first.id: value})
        self._check_gates(snapshot)
        answers = await self._gate_answers(snapshot)
        if not answers or answers[-1][1].approved:
            raise FlowError(
                f"flow {self.graph.name!r}: nothing waits for a decision", flow=self.graph.name
            )
        name, gate = answers[-1]
        if gate.pending is None:  # denied already (in the store, out of band)
            stored = gate.decision
            if stored is None or not same_decision(stored, decision):
                logger.warning("flow %s gate %s was decided otherwise", self.graph.name, name)
                approval_id = gate.record.approval_id if gate.record is not None else ""
                raise ApprovalAlreadyDecided(approval_id, status="denied")
            await spend(gate, port)
            return await self._denied(name)
        pending = gate.pending
        if not decision.approved:  # a denied gate ends the run: nothing in its superstep runs
            await settle_own(store, pending, decision)
            return await self._denied(name)
        if store is not None and isinstance(port, PreparedApproval):
            await _decide(store, pending, decision)  # the gated node's port consumes it
            return _GATES  # every gate of the superstep must be decided before it runs
        bound = binding(pending, scope=self.info.digest)
        signed = resume_value(self.info.mac_key, payload(pending, bound), decision)
        return Command(resume=signed)


async def _holds(store: ApprovalStore, request: ApprovalRequest) -> bool:
    try:
        await store.get(request.approval_id)
    except ApprovalNotFound:
        return False
    return True


async def _decide(
    store: ApprovalStore, pending: ApprovalRequest, decision: ApprovalDecision
) -> None:
    """Record `decision` on the gate record; ApprovalAlreadyDecided if it holds another."""
    try:
        await store.decide(pending.approval_id, decision)
    except ApprovalAlreadyDecided:
        record = await store.get(pending.approval_id)
        if record.consumed:
            logger.warning("approval %s was already used", pending.approval_id)
            raise
        if not same_decision(record.decision, decision):
            logger.warning("approval %s was decided otherwise in the store", pending.approval_id)
            raise


def _ours(interrupts: Any) -> list[Any]:
    return [
        i
        for i in interrupts or ()
        if isinstance(i.value, Mapping) and i.value.get("type") == PAYLOAD_TYPE
    ]


def _ordered(nodes: Mapping[str, Any], interrupts: list[Any]) -> list[Any]:
    order = {name: index for index, name in enumerate(nodes)}

    def key(interrupt: Any) -> int:
        node = interrupt.value.get("request", {}).get("metadata", {}).get("node", "")
        return order.get(node, len(order))

    return sorted(interrupts, key=key)


def _first_request(nodes: Mapping[str, Any], interrupts: Any) -> ApprovalRequest | None:
    ours = _ours(interrupts)
    if not ours:
        logger.warning("compiled flow paused for an interrupt that is not an approval")
        return None
    return ApprovalRequest.model_validate(_ordered(nodes, ours)[0].value["request"])


__all__ = ["DEFAULT_RECURSION_LIMIT", "compiled_flow", "run_compiled"]
