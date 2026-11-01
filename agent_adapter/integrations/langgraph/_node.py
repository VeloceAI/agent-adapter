"""One compiled flow node: a local-runner node visit inside a LangGraph task.

The visit runs through `flow._nodes.run_node`, so hooks, timeout, retries, `Call`, `Map`
(`executor.run_many`) and `Join` behave as on the local runner. An approval the visit leaves
pending pauses the task: with a store-backed port the store decides (a resume value never
does); otherwise a signed resume value does, answered back into the visit as a
`ResumeAnswer`. The paused visit's progress is sealed into its interrupt value (`_durable`)
and every answer is claimed once per task run (`_spent`).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import BaseModel

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalDenied,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.integration import (
    AnswerEntry,
    NotSealable,
    approval_store,
    make_answer,
    unsealable_calls,
)
from agent_adapter.flow.graph import LOOP_FIELD, VISITS_FIELD, FlowGraph, NodeSpec
from agent_adapter.flow.state import apply_updates, validated_state
from agent_adapter.flow.visit import (
    SUPERSEDED,
    TAMPERED,
    NodeEnv,
    NodeOutcome,
    node_call_id,
    not_sealable,
    route_state,
    route_targets,
    run_node,
    withdraw,
)
from agent_adapter.hooks.base import Hook, NodeInvocation
from agent_adapter.integrations.langgraph._channels import EDIT, UPDATE, tagged
from agent_adapter.integrations.langgraph._context import run_ids, task_namespace
from agent_adapter.integrations.langgraph._durable import (
    STATE_KEY,
    Seal,
    Surfaced,
    TaskState,
    load,
    seal,
)
from agent_adapter.integrations.langgraph._gate import ask_gate, gate_request
from agent_adapter.integrations.langgraph._spent import (
    ClaimKey,
    claim,
    mark_claims_started,
    mark_started,
    task_scope,
)
from agent_adapter.integrations.langgraph._store_decisions import StoreDecisions
from agent_adapter.integrations.langgraph.approval import (
    ask_claimed,
    binding,
    defer_to_node,
    payload,
    resume_count,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledFlow:
    """What every compiled node of one flow shares; `run_compiled` reads it too."""

    graph: FlowGraph
    executor: Executor
    env: NodeEnv
    gated: tuple[str, ...]  # nodes with a pause gate, in declaration order
    digest: str  # graph digest, sealed into paused task state
    seal_key: bytes  # seals paused task state
    mac_key: bytes  # signs resume values (the LangGraphApproval port's key)


class FlowNode:
    """The LangGraph node of one flow node."""

    def __init__(self, info: CompiledFlow, spec: NodeSpec) -> None:
        self.info = info
        self.spec = spec
        self.__name__ = spec.name

    async def __call__(self, state: dict[str, Any], config: RunnableConfig) -> Any:
        """Run one visit and return its writes (tagged) and route targets as a `Command`."""
        visit = _Visit(self.info, self.spec, state, config)
        with task_scope(visit.run_id, visit.namespace), defer_to_node(visit.namespace):
            return await visit.run()

    def __repr__(self) -> str:
        return f"FlowNode({self.info.graph.name!r}, {self.spec.name!r})"


class _Visit:
    def __init__(
        self, info: CompiledFlow, spec: NodeSpec, state: Mapping[str, Any], config: Any
    ) -> None:
        graph = info.graph
        self.info, self.spec, self.flow, self.name = info, spec, graph.name, spec.name
        self.config = config
        fields = graph.state_model.model_fields
        values = {key: value for key, value in state.items() if key in fields}
        what = f"state of flow {self.flow!r} at node {self.name!r}"
        self.state: BaseModel = validated_state(graph.state_model, values, what=what)
        self.visits: dict[str, int] = dict(state.get(VISITS_FIELD) or {})
        self.loops: dict[str, int] = dict(state.get(LOOP_FIELD) or {})
        self.visit = self.visits.get(self.name, 0) + 1
        self.visit_id = node_call_id(self.flow, self.name, self.visit)
        self.run_id, thread_id = run_ids(config)
        if not self.run_id:
            raise ToolUsageError(
                f"flow {self.flow!r}: a compiled flow needs config['configurable']['thread_id'] "
                "(or 'run_id') to name its run"
            )
        self.namespace = task_namespace(config)
        self.ctx: ToolContext = info.executor.context(run_id=self.run_id, thread_id=thread_id)
        self.gate_claims: list[tuple[str, ...]] = []  # started when the gated node's body runs
        store: ApprovalStore | None = approval_store(info.executor.approval)
        self.decisions = (
            None
            if store is None
            else StoreDecisions(
                store,
                flow=self.flow,
                run_id=self.run_id,
                namespace=self.namespace,
                gate_claims=self.gate_claims,
                pause=self._pause,
            )
        )
        self.seal = Seal(info.seal_key, info.digest, self.run_id, self.namespace, info.env.allowed)
        self.task = TaskState()

    async def run(self) -> Any:
        loaded = await load(self.config, self.seal, self.flow)  # this execution's own copy
        if loaded is None and self.namespace and resume_count(self.config) > 0:
            # answers were given to a pause whose sealed progress is gone: never re-run blind
            logger.warning(
                "flow %s node %s: resume values without sealed state", self.flow, self.name
            )
            raise FlowError(
                f"flow {self.flow!r}: {TAMPERED}: the paused node's sealed state is missing",
                flow=self.flow,
                node=self.name,
            )
        self.task = loaded if loaded is not None else TaskState()
        replayed = [(s, *await self._answer(s)) for s in list(self.task.surfaced)]
        gate = await self._gate(replayed) if self.spec.pause is not False else None
        state = self._edited(gate)
        outcome = await self._visit(state, gate, replayed)
        return await self._command(state, gate, outcome)

    async def _answer(self, surfaced: Surfaced) -> tuple[ApprovalDecision, ClaimKey | None]:
        """A surfaced approval's decision (kept, from the store, or from its resume value)
        and the claim on it (None: the port decided at once)."""
        if surfaced.via == "port" and surfaced.decision is not None:
            return surfaced.decision, None
        decided = None if self.decisions is None else await self.decisions.decide(surfaced)
        if decided is not None:
            return decided
        request = surfaced.request
        decision, claimed = ask_claimed(
            request,
            key=self.info.mac_key,
            exact=surfaced.exact,
            extra={STATE_KEY: self._sealed()},
            scope=self.info.digest,
            started=False,
        )
        if request.kind == "node":
            bound = binding(request, exact=surfaced.exact, scope=self.info.digest)
            self.gate_claims.append(("resume", bound))
        surfaced.via = "interrupt"
        return decision, claimed

    def _pause(self, request: ApprovalRequest) -> NoReturn:
        """Pause the task for a store-backed approval (no resume value is read)."""
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Interrupt

        exact = request.kind == "node" or request.call_id.startswith(f"{self.visit_id}:")
        bound = binding(request, exact=exact, with_id=True, scope=self.info.digest)
        value = payload(request, bound, {STATE_KEY: self._sealed()})
        raise GraphInterrupt((Interrupt.from_ns(value=value, ns=self.namespace),))

    def _sealed(self) -> dict[str, Any]:
        return seal(self.seal, self.task)

    async def _gate(self, replayed: list[_Replayed]) -> ApprovalDecision:
        """The gate's decision: replayed, from the port, the store or a resume value."""
        if replayed and replayed[0][0].request.kind == "node":
            decision = replayed[0][1]
        else:
            request = gate_request(
                self.flow, self.spec, self.state, self.ctx, self.visit, self.namespace
            )
            port = self.info.executor.approval
            answer = await ask_gate(self.flow, self.name, request, port, peek=False)
            if answer.pending is None:
                decision = _decided(answer.decision)
                self.task.surfaced.append(Surfaced(request, True, "port", decision))
            else:
                surfaced = Surfaced(answer.pending, True)
                self.task.surfaced.append(surfaced)
                decision, _ = await self._answer(surfaced)
        if not decision.approved:
            logger.info("flow %s node %s denied: %s", self.flow, self.name, decision.reason)
            raise ApprovalDenied(
                f"node:{self.flow}.{self.name}", call_id=self.visit_id, reason=decision.reason
            )
        claim(self.run_id, self.namespace, "gate", self.visit_id, started=False)
        self.gate_claims.append(("gate", self.visit_id))
        return decision

    def _edited(self, gate: ApprovalDecision | None) -> BaseModel:
        if gate is None or gate.edited_args is None:
            return self.state
        try:
            return apply_updates(self.state, [gate.edited_args], self.info.graph.rules)
        except FlowError as exc:
            raise FlowError(
                f"flow {self.flow!r} node {self.name!r}: edited state is invalid: {exc}",
                flow=self.flow,
                node=self.name,
            ) from exc

    async def _visit(
        self,
        state: BaseModel,
        gate: ApprovalDecision | None,
        replayed: list[_Replayed],
    ) -> NodeOutcome:
        """Run the visit; each approval it leaves pending is one pause, then a re-run."""
        entries: list[AnswerEntry] = []
        claims: list[tuple[ClaimKey, int]] = []  # an answer's claim and its entry's index
        for surfaced, decision, claimed in replayed:
            if surfaced.request.kind == "tool":
                _add(entries, claims, (surfaced.request, decision, surfaced.exact), claimed)
        while True:
            answer = make_answer(entries) if entries else None
            try:
                outcome = await run_node(
                    self._env(),
                    self.spec,
                    state,
                    self.ctx,
                    visit=self.visit,
                    approval=gate,
                    answer=answer,
                    items=self.task.items or None,
                    journal=self.task.journal or None,
                )
            finally:  # an answer the executor used started its call, whatever its context
                if answer is not None:
                    used = answer.used()
                    mark_claims_started(key for key, index in claims if used[index])
            pending = outcome.pending
            if pending is None:
                return outcome
            exact = pending.call_id.startswith(f"{self.visit_id}:")
            await self._withdraw_superseded(outcome, pending)
            self.task.items = dict(outcome.items)
            self.task.journal = dict(outcome.journal)
            surfaced = Surfaced(pending, exact)
            self.task.surfaced.append(surfaced)
            await self._check_sealable(outcome)
            decision, claimed = await self._answer(surfaced)
            _add(entries, claims, (pending, decision, exact), claimed)

    def _env(self) -> NodeEnv:
        """The node environment; a gated node marks its gate claims started as its body runs."""
        if not self.gate_claims:
            return self.info.env
        env = self.info.env
        return dataclasses.replace(env, hooks=(*env.hooks, _GateStarted(self.gate_claims)))

    async def _check_sealable(self, outcome: NodeOutcome) -> None:
        """A pause that could not be replayed fails closed; its pending records are withdrawn."""
        try:
            problems = unsealable_calls(self.task.journal)
            if problems:
                raise NotSealable("journal", "call", problems[0])
            self._sealed()
        except NotSealable as exc:
            await withdraw(self.flow, self.ctx, outcome.pendings)
            raise not_sealable(self.flow, exc) from exc

    async def _withdraw_superseded(self, outcome: NodeOutcome, pending: ApprovalRequest) -> None:
        """Deny the visit's other pending records whose call ids a re-run does not repeat."""
        assigned = re.compile(re.escape(self.visit_id) + r":\d+")
        stale = [
            (request, store)
            for request, store in outcome.pendings
            if request.approval_id != pending.approval_id
            and not assigned.fullmatch(request.call_id)
        ]
        if stale:
            await withdraw(self.flow, self.ctx, stale, SUPERSEDED)

    async def _command(
        self, state: BaseModel, gate: ApprovalDecision | None, outcome: NodeOutcome
    ) -> Any:
        from langgraph.types import Command

        graph = self.info.graph
        seen = route_state(graph, state, outcome)  # also checks keys and values
        writes: list[tuple[str, Any]] = []
        if gate is not None and gate.edited_args is not None:
            writes.extend((k, tagged(self.name, EDIT, 0, v)) for k, v in gate.edited_args.items())
        for seq, update in enumerate(outcome.updates):
            writes.extend((k, tagged(self.name, UPDATE, seq, v)) for k, v in update.items())
        writes.append((VISITS_FIELD, tagged(self.name, UPDATE, 0, {self.name: self.visit})))
        targets: list[str] = []
        route = graph.routes.get(self.name)
        if route is not None:
            loops = dict(self.loops)
            targets = await route_targets(self.flow, route, seen, self.visits, loops)
            if loops != self.loops:
                counted = {self.name: loops[self.name]}
                writes.append((LOOP_FIELD, tagged(self.name, UPDATE, 0, counted)))
        return Command(update=writes, goto=list(dict.fromkeys(targets)))


class _GateStarted(Hook):
    """Runs after the node's own before_node hooks: the gated body starts now."""

    def __init__(self, claims: list[tuple[str, ...]]) -> None:
        self.claims = claims

    async def before_node(self, inv: NodeInvocation) -> None:
        for parts in self.claims:
            mark_started(*parts)


_Replayed = tuple[Surfaced, ApprovalDecision, ClaimKey | None]  # an answer and its claim


def _add(
    entries: list[AnswerEntry],
    claims: list[tuple[ClaimKey, int]],
    entry: AnswerEntry,
    claimed: ClaimKey | None,
) -> None:
    if claimed is not None:
        claims.append((claimed, len(entries)))
    entries.append(entry)


def _decided(decision: ApprovalDecision | None) -> ApprovalDecision:
    if decision is None:  # pragma: no cover - GateAnswer holds a decision or a pending request
        raise ToolUsageError("gate answered with neither a decision nor a pending request")
    return decision
