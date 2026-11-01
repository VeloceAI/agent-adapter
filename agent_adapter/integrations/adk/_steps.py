"""The work of each child node the ADK flow driver runs: gates, node visits, answers.

Each step returns JSON (the `exec._canon` tagged form), which ADK keeps in the child's
output event; a re-run driver gets that output back without the step running again.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_adapter.approval.batch import PreparedApproval
from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    ApprovalPending,
    FlowError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import (
    AnswerEntry,
    JournalData,
    NotSealable,
    Pending,
    approval_store,
    decode,
    encode,
    make_answer,
    unsealable_calls,
)
from agent_adapter.flow.graph import FlowGraph, NodeSpec
from agent_adapter.flow.visit import (
    NOT_SEALABLE,
    SPENT,
    SUPERSEDED,
    NodeEnv,
    NodeOutcome,
    ask_node_gate,
    call_route,
    node_gate_request,
    route_state,
    run_node,
    spend_once,
    superseded,
    withdraw,
)
from agent_adapter.integrations.adk._tokens import token_decision

logger = logging.getLogger(__name__)

MAX_STORES = 10_000


@dataclass(frozen=True)
class Plan:
    """What a compiled flow's driver needs: the graph, the node environment and the step cap."""

    graph: FlowGraph
    env: NodeEnv
    max_steps: int

    @property
    def flow(self) -> str:
        return self.graph.name


@dataclass(frozen=True)
class Visit:
    """A node visit attempt's outcome, as the driver reads it back."""

    node: str
    run_id: str  # the attempt's ADK run id
    updates: tuple[Mapping[str, Any], ...] = ()
    labels: tuple[Any, ...] | None = None  # the route's labels, when the node has a route
    pending: ApprovalRequest | None = None
    items: Mapping[int, Any] = field(default_factory=dict)
    journal: JournalData = field(default_factory=dict)
    pendings: tuple[ApprovalRequest, ...] = ()
    visit_id: str = ""

    def outcome(self) -> NodeOutcome:
        """The local runner's view of this outcome (stores looked up by approval id)."""
        pendings = tuple((request, _STORES.get(request.approval_id)) for request in self.pendings)
        return NodeOutcome(
            self.node,
            updates=self.updates,
            pending=self.pending,
            pendings=pendings,
            visit_id=self.visit_id,
        )


class _StoreTable:
    """The approval store behind each pending record a visit left (stores are not JSON)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # flows may run on several loops (run_sync threads)
        self._stores: OrderedDict[str, ApprovalStore] = OrderedDict()

    def keep(self, pendings: Sequence[Pending]) -> None:
        with self._lock:
            for request, store in pendings:
                if store is not None:
                    self._stores[request.approval_id] = store
            while len(self._stores) > MAX_STORES:
                self._stores.popitem(last=False)

    def get(self, approval_id: str) -> ApprovalStore | None:
        with self._lock:
            return self._stores.get(approval_id)


_STORES = _StoreTable()


def _encode(plan: Plan, node: str, value: Any) -> Any:
    try:
        return encode(value)
    except NotSealable as exc:
        raise _not_sealable(plan, node, exc) from exc


def _not_sealable(plan: Plan, node: str, exc: NotSealable) -> FlowError:
    logger.warning("flow %s node %s: %s at %s", plan.flow, node, NOT_SEALABLE, exc.path)
    return FlowError(
        f"flow {plan.flow!r} node {node!r}: {NOT_SEALABLE}: ADK keeps every node output in "
        f"its session events, and {exc}",
        flow=plan.flow,
        node=node,
    )


def decoded(plan: Plan, node: str, data: Any) -> Any:
    """A step output read back; classes outside the state model's reach are refused.

    Raises:
        FlowError: the output holds a value that cannot be rebuilt.
    """
    try:
        return decode(data, None, plan.env.allowed)
    except NotSealable as exc:
        raise FlowError(
            f"flow {plan.flow!r} node {node!r}: a stored output cannot be read back: {exc}",
            flow=plan.flow,
            node=node,
        ) from exc


async def gate_step(
    plan: Plan,
    node: NodeSpec,
    state: BaseModel,
    ctx: ToolContext,
    visit: int,
    answered: tuple[ApprovalRequest, Any] | None,
) -> dict[str, Any]:
    """A pause gate: the decision a token carries for the answered request, else the port's."""
    port = plan.env.executor.approval
    if answered is not None:
        request, response = answered
        decision = await _settled(token_decision(response, request), request, port)
        if decision is None:
            decision = await _stored_decision(request, port)
        if decision is not None:
            return {"decision": _encode(plan, node.name, decision)}
    else:
        request = node_gate_request(plan.flow, node, state, ctx, visit)
        _encode(plan, node.name, request)  # refused before the port makes a record
    answer = await ask_node_gate(plan.flow, node, request, port)
    if isinstance(answer, ApprovalRequest):
        return {"pending": _encode(plan, node.name, answer)}
    return {"decision": _encode(plan, node.name, answer)}


async def answer_step(
    plan: Plan, node: str, pending: ApprovalRequest, response: Any
) -> dict[str, Any]:
    """A tool pause's answer: the token's decision, settled in the port's store, or none."""
    port = plan.env.executor.approval
    decision = await _settled(token_decision(response, pending), pending, port)
    if decision is None:
        decision = await _stored_decision(pending, port)
    if decision is None:
        return {"refused": True}
    return {"decision": _encode(plan, node, decision)}


async def _stored_decision(request: ApprovalRequest, port: Any) -> ApprovalDecision | None:
    """The no-token fallback: use the stored record of `request` by approval id, once.

    Never finds or creates a record by call id, so a resume that lost a race to another
    process leaves no orphan pending record. None: still pending, or the port decides.

    Raises:
        ApprovalAlreadyDecided: the request was spent here or its record consumed elsewhere.
    """
    await _refuse_spent(request, port)
    store = approval_store(port)
    if store is None or not isinstance(port, PreparedApproval):
        return None
    try:
        record = await store.get(request.approval_id)
        decision = await port.consume(record, request)
    except (ApprovalNotFound, ApprovalPending):
        return None
    except ToolUsageError as exc:
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "approval %s (%s) cannot be consumed; the port decides",
            request.approval_id,
            request.tool,
        )
        return None
    except ApprovalDenied as exc:
        logger.warning(
            "approval %s (%s) was used by a concurrent resume; this one is refused",
            request.approval_id,
            request.tool,
        )
        raise ApprovalAlreadyDecided(request.approval_id, status="consumed") from exc
    SPENT.spend((request.run_id, request.approval_id))
    return decision


async def _refuse_spent(request: ApprovalRequest, port: Any) -> None:
    """Before the port is asked again: refuse a request that was already answered.

    A concurrent (or repeated) resume of the same pause finds the request spent here or
    its store record consumed; asking the port again would make a fresh record and let the
    call run a second time.

    Raises:
        ApprovalAlreadyDecided: the request was spent in this process or consumed in the store.
    """
    SPENT.check((request.run_id, request.approval_id))
    store = approval_store(port)
    if store is None:
        return
    try:
        record = await store.get(request.approval_id)
    except ApprovalNotFound:
        return
    if record.consumed:
        logger.warning(
            "approval %s (%s) was already used; this resume is refused",
            request.approval_id,
            request.tool,
        )
        raise ApprovalAlreadyDecided(request.approval_id, status=record.status)


async def _settled(
    decision: ApprovalDecision | None, request: ApprovalRequest, port: Any
) -> ApprovalDecision | None:
    """`decision` once it is settled in the port's store and spent; None when refused."""
    if decision is None:
        return None
    try:
        await spend_once(request, decision, port)
    except (ApprovalAlreadyDecided, ToolUsageError) as exc:
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "approval %s (%s) could not be settled; the port decides",
            request.approval_id,
            request.tool,
        )
        return None
    return decision


async def visit_step(
    plan: Plan,
    node: NodeSpec,
    state: BaseModel,
    ctx: ToolContext,
    *,
    visit: int,
    approval: ApprovalDecision | None,
    entries: Sequence[AnswerEntry] = (),
    previous: Visit | None = None,
) -> dict[str, Any]:
    """One attempt of a node visit; a resumed attempt gets the answers and replays its calls."""
    outcome = await run_node(
        plan.env,
        node,
        state,
        ctx,
        visit=visit,
        approval=approval,
        answer=make_answer(entries) if entries else None,
        items=previous.items if previous is not None else None,
        journal=previous.journal if previous is not None else None,
    )
    if outcome.pending is None:
        labels = await _labels(plan, node.name, state, outcome)
        return {
            "updates": _encode(plan, node.name, list(outcome.updates)),
            "labels": None if labels is None else _encode(plan, node.name, labels),
        }
    _STORES.keep(outcome.pendings)
    return await _paused(plan, ctx, outcome)


async def _paused(plan: Plan, ctx: ToolContext, outcome: NodeOutcome) -> dict[str, Any]:
    """A waiting visit's output; one a resume could not continue is withdrawn and refused."""
    try:
        problems = unsealable_calls(outcome.journal)
        if problems:
            raise NotSealable(f"journal.{outcome.node}", "call", problems[0])
        return {
            "pending": encode(outcome.pending),
            "items": encode(dict(outcome.items)),
            "journal": outcome.journal,
            "pendings": encode([request for request, _ in outcome.pendings]),
            "visit_id": outcome.visit_id,
        }
    except NotSealable as exc:
        await withdraw(plan.flow, ctx, outcome.pendings)
        raise _not_sealable(plan, outcome.node, exc) from exc


async def _labels(
    plan: Plan, node: str, state: BaseModel, outcome: NodeOutcome
) -> list[Any] | None:
    """The labels the node's route picks, on the input state plus the node's own updates."""
    route = plan.graph.routes.get(node)
    if route is None:
        return None
    return await call_route(plan.flow, route, route_state(plan.graph, state, outcome))


async def withdraw_step(
    plan: Plan, ctx: ToolContext, waiting: Sequence[Visit], pending: ApprovalRequest
) -> dict[str, Any]:
    """Deny and consume the records a resume asks for afresh (local runner rule)."""
    default = approval_store(plan.env.executor.approval)
    outcomes = [visit.outcome() for visit in waiting]
    stale = [
        (request, store if store is not None else default)
        for request, store in superseded(outcomes, pending)
    ]
    if stale:
        await withdraw(plan.flow, ctx, stale, SUPERSEDED)
    return {"withdrawn": [request.approval_id for request, _ in stale]}


def read_visit(plan: Plan, node: str, run_id: str, data: Mapping[str, Any]) -> Visit:
    """A `visit_step` output as a `Visit`."""
    if "updates" in data:
        labels = data.get("labels")
        return Visit(
            node,
            run_id,
            updates=tuple(decoded(plan, node, data["updates"])),
            labels=None if labels is None else tuple(decoded(plan, node, labels)),
        )
    return Visit(
        node,
        run_id,
        pending=decoded(plan, node, data["pending"]),
        items=decoded(plan, node, data.get("items", {})),
        journal=dict(data.get("journal") or {}),
        pendings=tuple(decoded(plan, node, data.get("pendings", []))),
        visit_id=str(data.get("visit_id", "")),
    )
