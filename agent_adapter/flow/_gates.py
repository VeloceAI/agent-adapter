"""Pause gates (`pause=`) before a superstep, and checking what a resume may answer."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import (
    ApprovalPending,
    FlowError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._canon import Encoder
from agent_adapter.flow._invoke import describe_error
from agent_adapter.flow._nodes import node_call_id
from agent_adapter.flow._result import FlowCheckpoint
from agent_adapter.flow._seal import check_sealable
from agent_adapter.flow.graph import FlowGraph, NodeSpec
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

NO_PORT = "no approval port configured"


@dataclass(frozen=True)
class GateOutcome:
    """Granted decisions by node, or the gate that is pending or denied."""

    decisions: dict[str, ApprovalDecision] = field(default_factory=dict)
    pending: ApprovalRequest | None = None
    denied: str | None = None


async def check_gates(
    *,
    flow: str,
    nodes: Sequence[NodeSpec],
    state: BaseModel,
    ctx: ToolContext,
    visits: Mapping[str, int],
    port: ApprovalPort | None,
    granted: Mapping[str, ApprovalDecision],
    encoder: Encoder | None = None,
    scope: str = "",
) -> GateOutcome:
    """Ask each paused node's gate in order; stop at the first pending or denied one.

    Raises:
        FlowError: `state-not-sealable`: a pause here could not be sealed (checked before the
            port is asked, so no record is made); or the port or the payload failed.
    """
    decisions: dict[str, ApprovalDecision] = {}
    for node in nodes:
        if node.pause is False:
            continue
        decision = granted.get(node.name)
        if decision is None:
            visit = visits.get(node.name, 0) + 1
            request = _request(flow, node, state, ctx, visit, scope)
            check_sealable(flow, encoder, state=state, pending=request, decisions=decisions)
            answer = await _ask(flow, node, request, port)
            if isinstance(answer, ApprovalRequest):
                return GateOutcome(decisions, pending=answer)
            decision = answer
        if not decision.approved:
            logger.info("flow %s node %s denied: %s", flow, node.name, decision.reason)
            return GateOutcome(decisions, denied=node.name)
        decisions[node.name] = decision
    return GateOutcome(decisions)


def _request(
    flow: str, node: NodeSpec, state: BaseModel, ctx: ToolContext, visit: int, scope: str = ""
) -> ApprovalRequest:
    metadata: dict[str, Any] = {"flow": flow, "node": node.name, "visit": visit}
    if scope:
        metadata["scope"] = scope
    return ApprovalRequest(
        approval_id=new_id("apr_"),
        run_id=ctx.run_id,
        call_id=node_call_id(flow, node.name, visit, scope),
        tool=f"node:{flow}.{node.name}",
        arguments=_payload(flow, node, state),
        kind="node",
        hint=f"approve node {node.name!r} of flow {flow!r}",
        metadata=metadata,
        created_at=time.time(),
    )


async def _ask(
    flow: str, node: NodeSpec, request: ApprovalRequest, port: ApprovalPort | None
) -> ApprovalDecision | ApprovalRequest:
    if port is None:
        logger.warning("flow %s node %s needs approval but %s; denied", flow, node.name, NO_PORT)
        return ApprovalDecision(approved=False, reason=NO_PORT, decided_by="flow")
    try:
        return await port.request(request)
    except ApprovalPending as exc:
        return request.model_copy(update={"approval_id": exc.approval_id or request.approval_id})
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(
            logger, logging.WARNING, exc, "flow %s node %s: approval port failed", flow, node.name
        )
        raise FlowError(
            f"flow {flow!r} node {node.name!r}: approval failed: {describe_error(exc)}",
            flow=flow,
            node=node.name,
        ) from exc


def _payload(flow: str, node: NodeSpec, state: BaseModel) -> dict[str, Any]:
    if not callable(node.pause):
        return {}
    try:
        payload = node.pause(state)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        raise FlowError(
            f"flow {flow!r} node {node.name!r}: pause payload failed: {type(exc).__name__}",
            flow=flow,
            node=node.name,
        ) from exc
    if not isinstance(payload, Mapping):
        raise FlowError(
            f"flow {flow!r} node {node.name!r}: pause must return a mapping, "
            f"got {type(payload).__name__}",
            flow=flow,
            node=node.name,
        )
    return dict(payload)


def check_resume(graph: FlowGraph, checkpoint: Any, decision: Any) -> ApprovalRequest:
    """The pending request of a checkpoint that can resume `graph`.

    Raises:
        ToolUsageError: not a checkpoint or not a decision.
        FlowError: another flow or state model, nothing pending, or a node not in the frontier.
    """
    if not isinstance(checkpoint, FlowCheckpoint):
        raise ToolUsageError(f"expected a FlowCheckpoint, got {type(checkpoint).__name__}")
    if not isinstance(decision, ApprovalDecision):
        raise ToolUsageError(f"expected an ApprovalDecision, got {type(decision).__name__}")
    if checkpoint.flow != graph.name:
        raise FlowError(
            f"checkpoint of flow {checkpoint.flow!r} cannot resume flow {graph.name!r}",
            flow=graph.name,
        )
    if not isinstance(checkpoint.state, graph.state_model):
        raise FlowError(
            f"flow {graph.name!r}: checkpoint state is {type(checkpoint.state).__name__}, "
            f"not {graph.state_model.__name__}",
            flow=graph.name,
        )
    pending = checkpoint.pending
    if pending is None:
        raise FlowError(f"flow {graph.name!r}: the checkpoint has nothing pending", flow=graph.name)
    if pending.kind == "tool" and not pending.tool:
        logger.warning(
            "flow %s: pending approval %s names no tool", graph.name, pending.approval_id
        )
        raise FlowError(
            f"flow {graph.name!r}: pending approval {pending.approval_id!r} names no tool; "
            "resume refused (an answer must bind to one tool and its arguments)",
            flow=graph.name,
        )
    node = pending.metadata.get("node")
    if node not in checkpoint.frontier:
        raise FlowError(
            f"flow {graph.name!r}: pending node {node!r} is not in the checkpoint's frontier",
            flow=graph.name,
        )
    return pending
