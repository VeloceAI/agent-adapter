"""Pause gates of compiled flows: the gate request and asking the executor's approval port."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from agent_adapter.approval.batch import PreparedApproval
from agent_adapter.approval.store import ApprovalRecord, ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    ApprovalPending,
    FlowError,
    log_failure,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import approval_store
from agent_adapter.flow.graph import NodeSpec
from agent_adapter.flow.visit import NO_PORT, describe_error, node_gate_request
from agent_adapter.integrations.langgraph.approval import LangGraphApproval

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateAnswer:
    """The port's answer: a decision now, or the request left pending (with its approval id)."""

    decision: ApprovalDecision | None = None
    pending: ApprovalRequest | None = None
    record: ApprovalRecord | None = None  # a peeked store record, consumed only by `spend`
    request: ApprovalRequest | None = None

    @property
    def approved(self) -> bool:
        """True when the gate may run."""
        return self.decision is not None and self.decision.approved


TASK_KEY = "langgraph_task"  # request metadata: the LangGraph task (namespace) of the visit


def gate_request(
    flow: str, node: NodeSpec, state: BaseModel, ctx: ToolContext, visit: int, task: str
) -> ApprovalRequest:
    """The gate's request as the local runner makes it, naming the LangGraph task it gates.

    Its approval id is derived from the run, the visit and the task, so a store record of
    this very gate is found again (and, once spent, never asked again).
    """
    request = node_gate_request(flow, node, state, ctx, visit)
    text = json.dumps([request.run_id, request.call_id, task], ensure_ascii=False)
    approval_id = "apr_" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    update = {"approval_id": approval_id, "metadata": {**request.metadata, TASK_KEY: task}}
    return request.model_copy(update=update)


async def spent_gate(
    store: ApprovalStore | None, request: ApprovalRequest
) -> ApprovalDecision | None:
    """The denial a spent record of this gate holds; None when there is no spent record.

    Raises:
        ApprovalAlreadyDecided: the gate's approval was already used (a grant is used once).
    """
    if store is None:
        return None
    try:
        record = await store.get(request.approval_id)
    except ApprovalNotFound:
        return None
    if not record.consumed:
        return None
    if record.decision is not None and not record.decision.approved:
        return record.decision
    logger.warning("gate %s of run %s was already granted and used", request.tool, request.run_id)
    raise ApprovalAlreadyDecided(request.approval_id, status="consumed")


async def ask_gate(
    flow: str, node: str, request: ApprovalRequest, port: ApprovalPort | None, *, peek: bool
) -> GateAnswer:
    """Ask `port` about a gate; `peek` (outside a node) never consumes a decided record.

    Raises:
        FlowError: the port failed.
        ApprovalAlreadyDecided: the gate's grant was already used (never asked again).
    """
    if port is None:
        logger.warning("flow %s node %s needs approval but %s; denied", flow, node, NO_PORT)
        return GateAnswer(ApprovalDecision(approved=False, reason=NO_PORT, decided_by="flow"))
    denied = await spent_gate(approval_store(port), request)
    if denied is not None:  # a spent denial stays a denial
        return GateAnswer(denied)
    try:
        if peek and isinstance(port, LangGraphApproval):
            return GateAnswer(pending=request)  # the gated node interrupts for it
        if peek and isinstance(port, PreparedApproval):
            record = await port.prepare(request)
            if record.decision is None:
                return GateAnswer(
                    pending=request.model_copy(update={"approval_id": record.approval_id})
                )
            # the node consumes an approval when it runs; `spend` consumes a denial
            return GateAnswer(record.decision, record=record, request=request)
        return GateAnswer(await port.request(request))
    except ApprovalDenied as exc:  # e.g. a concurrent resume used the record first
        return GateAnswer(ApprovalDecision(approved=False, reason=exc.reason, decided_by="flow"))
    except ApprovalPending as exc:
        approval_id = exc.approval_id or request.approval_id
        return GateAnswer(pending=request.model_copy(update={"approval_id": approval_id}))
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(
            logger, logging.WARNING, exc, "flow %s node %s: approval port failed", flow, node
        )
        raise FlowError(
            f"flow {flow!r} node {node!r}: approval failed: {describe_error(exc)}",
            flow=flow,
            node=node,
        ) from exc


async def spend(answer: GateAnswer, port: ApprovalPort | None) -> None:
    """Consume a peeked denial, so the record that ended the run is used once."""
    if answer.record is None or answer.request is None or not isinstance(port, PreparedApproval):
        return
    try:
        await port.consume(answer.record, answer.request)
    except ApprovalDenied:
        logger.debug("gate record %s was already consumed", answer.record.approval_id)
