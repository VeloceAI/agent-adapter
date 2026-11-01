"""Approvals a store-backed port decides for a paused flow node: read once, consumed once."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import NoReturn

from agent_adapter.approval.store import ApprovalStore, request_digest
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import same_decision
from agent_adapter.flow.visit import TAMPERED
from agent_adapter.integrations.langgraph._durable import Surfaced
from agent_adapter.integrations.langgraph._spent import CLAIMS, ClaimKey, claim, note_consumed

logger = logging.getLogger(__name__)


class StoreDecisions:
    """The store's decisions for one node visit; a pending record pauses the task."""

    def __init__(
        self,
        store: ApprovalStore,
        *,
        flow: str,
        run_id: str,
        namespace: str,
        gate_claims: list[tuple[str, ...]],
        pause: Callable[[ApprovalRequest], NoReturn],
    ) -> None:
        self.store = store
        self.flow, self.run_id, self.namespace = flow, run_id, namespace
        self.gate_claims = gate_claims  # shared with the visit: started when the gate runs
        self.pause = pause

    async def decide(self, surfaced: Surfaced) -> tuple[ApprovalDecision, ClaimKey] | None:
        """The store's decision, consumed once; None when the store does not hold it.

        Raises:
            GraphInterrupt: still pending in the store (the task pauses again).
            ApprovalAlreadyDecided: another execution used the record.
        """
        request = surfaced.request
        try:
            record = await self.store.get(request.approval_id)
        except ApprovalNotFound:
            return None
        stored = record.request
        same = (stored.run_id, stored.call_id) == (request.run_id, request.call_id)
        if not same or request_digest(stored) != request_digest(request):
            raise ToolUsageError(f"approval {request.approval_id!r} is for another request")
        unused = CLAIMS.unused(self.run_id, self.namespace, "store", request.approval_id)
        if surfaced.via == "store" or (record.consumed and unused):
            # consumed by this task before a pause, or before a failure ahead of the call
            kept = surfaced.decision if surfaced.via == "store" else record.decision
            if kept is None or not record.consumed or not same_decision(record.decision, kept):
                raise FlowError(f"flow {self.flow!r}: {TAMPERED}: approval record changed")
            claimed = self._claim(surfaced)
            note_consumed(self.run_id, self.namespace, "store", request.approval_id)
            surfaced.via, surfaced.decision = "store", kept
            return kept, claimed
        if record.consumed:
            logger.warning("flow %s: approval %s was already used", self.flow, request.approval_id)
            raise ApprovalAlreadyDecided(request.approval_id, status="consumed")
        if record.decision is None:
            self.pause(request)
        claimed = self._claim(surfaced)
        if not await self.store.consume(request.approval_id):
            raise ApprovalAlreadyDecided(request.approval_id, status="consumed")
        note_consumed(self.run_id, self.namespace, "store", request.approval_id)
        surfaced.via, surfaced.decision = "store", record.decision
        return record.decision, claimed

    def _claim(self, surfaced: Surfaced) -> ClaimKey:
        approval_id = surfaced.request.approval_id
        claimed = claim(self.run_id, self.namespace, "store", approval_id, started=False)
        if surfaced.request.kind == "node":
            self.gate_claims.append(("store", approval_id))
        return claimed
