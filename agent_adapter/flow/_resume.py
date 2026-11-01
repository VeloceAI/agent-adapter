"""Making a checkpoint safe to resume: check its answers, settle it, and spend it once."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Unpack

from agent_adapter.approval.store import ApprovalStore, request_digest
from agent_adapter.core._bounded import Bounded, Bounds
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._approve import approval_store, same_decision
from agent_adapter.flow._nodes import node_call_id
from agent_adapter.flow._result import FlowCheckpoint, ResumeEntry
from agent_adapter.flow._seal import TAMPERED

logger = logging.getLogger(__name__)


def resume_entry(pending: ApprovalRequest, decision: ApprovalDecision) -> ResumeEntry:
    """The answer for a paused tool approval: exact call id for `Call` / `Map` calls."""
    meta = pending.metadata
    node, visit = str(meta.get("node", "")), int(meta.get("visit", 0))
    prefix = node_call_id(str(meta.get("flow", "")), node, visit)
    return pending, decision, pending.call_id.startswith(f"{prefix}:")


async def check_answers(store: ApprovalStore, checkpoint: FlowCheckpoint) -> None:
    """Every answer found in the store must be a consumed record with that decision.

    An answer the store does not hold came from another port (the seal binds it).

    Raises:
        FlowError: an answer's record is not consumed or holds another request or decision.
    """
    for node, entries in checkpoint._answers.items():
        for request, decision, _exact in entries:
            try:
                record = await store.get(request.approval_id)
            except ApprovalNotFound:
                logger.debug("answer %s is not in the runner's store", request.approval_id)
                continue
            ok = (
                record.consumed
                and request_digest(record.request) == request_digest(request)
                and same_decision(record.decision, decision)
            )
            if not ok:
                logger.warning(
                    "flow %s node %s: answer %s has no matching store record",
                    checkpoint.flow,
                    node,
                    request.approval_id,
                )
                raise FlowError(
                    f"flow {checkpoint.flow!r}: {TAMPERED}: answer {request.approval_id!r} "
                    "does not match the approval store",
                    flow=checkpoint.flow,
                    node=node,
                )


async def settle_deferred(
    store: ApprovalStore, pending: ApprovalRequest, decision: ApprovalDecision
) -> None:
    """Record `decision` on the pending store record and consume it; a spent record refuses.

    Raises:
        ApprovalAlreadyDecided: the record was consumed, or decided otherwise.
        ToolUsageError: the record was made for another request.
    """
    try:
        record = await store.get(pending.approval_id)
    except ApprovalNotFound:
        logger.debug("approval %s is not in the store; nothing to settle", pending.approval_id)
        return
    if request_digest(record.request) != request_digest(pending):
        logger.warning("approval %s does not match the checkpoint: refusing", pending.approval_id)
        raise ToolUsageError(f"approval {pending.approval_id!r} was made for another request")
    if record.consumed:
        raise ApprovalAlreadyDecided(pending.approval_id, status=record.status)
    if record.status == "pending":
        try:
            record = await store.decide(pending.approval_id, decision)
        except ApprovalAlreadyDecided:
            record = await store.get(pending.approval_id)
    if not same_decision(record.decision, decision):
        logger.warning("approval %s was decided otherwise in the store", pending.approval_id)
        raise ApprovalAlreadyDecided(pending.approval_id, status=record.status)
    if not await store.consume(pending.approval_id):
        raise ApprovalAlreadyDecided(pending.approval_id, status="consumed")


class SpentCheckpoints(Bounded):
    """Process-wide memory of resumed approvals: a bounded LRU with a time to live.

    After an entry is evicted (size or age) only the approval store, if any, still refuses
    a second resume of that checkpoint.
    """

    def __init__(self, **bounds: Unpack[Bounds]) -> None:
        super().__init__(**bounds)
        self._spent: OrderedDict[tuple[str, str], float] = OrderedDict()

    def _expire(self) -> None:
        horizon = self._horizon()
        while self._spent and next(iter(self._spent.values())) < horizon:
            self._spent.popitem(last=False)

    def check(self, key: tuple[str, str]) -> None:
        """Raise when `key` was already spent.

        Raises:
            ApprovalAlreadyDecided: it was.
        """
        with self._lock:
            self._expire()
            if key in self._spent:
                logger.warning("approval %s was already resumed: refusing", key[1])
                raise ApprovalAlreadyDecided(key[1], status="resumed")

    def spend(self, key: tuple[str, str]) -> None:
        """Mark `key` spent; a racer that spent it first wins.

        Raises:
            ApprovalAlreadyDecided: it was already spent.
        """
        with self._lock:
            self._expire()
            if key in self._spent:
                raise ApprovalAlreadyDecided(key[1], status="resumed")
            self._spent[key] = self._clock()
            while len(self._spent) > self.max_entries:
                self._spent.popitem(last=False)

    def clear(self) -> None:
        """Forget everything (tests)."""
        with self._lock:
            self._spent.clear()


SPENT = SpentCheckpoints()


async def spend_once(
    pending: ApprovalRequest, decision: ApprovalDecision, port: ApprovalPort | None
) -> None:
    """Spend a checkpoint's pending approval: settle it in the store first, then remember it.

    Raises:
        ApprovalAlreadyDecided: the checkpoint was already resumed (here or via the store).
        ToolUsageError: the store record belongs to another request.
    """
    key = (pending.run_id, pending.approval_id)
    SPENT.check(key)
    store = approval_store(port)
    if store is not None:
        await settle_deferred(store, pending, decision)
    SPENT.spend(key)
