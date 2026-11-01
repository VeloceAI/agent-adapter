"""A `DeferredApproval` over a fresh memory store, with a reviewer helper, for tests."""

from __future__ import annotations

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.types import ApprovalDecision


class TwoPhaseApproval(DeferredApproval):
    """The real two-phase port (`prepare` / `consume`) on its own `MemoryApprovalStore`."""

    def __init__(self) -> None:
        super().__init__(MemoryApprovalStore())

    async def decide_all(self, decision: ApprovalDecision) -> None:
        """Decide every pending record (a reviewer working through the queue)."""
        for record in await self.store.list(status="pending"):
            await self.store.decide(record.approval_id, decision)
