"""A small shop registry and helpers shared by the policy tests."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalRequest


def shop(log: list[str] | None = None) -> Registry:
    """refund (amount), lookup_order, wipe_orders, send_note (free text) and a code-confirm tool."""
    seen = log if log is not None else []
    reg = Registry()

    @reg.tool(tags=["money"])
    async def refund(order_id: str, amount: float) -> str:
        """Refund an order."""
        seen.append(f"refund {order_id} {amount}")
        return "refunded"

    @reg.tool(effects="read_only")
    async def lookup_order(order_id: str) -> dict[str, Any]:
        """Look up an order."""
        seen.append(f"lookup {order_id}")
        return {"order_id": order_id, "status": "shipped"}

    @reg.tool(effects="destructive")
    async def wipe_orders() -> str:
        """Delete every order."""
        seen.append("wipe")
        return "wiped"

    @reg.tool
    async def send_note(text: str) -> str:
        """Send a note to the shop."""
        seen.append(f"note {text}")
        return text

    @reg.tool(confirm=True)
    async def close_shop() -> str:
        """Close the shop (always approved by a person)."""
        seen.append("close")
        return "closed"

    return reg


class Approvals:
    """An approval port that records each request and answers `answer`."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> Any:
        from agent_adapter.core.types import ApprovalDecision

        self.asked.append(request)
        return ApprovalDecision(approved=self.answer, decided_by="test")


def caller(*, user: str = "", team: str = "", run_id: str = "run_t") -> ToolContext:
    """A context for one caller: `ctx.user_id` and `ctx.extras["team"]`."""
    extras = {"team": team} if team else {}
    return ToolContext(run_id=run_id, user_id=user, extras=extras)
