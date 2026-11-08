"""Policy file: one file sets access, approvals, guards and budgets for every surface."""

import asyncio
import tempfile
from pathlib import Path

from agent_adapter import (
    ApprovalDecision,
    ApprovalRequest,
    CallbackApproval,
    Executor,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.policy import load_policy

# --8<-- [start:file]
POLICY = """
environments = ["dev", "prod"]
default = { dev = "allow", prod = "deny" }

[[rules]]
tools = ["lookup_*", "send_note"]
allow = true

[[rules]]
tools = ["refund"]
allow = true
teams = ["support"]
confirm = "amount > 100"

[[rules]]
tools = ["wipe_orders"]
allow = false
reason = "never from an agent"

[[guards]]
guard = "pii"
tools = ["send_note"]
on = ["args"]

[[budgets]]
max_usd = 20
scope = "user"
per = "day"
"""
# --8<-- [end:file]

registry = Registry()


@registry.tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"{order_id}: shipped"


@registry.tool
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount:g} on {order_id}"


@registry.tool
def wipe_orders() -> str:
    """Delete every order."""
    return "wiped"


@registry.tool
def send_note(text: str) -> str:
    """Send a note to the shop."""
    return "sent"


def reviewer(request: ApprovalRequest) -> ApprovalDecision:
    print("approval asked for", request.tool)
    return ApprovalDecision(approved=True, decided_by="shift lead")


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "policy.toml"
        path.write_text(POLICY, encoding="utf-8")
        # --8<-- [start:apply]
        policy = load_policy(path, env="prod")  # or set AA_ENV=prod
        executor = policy.apply(Executor(registry, approval=CallbackApproval(reviewer)))
        # --8<-- [end:apply]
    support = ToolContext(user_id="ann", extras={"team": "support"})
    print(await executor.run("lookup_order", {"order_id": "o-1"}, support))
    print(await executor.run("refund", {"order_id": "o-1", "amount": 30}, support))
    print(await executor.run("refund", {"order_id": "o-2", "amount": 300}, support))
    for name, args in (
        ("wipe_orders", {}),
        ("refund", {"order_id": "o-3", "amount": 5}),
        ("send_note", {"text": "call ann@example.com"}),
    ):
        ctx = ToolContext(user_id="bob", extras={"team": "sales"})
        result = await executor.run_call(ToolCall(name=name, arguments=args), ctx)
        print(name, "->", result.error.kind if result.error else result.value)
    # --8<-- [start:explain]
    print(policy.explain("refund"))
    # --8<-- [end:explain]


if __name__ == "__main__":
    asyncio.run(main())
