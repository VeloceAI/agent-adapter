"""Starter Agno agent: tools, approval, a budget and recorded runs. Run: python agent.py"""

import asyncio
import os

from agno.agent import Agent

from agent_adapter import (
    ApprovalDecision,
    ApprovalRequest,
    BudgetHook,
    CallbackApproval,
    Executor,
    Ledger,
    Registry,
    run_sync,
)
from agent_adapter.integrations.agno import agno_functions, resolve_requirements
from agent_adapter.viewer import record

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
def refund_order(order_id: str, amount: float) -> str:
    """Refund an order (needs approval)."""
    return f"refunded {amount:.2f} USD on {order_id}"


def review(request: ApprovalRequest) -> ApprovalDecision:
    """Replace with a real review (a person, a queue): approves refunds up to 50 USD."""
    approved = request.arguments["amount"] <= 50
    return ApprovalDecision(approved=approved, reason="" if approved else "over 50 USD")


approval = CallbackApproval(review)
# One per process. BudgetHook counts only what reaches the ledger: model calls made through
# executor.model(...). The framework's own model calls are not counted.
executor = Executor(registry, approval=approval, hooks=[BudgetHook(1.0)], ledger=Ledger())
record(executor, source="agno")  # see the runs with: agent-adapter ui
tools = agno_functions(registry, executor=executor)


async def ask(model: str, prompt: str) -> str:
    """Run the agent; each paused refund goes to `review`, then the run continues."""
    agent = Agent(model=model, tools=tools, telemetry=False)
    output = await agent.arun(prompt)
    while output.is_paused:
        await resolve_requirements(output, approval, registry)
        output = await agent.acontinue_run(run_response=output)
    return str(output.content)


if __name__ == "__main__":
    model = os.environ.get("AGENT_MODEL")  # an Agno model string, "provider:model"
    if model:
        print(asyncio.run(ask(model, "Refund 20 USD on order A1.")))
    else:
        print("tools:", [tool.name for tool in tools], "- set AGENT_MODEL to run the agent")
        print(run_sync(executor.run("refund_order", {"order_id": "A1", "amount": 20.0})))
