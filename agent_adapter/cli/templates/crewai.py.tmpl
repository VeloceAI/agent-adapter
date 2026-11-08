"""Starter CrewAI crew: tools, approval, a budget and recorded runs. Run: python crew.py"""

import os

from crewai import Agent, Crew, Task

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
from agent_adapter.integrations.crewai import crewai_tools
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


# One per process. CrewAI has no approval gate of its own: the executor asks `review`.
# BudgetHook counts only what reaches the ledger: model calls made through
# executor.model(...). The framework's own model calls are not counted.
executor = Executor(
    registry, approval=CallbackApproval(review), hooks=[BudgetHook(1.0)], ledger=Ledger()
)
record(executor, source="crewai")  # see the runs with: agent-adapter ui
tools = crewai_tools(registry, executor=executor)


def ask(model: str, request: str) -> str:
    """Run a one-agent crew on `request`."""
    helper = Agent(
        role="shop helper",
        goal="Answer shop questions with the tools.",
        backstory="You help customers of an online shop.",
        tools=tools,
        llm=model,
    )
    task = Task(description=request, expected_output="A short answer.", agent=helper)
    return str(Crew(agents=[helper], tasks=[task]).kickoff())


if __name__ == "__main__":
    model = os.environ.get("AGENT_MODEL")  # a CrewAI model name, "provider/model"
    if model:
        print(ask(model, "Refund 20 USD on order A1."))
    else:
        print("tools:", [tool.name for tool in tools], "- set AGENT_MODEL to run the crew")
        print(run_sync(executor.run("refund_order", {"order_id": "A1", "amount": 20.0})))
