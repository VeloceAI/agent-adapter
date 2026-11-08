"""Starter Google ADK agent: tools, approval, a budget and recorded runs. Run: python agent.py"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

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
from agent_adapter.integrations.adk import adk_tools, resolve_confirmations
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
record(executor, source="adk")  # see the runs with: agent-adapter ui
tools = adk_tools(registry, executor=executor)


async def ask(model: str, prompt: str) -> str:
    """Run the agent; ADK pauses each refund, `review` answers, then the run goes on."""
    from google.genai import types  # only the model path needs it

    agent = LlmAgent(name="shop_helper", model=model, tools=tools)
    sessions = InMemorySessionService()
    runner = Runner(app_name="shop", agent=agent, session_service=sessions)
    session = await sessions.create_session(app_name="shop", user_id="user")
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    answer = ""
    while message is not None:
        run = runner.run_async(user_id="user", session_id=session.id, new_message=message)
        events = [event async for event in run]
        for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                answer = event.content.parts[0].text or answer
        message = await resolve_confirmations(events, approval, registry)
    return answer


if __name__ == "__main__":
    model = os.environ.get("AGENT_MODEL")  # an ADK model name
    if model:
        print(asyncio.run(ask(model, "Refund 20 USD on order A1.")))
    else:
        print("tools:", [tool.name for tool in tools], "- set AGENT_MODEL to run the agent")
        print(run_sync(executor.run("refund_order", {"order_id": "A1", "amount": 20.0})))
