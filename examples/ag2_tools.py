"""AG2: registry tools as FunctionTools; an approval gate in AG2's tool middleware asks one port."""

import asyncio
import json
from typing import Any

from ag2 import Agent
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.testing import TestConfig

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.ag2 import ag2_tools

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


def reviewer(request: ApprovalRequest) -> ApprovalDecision:
    """A stand-in for a human: approves invoices for one customer only."""
    approved = request.arguments["customer_id"] == "cus_1024"
    return ApprovalDecision(approved=approved, reason="" if approved else "unknown customer")


# --8<-- [start:wire]
port = CallbackApproval(reviewer)


def build_agent(config: Any) -> Agent:
    # approval=port: a gate in each tool's middleware asks the port about confirm tools and
    # hands the approved call a one-use token, so the executor does not ask again.
    tools = ag2_tools(registry, executor=Executor(registry), approval=port)
    return Agent("shop", config=config, tools=tools)


async def ask(agent: Agent, prompt: str) -> list[tuple[str | None, Any]]:
    reply = await agent.ask(prompt)
    events = await reply.history.get_events()
    batches = [e for e in events if isinstance(e, ToolResultsEvent)]  # one per model turn
    return [(r.name, r.result.parts[0]) for batch in batches for r in batch.results]


# --8<-- [end:wire]


def scripted(tool: str, args: dict[str, Any]) -> TestConfig:
    """Stands in for a real model: asks for one tool call, then says it is done."""
    call = ToolCallEvent(name=tool, arguments=json.dumps(args), id="call_1")
    return TestConfig(call, "done")


def shown(results: list[tuple[str | None, Any]]) -> list[tuple[str | None, Any]]:
    """What the model sees: text, or the data of a data part."""
    return [(name, getattr(part, "content", getattr(part, "data", part))) for name, part in results]


async def main() -> None:
    print(shown(await ask(build_agent(scripted("shipping_rate", {"region": "eu"})), "EU?")))
    for customer in ("cus_1024", "cus_2048"):
        agent = build_agent(scripted("send_invoice", {"customer_id": customer, "amount": 49.5}))
        print(shown(await ask(agent, "Send the invoice")))


if __name__ == "__main__":
    asyncio.run(main())
