"""Strands Agents: registry tools as AgentTools; a before-tool-call hook asks an approval port."""

import asyncio
import json
from typing import Any

from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.models.model import Model

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.strands import strands_approval_hook, strands_tools

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
def build_agent(model: Model) -> Agent:
    tools = strands_tools(registry, executor=Executor(registry))
    agent = Agent(model=model, tools=tools, callback_handler=None)
    hook = strands_approval_hook(CallbackApproval(reviewer), registry)
    agent.hooks.add_callback(BeforeToolCallEvent, hook)  # asks before send_invoice runs
    return agent


# --8<-- [end:wire]


class ScriptedModel(Model):
    """Stands in for a real model: asks for one tool call, then says it is done."""

    def __init__(self, tool: str, args: dict[str, Any]) -> None:
        self.tool, self.args, self.turns = tool, args, 0

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
        yield

    async def stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.turns += 1
        yield {"messageStart": {"role": "assistant"}}
        if self.turns == 1:
            start = {"toolUse": {"toolUseId": "tu_1", "name": self.tool}}
            yield {"contentBlockStart": {"start": start}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.args)}}}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use" if self.turns == 1 else "end_turn"}}


def tool_result(agent: Agent) -> Any:
    (result,) = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    return result["status"], result["content"][0]


async def main() -> None:
    for tool, args in [
        ("shipping_rate", {"region": "eu"}),
        ("send_invoice", {"customer_id": "cus_1024", "amount": 49.5}),
        ("send_invoice", {"customer_id": "cus_2048", "amount": 49.5}),
    ]:
        agent = build_agent(ScriptedModel(tool, args))
        await agent.invoke_async("go")
        print(tool_result(agent))


if __name__ == "__main__":
    asyncio.run(main())
