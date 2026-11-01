"""CAMEL-AI: registry tools as FunctionTools; the executor's approval port decides confirm tools."""

import asyncio
import json
import time
from typing import Any

from camel.agents import ChatAgent
from camel.models.stub_model import StubModel
from camel.types import ModelType
from openai.types.chat import ChatCompletion

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.camel import camel_tools

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
def build_agent(model: Any) -> ChatAgent:
    # The executor's port decides send_invoice (confirm=True): CAMEL has no gate of its own.
    executor = Executor(registry, approval=CallbackApproval(reviewer))
    return ChatAgent(model=model, tools=camel_tools(registry, executor=executor))


async def ask(agent: ChatAgent, prompt: str) -> list[tuple[str, Any]]:
    response = await agent.astep(prompt)  # astep awaits the executor; step uses run_sync
    return [(call.tool_name, call.result) for call in response.info["tool_calls"]]


# --8<-- [end:wire]


class ScriptedModel(StubModel):
    """Stands in for a real model: asks for one tool call, then says it is done."""

    def __init__(self, tool: str, args: dict[str, Any]) -> None:
        super().__init__(ModelType.STUB)
        self.call: tuple[str, dict[str, Any]] | None = (tool, args)

    async def _arun(self, messages: Any, response_format: Any = None, tools: Any = None) -> Any:
        message: dict[str, Any] = {"role": "assistant", "content": "done"}
        if self.call is not None:
            name, args = self.call
            self.call = None
            function = {"name": name, "arguments": json.dumps(args)}
            calls = [{"id": "call_1", "type": "function", "function": function}]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
        choice = {"index": 0, "finish_reason": "stop", "message": message}
        usage = {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}
        return ChatCompletion.model_validate(
            {
                "id": "scripted",
                "model": "stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "choices": [choice],
                "usage": usage,
            }
        )


async def main() -> None:
    print(
        await ask(
            build_agent(ScriptedModel("shipping_rate", {"region": "eu"})), "Shipping to the EU?"
        )
    )
    for customer in ("cus_1024", "cus_2048"):
        agent = build_agent(
            ScriptedModel("send_invoice", {"customer_id": customer, "amount": 49.5})
        )
        print(await ask(agent, "Send the invoice"))


if __name__ == "__main__":
    asyncio.run(main())
