"""Haystack: registry tools as Tools; a before_tool confirmation hook asks an approval port."""

import asyncio
from typing import Any

from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.haystack import confirmation_hook, haystack_tools

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
def build_agent(chat_generator: Any) -> Agent:
    hook = confirmation_hook(CallbackApproval(reviewer), registry)  # asks before send_invoice
    return Agent(
        chat_generator=chat_generator,
        tools=haystack_tools(registry, executor=Executor(registry)),
        hooks={"before_tool": [hook]},
    )


# --8<-- [end:wire]


@component
class ScriptedGenerator:
    """Stands in for a real chat generator: asks for one tool call, then says it is done."""

    def __init__(self, tool: str, args: dict[str, Any]) -> None:
        self.call = ToolCall(tool_name=tool, arguments=args, id="call_1")
        self.turns = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.turns += 1
        if self.turns == 1:
            return {"replies": [ChatMessage.from_assistant(tool_calls=[self.call])]}
        return {"replies": [ChatMessage.from_assistant("done")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.run(messages, tools, **kwargs)


async def main() -> None:
    calls: list[tuple[str, dict[str, Any]]] = [
        ("shipping_rate", {"region": "eu"}),
        ("send_invoice", {"customer_id": "cus_1024", "amount": 49.5}),
        ("send_invoice", {"customer_id": "cus_2048", "amount": 49.5}),
    ]
    for tool, args in calls:
        output = await build_agent(ScriptedGenerator(tool, args)).run_async(
            messages=[ChatMessage.from_user("go")]
        )
        (result,) = [r.result for m in output["messages"] for r in m.tool_call_results]
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
