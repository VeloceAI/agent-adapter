"""Microsoft Agent Framework: registry tools as FunctionTools; one approval port decides."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.agent_framework import agent_framework_tools, resolve_approvals

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
port = CallbackApproval(reviewer)  # a DeferredApproval store makes the run-time check silent


def build_agent(client: BaseChatClient) -> Agent:
    tools = agent_framework_tools(registry, executor=Executor(registry, approval=port))
    # send_invoice has confirm=True, so it is exported with approval_mode="always_require"
    return Agent(client=client, tools=tools)


async def run_with_approvals(agent: Agent, prompt: str) -> str:
    session = agent.create_session()
    response = await agent.run(prompt, session=session)
    while response.user_input_requests:  # MAF paused: a confirm tool wants to run
        answers = await resolve_approvals(response, port, registry, session=session)
        response = await agent.run(Message(role="user", contents=answers), session=session)
    return response.text


# --8<-- [end:wire]


class ScriptedClient(FunctionInvocationLayer, BaseChatClient):
    """Stands in for a real chat client: asks for one tool call, then answers with its result."""

    def __init__(self, name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self.call = Content.from_function_call(call_id="call_1", name=name, arguments=args)

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        results = [c.result for m in messages for c in m.contents if c.type == "function_result"]
        if results:
            return ChatResponse(messages=[Message(role="assistant", contents=[str(results[-1])])])
        return ChatResponse(messages=[Message(role="assistant", contents=[self.call])])


async def main() -> None:
    plain = build_agent(ScriptedClient("shipping_rate", {"region": "us"}))
    print(await run_with_approvals(plain, "Shipping to the US?"))
    for customer in ("cus_1024", "cus_2048"):
        agent = build_agent(
            ScriptedClient("send_invoice", {"customer_id": customer, "amount": 49.5})
        )
        print(await run_with_approvals(agent, "Send the invoice"))


if __name__ == "__main__":
    asyncio.run(main())
