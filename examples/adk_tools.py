"""Google ADK: registry tools as BaseTools; your approval port answers ADK's confirmations."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.adk import adk_tools, resolve_confirmations

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


def build_agent(model: BaseLlm) -> LlmAgent:
    tools = adk_tools(registry, executor=Executor(registry, approval=port))
    # send_invoice has confirm=True: ADK asks for a ToolConfirmation before it runs
    return LlmAgent(name="shop_helper", model=model, tools=tools)


async def run_with_approvals(runner: Runner, session_id: str, prompt: str) -> list[Any]:
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    results: list[Any] = []
    while message is not None:
        run = runner.run_async(user_id="u", session_id=session_id, new_message=message)
        events = [event async for event in run]
        results += [(r.name, r.response) for e in events for r in e.get_function_responses()]
        # None when nothing waits; else the answers, with one-use tokens for approved calls
        message = await resolve_confirmations(events, port, registry)
    return results


# --8<-- [end:wire]


class ScriptedLlm(BaseLlm):
    """Stands in for Gemini: asks for one tool call, then says it is done."""

    model: str = "scripted"
    call: dict[str, Any] = Field(default_factory=dict)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        answered = any(p.function_response for c in llm_request.contents for p in c.parts or ())
        if answered:
            part = types.Part(text="done")
        else:
            part = types.Part(function_call=types.FunctionCall(id="call_1", **self.call))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def main() -> None:
    calls = [
        {"name": "shipping_rate", "args": {"region": "eu"}},
        {"name": "send_invoice", "args": {"customer_id": "cus_1024", "amount": 49.5}},
        {"name": "send_invoice", "args": {"customer_id": "cus_2048", "amount": 49.5}},
    ]
    for call in calls:
        sessions = InMemorySessionService()
        agent = build_agent(ScriptedLlm(call=call))
        runner = Runner(app_name="shop", agent=agent, session_service=sessions)
        session = await sessions.create_session(app_name="shop", user_id="u")
        results = await run_with_approvals(runner, session.id, "help")
        print(results[-1])


if __name__ == "__main__":
    asyncio.run(main())
