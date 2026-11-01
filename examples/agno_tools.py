"""Agno: registry tools as Functions; one approval port resolves Agno's confirmation pauses."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.agno import agno_functions, resolve_requirements

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


def build_agent(model: Model) -> Agent:
    tools = agno_functions(registry, executor=Executor(registry, approval=port))
    # send_invoice has confirm=True, so it is exported with requires_confirmation=True
    return Agent(model=model, tools=tools, telemetry=False)


async def run_with_approvals(agent: Agent, prompt: str) -> Any:
    output = await agent.arun(prompt)
    while output.is_paused:  # Agno paused: a confirm tool wants to run
        await resolve_requirements(output, port, registry)  # confirm() / reject() from the port
        output = await agent.acontinue_run(run_response=output)
    return [(tool.tool_name, tool.confirmed, tool.result) for tool in output.tools or []]


# --8<-- [end:wire]


@dataclass
class ScriptedModel(Model):
    """Stands in for a real model: asks for one tool call, then says it is done."""

    id: str = "scripted"
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    turns: int = 0

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.turns += 1
        if self.turns > 1:
            return ModelResponse(role="assistant", content="done")
        call = {"name": self.tool, "arguments": json.dumps(self.args)}
        calls = [{"id": "call_1", "type": "function", "function": call}]
        return ModelResponse(role="assistant", content="", tool_calls=calls)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> Any:
        return response

    def _parse_provider_response_delta(self, response: Any) -> Any:
        return response


async def main() -> None:
    plain = build_agent(ScriptedModel(tool="shipping_rate", args={"region": "eu"}))
    print(await run_with_approvals(plain, "Shipping to the EU?"))
    for customer in ("cus_1024", "cus_2048"):
        agent = build_agent(
            ScriptedModel(tool="send_invoice", args={"customer_id": customer, "amount": 49.5})
        )
        print(await run_with_approvals(agent, "Send the invoice"))


if __name__ == "__main__":
    asyncio.run(main())
