"""Pydantic AI: registry tools as Tools, with deferred approval that edits arguments."""

import asyncio
import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_adapter import (
    ApprovalDecision,
    ApprovalRequest,
    CallbackApproval,
    Executor,
    Registry,
    ToolCall,
    ToolContext,
)

registry = Registry()


def large(call: ToolCall, ctx: ToolContext) -> bool:
    return bool(call.arguments["amount"] > 100)  # small refunds run without review


@registry.tool(confirm=large)
def refund(order_id: str, amount: float) -> str:
    """Refund an order (large refunds need approval)."""
    return f"refunded {amount} on {order_id}"


def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stands in for a real model so the example runs offline: two refunds, then an answer."""
    if any(part.part_kind == "tool-return" for part in messages[-1].parts):
        return ModelResponse(parts=[TextPart("done")])
    return ModelResponse(
        parts=[
            ToolCallPart("refund", {"order_id": "A1", "amount": 20.0}, tool_call_id="c1"),
            ToolCallPart("refund", {"order_id": "B2", "amount": 900.0}, tool_call_id="c2"),
        ]
    )


def review(request: ApprovalRequest) -> ApprovalDecision:
    """A reviewer who caps refunds at 100."""
    return ApprovalDecision(approved=True, edited_args={**request.arguments, "amount": 100.0})


async def main() -> None:
    # --8<-- [start:run]
    from agent_adapter.integrations.pydantic_ai import pydantic_ai_tools, resolve_deferred

    agent = Agent(
        FunctionModel(scripted),
        tools=pydantic_ai_tools(registry, executor=Executor()),
        output_type=[str, DeferredToolRequests],  # lets the run stop for approval
    )
    result = await agent.run("Refund A1 and B2")
    if isinstance(result.output, DeferredToolRequests):  # B2 is over 100: paused
        port = CallbackApproval(review)
        results = await resolve_deferred(result.output, port, run_id=result.run_id)
        result = await agent.run(
            message_history=result.all_messages(), deferred_tool_results=results
        )
    # --8<-- [end:run]
    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if part.part_kind == "tool-return"
    ]
    print(returns)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
