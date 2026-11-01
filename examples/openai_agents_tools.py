"""OpenAI Agents SDK: registry tools as FunctionTools, with an approval that edits arguments."""

import asyncio
import json
from typing import Any

from agents import (
    Agent,
    Model,
    ModelResponse,
    RunConfig,
    Runner,
    Tool,
    ToolCallOutputItem,
    Usage,
)
from agents.items import TResponseOutputItem
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
def refund(order_id: str, amount: float) -> str:
    """Refund an order (needs approval)."""
    return f"refunded {amount} on {order_id}"


class ScriptedModel(Model):
    """Stands in for a real model so the example runs offline: two calls, then an answer."""

    async def get_response(self, system_instructions: Any, input: Any, *a: Any, **k: Any) -> Any:
        if isinstance(input, list) and input[-1].get("type") == "function_call_output":
            text = ResponseOutputText(type="output_text", text="done", annotations=[])
            message = ResponseOutputMessage(
                id="m1", type="message", role="assistant", status="completed", content=[text]
            )
            return ModelResponse(output=[message], usage=Usage(), response_id=None)
        calls = [
            ("shipping_rate", {"region": "us"}, "c1"),
            ("refund", {"order_id": "A1", "amount": 500.0}, "c2"),
        ]
        output: list[TResponseOutputItem] = [
            ResponseFunctionToolCall(
                type="function_call", call_id=cid, name=name, arguments=json.dumps(arguments)
            )
            for name, arguments, cid in calls
        ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def review(request: ApprovalRequest) -> ApprovalDecision:
    """A reviewer who caps refunds at 50."""
    return ApprovalDecision(approved=True, edited_args={**request.arguments, "amount": 50.0})


async def main() -> None:
    # --8<-- [start:run]
    from agent_adapter.integrations.openai_agents import function_tools, resolve_interruptions

    tools: list[Tool] = list(function_tools(registry, executor=Executor()))
    agent = Agent(name="support", instructions="Help.", tools=tools, model=ScriptedModel())
    config = RunConfig(tracing_disabled=True)

    result = await Runner.run(agent, "Refund order A1", run_config=config)
    if result.interruptions:  # `refund` has confirm=True: the run paused for approval
        # One-use tokens for the approved calls; the SDK's own flag never approves alone.
        state = await resolve_interruptions(result, CallbackApproval(review))
        result = await Runner.run(agent, state, run_config=config)
    # --8<-- [end:run]
    outputs = [item.output for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    print(outputs)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
