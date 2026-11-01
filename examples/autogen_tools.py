"""Run registry tools in AutoGen 0.7: the workbench path, then an `AssistantAgent` (offline)."""

import asyncio
import json
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from autogen_core.tools import StaticWorkbench

from agent_adapter import Executor, Registry, ToolError
from agent_adapter.integrations.autogen import autogen_tools

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    if region == "mars":
        raise ToolError(f"no rate for {region!r}")
    return 18.0


async def main() -> None:
    # --8<-- [start:tools]
    executor = Executor(registry)  # hooks, approval, cache, ledger apply to every call
    tools = autogen_tools(registry, executor=executor)
    rate = tools[0]
    print(rate.name, await rate.run_json({"region": "us"}, CancellationToken(), call_id="c1"))
    # The workbench (what AssistantAgent uses) turns adapter errors into is_error results.
    failed = await StaticWorkbench(tools).call_tool(
        "shipping_rate", {"region": "mars"}, call_id="c2"
    )
    print(f"is_error={failed.is_error}", failed.result[0].content)
    # --8<-- [end:tools]
    # --8<-- [start:agent]
    agent = AssistantAgent("assistant", model_client=OfflineClient(), tools=tools)
    result = await agent.run(task="Shipping to the US?")
    print("answer:", result.messages[-1].to_text())
    # --8<-- [end:agent]


class OfflineClient(ChatCompletionClient):
    """Always asks for shipping_rate(region="us"); stands in for a real model client."""

    usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=False,
        family="unknown",
        structured_output=False,
    )

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        call = FunctionCall(
            id="call_1", name="shipping_rate", arguments=json.dumps({"region": "us"})
        )
        return CreateResult(
            finish_reason="function_calls", content=[call], usage=self.usage, cached=False
        )

    def create_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self.usage

    def total_usage(self) -> RequestUsage:
        return self.usage

    def count_tokens(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def remaining_tokens(self, *args: Any, **kwargs: Any) -> int:
        return 1000

    @property
    def capabilities(self) -> Any:
        return self.info

    @property
    def model_info(self) -> ModelInfo:
        return self.info


if __name__ == "__main__":
    asyncio.run(main())
