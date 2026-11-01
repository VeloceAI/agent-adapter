"""Test tools offline: the plain function, its schema, the executor with fakes, a fake LLM."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from agent_adapter import (
    ApprovalDenied,
    ApprovalRequest,
    AutoDeny,
    CallableModel,
    CallbackApproval,
    Executor,
    LLMRequest,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.errors import ToolArgsInvalid
from agent_adapter.integrations.openai import chat_tool
from agent_adapter.tracing import EventLog

registry = Registry()
outages = {"left": 0}  # how many calls fail before the upstream recovers


# --8<-- [start:tools]
@registry.tool(retries=2, timeout=5)
def fetch_rate(currency: str) -> float:
    """Today's EUR rate for a currency."""
    if outages["left"]:
        outages["left"] -= 1
        raise ConnectionError("upstream down")  # retried by the executor
    return 0.92


@registry.tool(confirm=True)
def close_account(account_id: str) -> str:
    """Close a customer account."""
    return f"closed {account_id}"


# --8<-- [end:tools]


# --8<-- [start:plain]
async def test_schema_and_validation() -> None:
    spec = registry.get("fetch_rate")
    assert chat_tool(spec)["function"]["parameters"]["required"] == ["currency"]
    with pytest.raises(ToolArgsInvalid) as caught:
        await spec.invoke({"currency": 7})  # validated before the function runs
    assert caught.value.errors[0]["loc"] == ["currency"]


# --8<-- [end:plain]


# --8<-- [start:executor]
async def test_retries_without_waiting() -> None:
    outages["left"] = 2  # the first two attempts fail
    delays: list[float] = []

    async def no_sleep(seconds: float) -> None:  # injected: the backoff never really waits
        delays.append(seconds)

    events = EventLog()
    executor = Executor(registry, sink=events, sleep=no_sleep, rng=lambda: 0.5)
    result = await executor.run_call(ToolCall(name="fetch_rate", arguments={"currency": "USD"}))
    assert (result.value, result.attempts, len(delays)) == (0.92, 3, 2)
    assert len(events.of_type("tool.retry")) == 2


async def test_approval_paths() -> None:
    asked: list[ApprovalRequest] = []

    def reviewer(request: ApprovalRequest) -> bool:  # records the request, says yes
        asked.append(request)
        return True

    approve = CallbackApproval(reviewer)
    executor = Executor(registry, approval=approve)
    assert await executor.run("close_account", {"account_id": "a-1"}) == "closed a-1"
    assert asked[0].arguments == {"account_id": "a-1"}
    with pytest.raises(ApprovalDenied, match="denied by policy"):
        await executor.replace(approval=AutoDeny()).run("close_account", {"account_id": "a-2"})


# --8<-- [end:executor]


# --8<-- [start:fake_llm]
Reply = str | dict[str, Any]  # reply text, or LLMResponse fields


def scripted(*replies: Reply) -> Callable[[LLMRequest], Awaitable[Reply]]:
    """A fake model that returns the given replies in order."""
    queue = list(replies)

    async def reply(request: LLMRequest) -> Reply:
        return queue.pop(0)

    return reply


async def test_agent_loop_with_a_fake_model() -> None:
    outages["left"] = 0  # the upstream is healthy in this test
    call = ToolCall(name="fetch_rate", arguments={"currency": "USD"}, id="call_1")
    model = CallableModel(scripted({"tool_calls": [call]}, "The rate is 0.92 EUR."))
    question = (Message(role="user", content="USD rate?"),)
    result = await run_tool_loop(model, registry, question, executor=Executor(registry))
    assert result.finish == "stop"
    assert result.final.text == "The rate is 0.92 EUR."
    assert result.tool_results[0].value == 0.92


# --8<-- [end:fake_llm]


async def main() -> None:
    for test in (
        test_schema_and_validation,
        test_retries_without_waiting,
        test_approval_paths,
        test_agent_loop_with_a_fake_model,
    ):
        await test()
        print(test.__name__, "ok")


if __name__ == "__main__":
    asyncio.run(main())
