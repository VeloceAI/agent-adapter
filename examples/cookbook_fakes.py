"""Cookbook: give tools their clients through the context, so tests pass fakes instead."""

import asyncio
from typing import Protocol

from agent_adapter import Executor, Registry, ToolCall, ToolContext, ToolError
from agent_adapter.tracing import EventLog

registry = Registry()


# --8<-- [start:tool]
class Crm(Protocol):
    """What the tool needs from the CRM; the real client and the fake both fit."""

    async def find(self, email: str) -> dict[str, str] | None: ...


@registry.tool(retries=1)
async def customer_tier(email: str, ctx: ToolContext) -> str:
    """The customer's support tier."""
    crm: Crm = ctx.extras["crm"]  # injected by the caller, hidden from the model
    found = await crm.find(email)
    if found is None:
        raise ToolError(f"no customer {email}; ask for another address")
    return found["tier"]


# --8<-- [end:tool]


# --8<-- [start:fake]
class FakeCrm:
    """Answers from a dict; fails the first `outages` calls, like a flaky network."""

    def __init__(self, rows: dict[str, dict[str, str]], outages: int = 0) -> None:
        self.rows, self.outages, self.calls = rows, outages, 0

    async def find(self, email: str) -> dict[str, str] | None:
        self.calls += 1
        if self.outages:
            self.outages -= 1
            raise ConnectionError("crm down")
        return self.rows.get(email)


async def no_wait(seconds: float) -> None:
    """Stands in for asyncio.sleep: retries do not wait."""


# --8<-- [end:fake]


# --8<-- [start:tests]
async def test_tier_survives_one_outage() -> None:
    crm, events = FakeCrm({"a@example.com": {"tier": "gold"}}, outages=1), EventLog()
    executor = Executor(registry, sink=events, sleep=no_wait)
    ctx = executor.context(extras={"crm": crm})
    result = await executor.run_call(
        ToolCall(name="customer_tier", arguments={"email": "a@example.com"}), ctx
    )
    assert (result.value, result.attempts, crm.calls) == ("gold", 2, 2)
    assert [e.data["error_kind"] for e in events.of_type("tool.retry")] == ["internal-error"]


async def test_unknown_customer_tells_the_model_what_to_do() -> None:
    executor = Executor(registry, sleep=no_wait)
    ctx = executor.context(extras={"crm": FakeCrm({})})
    result = await executor.run_call(
        ToolCall(name="customer_tier", arguments={"email": "b@example.com"}), ctx
    )
    assert result.error is not None and result.error.kind == "tool-error"
    assert result.error.message == "no customer b@example.com; ask for another address"


# --8<-- [end:tests]


async def main() -> None:
    for test in (test_tier_survives_one_outage, test_unknown_customer_tells_the_model_what_to_do):
        await test()
        print(test.__name__, "ok")


if __name__ == "__main__":
    asyncio.run(main())
