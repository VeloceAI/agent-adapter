"""Cookbook: retry a flaky tool with backoff, and keep bad input from being retried."""

import asyncio

from agent_adapter import Executor, Registry, ToolCall, ToolError
from agent_adapter.exec import Backoff
from agent_adapter.tracing import EventLog

registry = Registry()
failures = {"left": 0}  # how many calls fail before the upstream recovers


# --8<-- [start:tools]
class UpstreamBusy(ToolError):
    """The upstream said "try later": a ToolError the executor may retry."""

    retryable = True


@registry.tool(retries=3, timeout=2)  # up to 4 attempts, each at most 2 s
def fetch_invoice(number: str) -> dict[str, str]:
    """Fetch an invoice from the billing API."""
    if not number.startswith("INV-"):
        raise ToolError("invoice numbers start with INV-")  # not retryable: the model fixes it
    if failures["left"]:
        failures["left"] -= 1
        if failures["left"] % 2:
            raise ConnectionError("reset by peer")  # a foreign exception: retried
        raise UpstreamBusy("billing API busy")  # retryable = True: retried
    return {"number": number, "status": "paid"}


# --8<-- [end:tools]


async def no_wait(seconds: float) -> None:
    """Stands in for asyncio.sleep so the example runs at once."""


# --8<-- [start:executor]
events = EventLog()
executor = Executor(
    registry,
    backoff=Backoff(base_s=0.5, max_s=8.0, multiplier=2.0, jitter=0.25),
    sink=events,
    sleep=no_wait,  # leave out in real code; tests inject it
    rng=lambda: 0.0,  # no jitter, so the delays below are exact
)
# --8<-- [end:executor]


async def main() -> None:
    # --8<-- [start:run]
    failures["left"] = 3  # three failures, then success
    ok = await executor.run_call(ToolCall(name="fetch_invoice", arguments={"number": "INV-7"}))
    print(ok.status, ok.attempts, [e.data["delay_s"] for e in events.of_type("tool.retry")])

    bad = await executor.run_call(ToolCall(name="fetch_invoice", arguments={"number": "7"}))
    print(bad.error.kind if bad.error else "", bad.attempts)  # ToolError: one attempt only

    failures["left"] = 9  # never recovers
    down = await executor.run_call(ToolCall(name="fetch_invoice", arguments={"number": "INV-8"}))
    assert down.error is not None  # every attempt failed
    print(down.error.kind, down.attempts, down.error.retryable)
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
