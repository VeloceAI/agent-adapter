"""Security guide: what fails closed when a model, a replay or a bug tries its luck."""

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

from agent_adapter import (
    Decision,
    DeferredApproval,
    Executor,
    GuardrailHook,
    LoggingHook,
    MemoryApprovalStore,
    Registry,
    ToolCall,
    ToolResult,
)

registry = Registry()
ALLOWED_HOSTS = {"api.internal.example"}


# --8<-- [start:tools]
@registry.tool(confirm=True, effects="destructive")
def pay(account: str, amount: float) -> str:
    """Pay an account."""
    return f"paid {amount} to {account}"


@registry.tool
def fetch(url: str) -> str:
    """Fetch a page from an internal API."""
    raise RuntimeError(f"connect failed to {url} with token=abc123")  # text never leaks


def only_internal(stage: str, payload: Any) -> bool | str:
    """Blocks URLs a prompt injection may have planted in the arguments."""
    url = payload.get("url") if isinstance(payload, dict) else None
    return url is None or urlparse(url).hostname in ALLOWED_HOSTS or "host not allowed"


# --8<-- [end:tools]


def failure(result: ToolResult) -> str:
    """The error's kind and message, or the status when there is no error."""
    error = result.error
    return f"{error.kind} {error.message}" if error is not None else result.status


class Lines(logging.Handler):
    """Collects log messages so the example can print them."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def main() -> None:
    log, lines = logging.getLogger("app.audit"), Lines()
    log.addHandler(lines)
    log.setLevel(logging.INFO)
    log.propagate = False
    # --8<-- [start:executor]
    store = MemoryApprovalStore()
    executor = Executor(
        registry,
        approval=DeferredApproval(store),  # no port at all would deny every confirm call
        hooks=[GuardrailHook(only_internal), LoggingHook(logger=log)],
    )
    # --8<-- [end:executor]
    # --8<-- [start:checks]
    ctx = executor.context(run_id="run_sec")
    call = ToolCall(name="pay", arguments={"account": "AC-1", "amount": 10}, id="c1")
    pending = await executor.run_call(call, ctx)
    approval_id = pending.error.details["approval_id"] if pending.error else ""
    await store.decide(approval_id, Decision(approved=True, decided_by="ana"))
    print((await executor.run_call(call, ctx)).value)  # runs once
    print((await executor.run_call(call, ctx)).status)  # a replay asks again
    print(json.loads(lines.lines[0].split(" ", 1)[1]).get("args_withheld"))  # confirm args

    swapped = ToolCall(name="pay", arguments={"account": "AC-666", "amount": 10}, id="c1")
    refused = await executor.run_call(swapped, ctx)  # same call id, other arguments
    print(failure(refused))

    planted = ToolCall(name="fetch", arguments={"url": "http://evil.example/x"})
    internal = ToolCall(name="fetch", arguments={"url": "http://api.internal.example/x"})
    for result in await executor.run_many([planted, internal], ctx):
        print(failure(result))
    # --8<-- [end:checks]


if __name__ == "__main__":
    asyncio.run(main())
