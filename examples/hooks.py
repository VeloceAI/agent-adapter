"""Built-in hooks: redact logs, cap spend, block payloads, pace calls; plus a custom hook."""

import asyncio

from agent_adapter import (
    BudgetHook,
    GuardrailHook,
    Hook,
    LedgerEntry,
    LoggingHook,
    RateLimitHook,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.errors import BudgetExceeded, GuardrailBlocked
from agent_adapter.hooks import Rate, ShortCircuit, ToolInvocation, redact, run_before

registry = Registry()


@registry.tool(kind="search")
def search(query: str, api_key: str = "") -> list[str]:
    """Search the help center."""
    return [f"{query}: article HC-101"]


class Ledger:
    """A minimal LedgerPort; use agent_adapter.cost in real code."""

    def __init__(self) -> None:
        self.rows: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.rows.append(entry)

    def total_usd(self, run_id: str | None = None, *, thread_id: str | None = None) -> float:
        return sum(e.cost_usd or 0.0 for e in self.entries(run_id, thread_id=thread_id))

    def entries(
        self, run_id: str | None = None, *, thread_id: str | None = None
    ) -> list[LedgerEntry]:
        return [
            e
            for e in self.rows
            if (run_id is None or e.run_id == run_id)
            and (thread_id is None or e.thread_id == thread_id)
        ]


# --8<-- [start:builtins]
def no_secrets(stage: str, payload: object) -> bool | str:
    return "password" not in str(payload) or "looks like a password"


hooks = [
    LoggingHook(),  # one JSON line per start / finish / error, secrets redacted
    BudgetHook(0.50, scope="thread"),  # stop once the conversation spent $0.50
    GuardrailHook(no_secrets, on=["args", "result"]),
    RateLimitHook({"search": Rate(calls=5, per=1.0), "llm": Rate(calls=60, per=60.0)}),
]
# executor = Executor(registry, hooks=hooks)  # the executor calls them in this order
# --8<-- [end:builtins]


# --8<-- [start:custom]
class CachedAnswers(Hook):
    """Answer known queries without running the tool."""

    name = "cached_answers"

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        if inv.call.arguments.get("query") == "returns":
            return ShortCircuit(["returns: free within 30 days"])
        return None


# --8<-- [end:custom]


def invocation(ctx: ToolContext, **args: object) -> ToolInvocation:
    return ToolInvocation(registry.get("search"), ToolCall(name="search", arguments=args), ctx)


async def main() -> None:
    # --8<-- [start:redact]
    print(redact({"query": "returns", "api_key": "sk-1", "headers": {"Authorization": "Bearer x"}}))
    # --8<-- [end:redact]

    ledger = Ledger()
    ledger.record(
        LedgerEntry(run_id="run_1", call_id="c1", kind="llm", cost_usd=0.6, thread_id="t1")
    )
    ctx = ToolContext(run_id="run_2", thread_id="t1", ledger=ledger)
    # --8<-- [start:budget]
    try:
        await run_before(hooks, "before_tool", invocation(ctx, query="returns"))
    except BudgetExceeded as exc:
        print(exc.scope, exc.limit_usd, round(exc.spent_usd, 2))  # thread t1 already spent 0.6
    # --8<-- [end:budget]

    fresh = ToolContext(run_id="run_3", thread_id="t2", ledger=ledger)
    try:
        await run_before(hooks, "before_tool", invocation(fresh, query="password=hunter2"))
    except GuardrailBlocked as exc:
        print(exc.stage, exc.reason)
    print(await run_before([CachedAnswers()], "before_tool", invocation(fresh, query="returns")))


if __name__ == "__main__":
    asyncio.run(main())
