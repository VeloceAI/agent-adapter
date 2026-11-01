"""Run tools through the Executor: one call, a batch, a plan, non-tool work, approval."""

import asyncio

from agent_adapter import (
    ApprovalDecision,
    ApprovalRequest,
    Executor,
    Registry,
    RunLimits,
    Step,
    ToolCall,
    run_sync,
)

registry = Registry()


# --8<-- [start:setup]
@registry.tool(parallel_safe=True, timeout=5, retries=2, cache_ttl=300)
async def search(query: str, top_k: int = 3) -> list[str]:
    """Search the help-center articles."""
    return [f"{query} #{i}" for i in range(top_k)]


@registry.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


class AskOnce:
    """An ApprovalPort that approves and fixes the amount."""

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, edited_args={**request.arguments, "amount": 45.0})


executor = Executor(registry, approval=AskOnce(), limits=RunLimits(max_calls=50))
# --8<-- [end:setup]


async def main() -> None:
    # --8<-- [start:run]
    hits = await executor.run("search", {"query": "returns"})
    result = await executor.run_call(ToolCall(name="search", arguments={"query": 7}))
    print(hits[0], result.status, result.error.kind if result.error else "")
    # --8<-- [end:run]

    # --8<-- [start:run_many]
    calls = [
        ToolCall(name="search", arguments={"query": q, "top_k": 1}) for q in ("returns", "shipping")
    ]
    results = await executor.run_many(calls, concurrency=4)
    print([r.value for r in results])
    # --8<-- [end:run_many]

    # --8<-- [start:run_plan]
    plan = [
        Step(id="topics", tool="search", args={"query": "topics", "top_k": 1}),
        Step(
            id="detail",
            tool="search",
            args=lambda deps: {"query": deps["topics"].value[0], "top_k": 1},
            depends_on=["topics"],
        ),
    ]
    by_step = await executor.run_plan(plan)
    print(by_step["detail"].value)
    # --8<-- [end:run_plan]

    # --8<-- [start:parallel]
    async def summarise(text: str) -> str:
        return text.upper()

    summaries = await executor.parallel(
        lambda: summarise("a"), lambda: summarise("b"), timeout=30, retries=1
    )
    print(summaries)
    # --8<-- [end:parallel]

    # --8<-- [start:approval]
    print(await executor.run("send_invoice", {"customer_id": "cus_1024", "amount": 49.5}))
    # --8<-- [end:approval]


def sync_view() -> str:
    # --8<-- [start:sync]
    async def work() -> str:
        async with executor.run_scope(thread_id="chat-42") as ctx:
            hits: list[str] = await executor.run("search", {"query": "warranty", "top_k": 1}, ctx)
            return hits[0]

    return run_sync(work, timeout=30)
    # --8<-- [end:sync]


if __name__ == "__main__":
    asyncio.run(main())
    print(sync_view())
