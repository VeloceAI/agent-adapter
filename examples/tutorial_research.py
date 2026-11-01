"""Tutorial: a research agent from plain functions, parallel retrieval, a model loop, approval."""

import asyncio
import json
from typing import Any

from agent_adapter import (
    CallableModel,
    Decision,
    DeferredApproval,
    Executor,
    Ledger,
    LLMRequest,
    MemoryApprovalStore,
    Message,
    Prices,
    Registry,
    ToolCall,
    ToolContext,
    Usage,
    run_tool_loop,
)

registry = Registry()
PUBLISHED: list[str] = []  # what the risky tool did, so the tutorial can show it ran once


# --8<-- [start:tools]
@registry.tool(kind="retrieval", parallel_safe=True, timeout=10, retries=2)
async def search_web(query: str, limit: int = 3) -> list[str]:
    """Search the web for a query."""
    return [f"web: {query} #{n}" for n in range(1, limit + 1)]


@registry.tool(kind="retrieval", parallel_safe=True, timeout=10, retries=2)
async def search_papers(query: str) -> list[str]:
    """Search papers for a query."""
    return [f"paper: {query} survey"]


def legacy_news_lookup(topic: str, region: str = "us") -> list[str]:
    """Existing code you do not want to edit: news headlines for a topic."""
    return [f"news ({region}): {topic} rules change"]


registry.wrap(legacy_news_lookup, name="search_news", kind="retrieval", parallel_safe=True)


@registry.tool(confirm=True, effects="additive")
def publish_report(title: str, body: str) -> str:
    """Publish a report to the team wiki (needs approval)."""
    PUBLISHED.append(title)
    return f"published {title!r} ({len(body)} chars)"


# --8<-- [end:tools]


# --8<-- [start:retrieval]
def retrieval_calls(query: str, prefix: str = "") -> list[ToolCall]:
    """One call per source; call ids must be unique within a run."""
    return [
        ToolCall(name="search_web", arguments={"query": query, "limit": 2}, id=f"{prefix}web"),
        ToolCall(name="search_papers", arguments={"query": query}, id=f"{prefix}papers"),
        ToolCall(name="search_news", arguments={"topic": query}, id=f"{prefix}news"),
    ]


async def retrieve(executor: Executor, query: str, ctx: ToolContext) -> list[str]:
    """Run every source at once; a failed source is skipped, not fatal."""
    results = await executor.run_many(retrieval_calls(query), ctx, concurrency=3)
    return [hit for result in results if result.ok for hit in result.value]


# --8<-- [end:retrieval]


# --8<-- [start:model]
async def fake_model(request: LLMRequest) -> dict[str, Any]:
    """Stands in for a real model: search, then publish, then answer."""
    last = request.messages[-1]
    usage = Usage(input_tokens=2_000, output_tokens=200)
    if last.role == "user":
        calls = tuple(retrieval_calls(last.content, prefix="loop-"))
        return {"tool_calls": calls, "finish_reason": "tool_calls", "usage": usage}
    if last.name != "publish_report":
        hits = [m.content for m in request.messages if m.role == "tool"]
        args = {"title": "EV battery", "body": "\n".join(hits)}
        call = ToolCall(name="publish_report", arguments=args, id="publish")
        return {"tool_calls": (call,), "finish_reason": "tool_calls", "usage": usage}
    return {"text": f"Done: {last.content}", "usage": usage}


MODEL = CallableModel(fake_model, provider="host", model="research-v1")
PRICES = Prices.load(
    {
        "version": "demo",
        "models": {"research-v1": {"provider": "host", "input": 1.0, "output": 4.0}},
    }
)
# --8<-- [end:model]


async def main() -> None:
    # --8<-- [start:executor]
    store, ledger = MemoryApprovalStore(), Ledger(prices=PRICES)
    executor = Executor(registry, approval=DeferredApproval(store), ledger=ledger)
    ctx = executor.context(run_id="research-1")
    # --8<-- [end:executor]

    print(await retrieve(executor, "EV battery", ctx))

    # --8<-- [start:loop]
    question = (Message(role="user", content="EV battery"),)
    paused = await run_tool_loop(MODEL, registry, question, executor=executor, ctx=ctx)
    print(paused.finish, PUBLISHED)  # the model asked to publish: the call waits
    # --8<-- [end:loop]

    # --8<-- [start:approve]
    [waiting] = await store.list(run_id="research-1", status="pending")
    print(waiting.request.tool, sorted(waiting.request.arguments))  # what a reviewer sees
    edit = {**waiting.request.arguments, "title": "EV battery (draft)"}
    await store.decide(waiting.approval_id, Decision(approved=True, edited_args=edit))
    done = await run_tool_loop(MODEL, registry, paused.messages, executor=executor, ctx=ctx)
    print(done.finish, done.final.text, PUBLISHED)  # ran once, with the edited title
    # --8<-- [end:approve]

    # --8<-- [start:cost]
    summary = ledger.summary("research-1")
    print(f"${summary.total_usd:.4f}", {kind: s.calls for kind, s in summary.by_kind.items()})
    print(json.dumps({stage: round(s.cost_usd, 4) for stage, s in summary.by_stage.items()}))
    # --8<-- [end:cost]
    executor.end_run("research-1")


if __name__ == "__main__":
    asyncio.run(main())
