"""Record a help-center search (fan-out, join, loop, cache, compression, approval) for the viewer.

Run it, then open the printed `agent-adapter ui` command. `--export PATH.html` also writes
one offline HTML file of the runs.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_adapter import (
    ApprovalPending,
    CallableModel,
    DeferredApproval,
    Executor,
    Flow,
    FlowRunner,
    Ledger,
    LLMRequest,
    MemoryCache,
    Message,
    Prices,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.errors import ConfigError
from agent_adapter.flow import Join, Map, Merge, Route, node_model
from agent_adapter.viewer import export, record

registry = Registry()


async def _hits(source: str, query: str) -> list[dict[str, str]]:
    await asyncio.sleep(0.02)  # a real search waits on the network; the calls overlap
    return [{"source_id": f"{source}:{query}", "title": f"{source.upper()}: {query}"}]


@registry.tool(kind="retrieval", parallel_safe=True, cache_ttl=300)
async def search_faq(query: str) -> list[dict[str, str]]:
    """Search the help-center FAQ."""
    return await _hits("faq", query)


@registry.tool(kind="retrieval", parallel_safe=True, cache_ttl=300)
async def search_forum(query: str) -> list[dict[str, str]]:
    """Search the community forum."""
    return await _hits("forum", query)


@registry.tool(kind="retrieval", parallel_safe=True, cache_ttl=300)
async def search_guides(query: str) -> list[dict[str, str]]:
    """Search the travel guides."""
    return await _hits("guide", query)


@registry.tool(confirm=True, effects="additive")
def escalate_ticket(ticket_id: str, note: str) -> str:
    """Hand a ticket to a human agent."""
    return f"ticket {ticket_id} escalated"


class Hit(BaseModel):
    source_id: str
    title: str


class HelpState(BaseModel):
    question: str
    plan_steps: list[dict[str, Any]] = Field(default_factory=list)
    hits: Annotated[list[Hit], Merge.keyed(by="source_id")] = Field(default_factory=list)
    rounds: Annotated[int, Merge.sum] = 0
    needs_more: bool = False
    answer: str = ""


PRICE = {"provider": "local", "input": 0.5, "output": 1.5}  # USD per million tokens, made up
SOURCES = ("search_faq", "search_forum", "search_guides")
HISTORY = "the traveller asked about baggage, seats and refunds for the trip " * 12


def count_tokens(messages: list[Message]) -> int:
    return sum(len(message.content) for message in messages) // 4


COMPRESS = CompressionPolicy(
    start_when_tokens_over=400,
    shrink_to_tokens=250,
    steps=[RemoveOldMessages(keep_first=0, keep_last=2)],
    count_tokens=count_tokens,
)


def plan(state: HelpState) -> dict[str, Any]:
    return {"plan_steps": [{"tool": name, "args": {"query": state.question}} for name in SOURCES]}


async def answer(state: HelpState, ctx: ToolContext) -> dict[str, Any]:
    model = node_model(ctx)
    if model is None:
        raise ConfigError("the answer node needs FlowRunner(model=...)")
    turns = [Message(role="user", content=f"turn {n}: {HISTORY}") for n in range(4)]
    turns.append(Message(role="user", content=f"{len(state.hits)} hits for {state.question}"))
    reply = await model.generate(LLMRequest(messages=tuple(turns), compression=COMPRESS))
    return {"rounds": 1, "needs_more": state.rounds == 0, "answer": reply.text}


def finish(state: HelpState) -> dict[str, Any]:
    return {"answer": state.answer.strip()}


flow = (
    Flow("help_search", state=HelpState)
    .node("plan", plan)
    .node("search", Map("plan_steps", tool_from="tool", into="hits"), concurrency=3)
    .node("merge", Join(["search"]))
    .node("answer", answer)
    .node("finish", finish)
    .edge("plan", "search")
    .edge("search", "merge")
    .edge("merge", "answer")
    .route(
        "answer",
        Route(lambda s: "again" if s.needs_more else "done", {"again": "search", "done": "finish"}),
        max_loops=1,
        on_exhausted="finish",
    )
    .end("finish")
)


def reply(request: LLMRequest) -> dict[str, Any]:
    """A fake model: it says how many messages reached it after compression."""
    text = f"answered from {len(request.messages)} messages"
    return {"text": text, "usage": {"input_tokens": 120, "output_tokens": 12}}


async def main(folder: Path) -> Path:
    runs = folder / "runs"
    approvals = folder / "approvals.db"
    prices = Prices.empty().merged({"version": "demo", "models": {"helper": PRICE}})
    executor = Executor(
        registry,
        cache=MemoryCache(),
        ledger=Ledger(prices=prices),
        approval=DeferredApproval.sqlite(approvals),
    )
    recorder = record(executor, folder=runs, source="example")  # every run is now recorded
    model = CallableModel(reply, provider="local", model="helper")
    runner = FlowRunner(executor, model=model, trace_state=True)  # fills the State tab
    result = await runner.run(flow, {"question": "baggage allowance"})
    print(result.status, result.steps, len(result.state.hits))

    async with executor.run_scope() as ctx:  # one parallel batch in its own run
        calls = [ToolCall(name=name, arguments={"query": "seat upgrade"}) for name in SOURCES]
        batch = await executor.run_many(calls, ctx)
    print([r.status for r in batch])

    try:
        await executor.run(escalate_ticket, {"ticket_id": "T-7", "note": "lost bag"})
    except ApprovalPending as pending:
        print("pending approval", pending.approval_id)
    recorder.flush()  # writes happen on a background thread; exit flushes them too
    print(f"agent-adapter ui {runs} --approvals {approvals} --reviewer you")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(".agent_adapter"), help="where")
    parser.add_argument("--export", type=Path, help="also write one offline HTML file here")
    args = parser.parse_args()
    written = asyncio.run(main(args.folder))
    if args.export:
        print("exported", export(written, args.export))
