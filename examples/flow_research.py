"""The spec research flow: fan-out retrieval, a join, a capped loop and an approval pause."""

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_adapter import (
    ApprovalDecision,
    CallableModel,
    DeferredApproval,
    Executor,
    Flow,
    FlowRunner,
    LLMRequest,
    MemoryApprovalStore,
    Message,
    Registry,
    ToolContext,
)
from agent_adapter.errors import ConfigError, FlowError
from agent_adapter.flow import Join, Map, Merge, Route, node_model, node_prompt

registry = Registry()


@registry.tool(kind="retrieval", parallel_safe=True)
async def search_articles(query: str) -> list[dict[str, str]]:
    """Search the help-center articles."""
    return [{"source_id": f"article:{query}", "text": f"The {query} help article"}]


@registry.tool(kind="retrieval", parallel_safe=True)
async def search_tickets(query: str) -> list[dict[str, str]]:
    """Search past support tickets; tickets link the article, so its hit comes back too."""
    return [
        {"source_id": f"ticket:{query}", "text": f"A solved ticket on {query}"},
        {"source_id": f"article:{query}", "text": f"The {query} help article"},
    ]


# --8<-- [start:state]
class Hit(BaseModel):
    source_id: str
    text: str


class ResearchState(BaseModel):
    question: str
    plan_steps: list[dict[str, Any]] = Field(default_factory=list)
    evidence: Annotated[list[Hit], Merge.keyed(by="source_id")] = Field(default_factory=list)
    rounds: Annotated[int, Merge.sum] = 0
    needs_more: bool = False
    answer: str = ""


# --8<-- [end:state]


# --8<-- [start:nodes]
async def classify(state: ResearchState) -> dict[str, Any]:
    steps = [
        {"tool": name, "args": {"query": "returns"}}
        for name in ("search_articles", "search_tickets")
    ]
    return {"plan_steps": steps}


async def reason(state: ResearchState, ctx: ToolContext) -> dict[str, Any]:
    model = node_model(ctx)  # the runner's model, bound to this node
    if model is None:
        raise ConfigError("the reason node needs FlowRunner(model=...)")
    request = LLMRequest(
        messages=(
            Message(role="system", content=node_prompt(ctx) or ""),
            Message(role="user", content=f"{len(state.evidence)} sources"),
        ),
        stage="reason",
    )
    reply = await model.generate(request)
    first_round = state.rounds == 0
    update: dict[str, Any] = {"rounds": 1, "needs_more": first_round, "answer": reply.text}
    if first_round:  # ask for one more source, then loop back to retrieval
        update["plan_steps"] = [
            *state.plan_steps,
            {"tool": "search_articles", "args": {"query": "shipping"}},
        ]
    return update


def approve_step(state: ResearchState) -> dict[str, str]:
    return {"answer": f"{state.answer} (approved)"}


# --8<-- [end:nodes]


# --8<-- [start:flow]
flow = (
    Flow("research", state=ResearchState)
    .node("classify", classify)
    .node("retrieve", Map("plan_steps", tool_from="tool", into="evidence"), concurrency=8)
    .node("merge", Join(["retrieve"]))
    .node("reason", reason, prompt="Answer from the evidence only.")
    .node("approve", approve_step, pause=lambda s: {"answer": s.answer})
    .edge("classify", "retrieve")
    .edge("retrieve", "merge")
    .edge("merge", "reason")
    .route(
        "reason",
        Route(lambda s: "retrieve" if s.needs_more else "approve", targets=["retrieve", "approve"]),
        max_loops=3,
        on_exhausted="approve",
    )
    .end("approve")
)
# --8<-- [end:flow]


async def answer(request: LLMRequest) -> str:
    return f"returns answer from {request.messages[-1].content}"


async def main() -> None:
    # --8<-- [start:run]
    store = MemoryApprovalStore()
    runner = FlowRunner(
        Executor(registry), approval=DeferredApproval(store), model=CallableModel(answer)
    )
    paused = await runner.run(flow, {"question": "What does the returns policy allow?"})
    if paused.pending is None or paused.checkpoint is None:
        raise FlowError(f"expected the approval pause, got {paused.status}")
    print(paused.status, paused.pending.tool, paused.pending.arguments)

    done = await runner.resume(flow, paused.checkpoint, ApprovalDecision(approved=True))
    print(done.status, done.steps, dict(done.visits))
    print([hit.source_id for hit in done.state.evidence])
    print(done.state.answer)
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
