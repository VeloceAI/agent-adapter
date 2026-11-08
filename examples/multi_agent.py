"""Multi-agent: a supervisor delegates to two specialists, a handoff, and a flow that routes."""

import asyncio

from pydantic import BaseModel

from agent_adapter import (
    CallableModel,
    Executor,
    Flow,
    FlowRunner,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    Registry,
    ToolCall,
    ToolContext,
    run_tool_loop,
)
from agent_adapter.exec import HandoffModel, LoopAgent, agent_tool, handoff
from agent_adapter.flow import Call, Route

tools = Registry()


@tools.tool
def ticket_price(kind: str) -> float:
    """Ticket price in EUR."""
    return {"adult": 18.0, "child": 9.0}.get(kind, 18.0)


@tools.tool
def find_artwork(title: str) -> str:
    """Where an artwork hangs."""
    return f"{title}: room 12, east wing"


def scripted(tool: str, args: dict, answer: str) -> CallableModel:
    """Stands in for a real model: calls one tool, then answers from its result."""

    def reply(request: LLMRequest) -> LLMResponse:
        last = request.messages[-1]
        if last.role == "tool":
            text = answer.format(result=last.content)
            return LLMResponse(
                provider="host",
                model="demo",
                text=text,
                message=Message(role="assistant", content=text),
            )
        call = ToolCall(name=tool, arguments=args, id=f"call_{tool}")
        return LLMResponse(
            provider="host",
            model="demo",
            tool_calls=(call,),
            finish_reason="tool_calls",
            message=Message(role="assistant", tool_calls=(call,)),
        )

    return CallableModel(reply, provider="host", model="demo")


# --8<-- [start:specialists]
supervisor_tools = Registry()
PRICES = Prices.load(
    {"version": "demo", "models": {"demo": {"provider": "host", "input": 2.0, "output": 8.0}}}
)
executor = Executor(supervisor_tools, ledger=Ledger(prices=PRICES))

tickets = LoopAgent(
    scripted("ticket_price", {"kind": "child"}, "A child ticket is {result} EUR."),
    tools.subset(names=["ticket_price"]),
    instructions="You answer ticket questions.",
)
collections = LoopAgent(
    scripted("find_artwork", {"title": "Sunflowers"}, "{result}."),
    tools.subset(names=["find_artwork"]),
    instructions="You know where every artwork hangs.",
)
supervisor_tools.add(
    agent_tool(tickets, name="tickets", description="Ticket desk.", executor=executor)
)
supervisor_tools.add(
    agent_tool(collections, name="collections", description="Collections desk.", executor=executor)
)
# --8<-- [end:specialists]


async def supervisor() -> None:
    # --8<-- [start:supervisor]
    model = scripted("tickets", {"task": "price of a child ticket"}, "Supervisor: {result}")
    ctx = executor.context()
    question = [Message(role="user", content="How much is a child ticket?")]
    result = await run_tool_loop(model, supervisor_tools, question, executor=executor, ctx=ctx)
    print(result.final.text)
    entries = executor.ledger.entries(ctx.run_id)  # the specialist's calls are in this run too
    print(sorted(f"{e.kind}:{e.stage}" for e in entries))
    # --8<-- [end:supervisor]


async def front_desk() -> None:
    # --8<-- [start:handoff]
    desk = Registry()
    desk.add(
        handoff(
            collections,
            name="to_collections",
            description="Pass to collections.",
            executor=executor,
        )
    )
    model = HandoffModel(scripted("to_collections", {"task": "Sunflowers?"}, "unused"), desk)
    result = await run_tool_loop(
        model,
        desk,
        [Message(role="user", content="Where are the Sunflowers?")],
        executor=executor,
    )
    print(result.final.text)
    # --8<-- [end:handoff]


# --8<-- [start:flow]
class Visit(BaseModel):
    question: str
    answer: str = ""


def answer_desk(visit: Visit) -> str:
    return "tickets" if "ticket" in visit.question else "collections"


flow = (
    Flow("help", state=Visit)
    .node("triage", lambda visit: {})
    .node("tickets", Call("tickets", args=lambda v: {"task": v.question}, into="answer"))
    .node("collections", Call("collections", args=lambda v: {"task": v.question}, into="answer"))
    .route("triage", Route(answer_desk, targets=["tickets", "collections"]))
    .end("tickets", "collections")
)
# --8<-- [end:flow]


async def routed() -> None:
    result = await FlowRunner(executor).run(flow, {"question": "Where is the Sunflowers?"})
    print(result.state.answer)


async def plain_function() -> None:
    async def poet(task: str, ctx: ToolContext) -> str:
        return f"A short poem about {task}."

    spec = agent_tool(poet, name="poet", description="Writes poems.")
    print(await Executor().run_spec(spec, {"task": "the east wing"}))


async def main() -> None:
    await supervisor()
    await front_desk()
    await routed()
    await plain_function()


if __name__ == "__main__":
    asyncio.run(main())
