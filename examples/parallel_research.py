"""Parallel research on a shared board: three agents fan out and claim work, a writer fans in."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
    Usage,
    run_tool_loop,
)
from agent_adapter.board import board_of, board_tools
from agent_adapter.exec import LoopAgent, agent_tool
from agent_adapter.tracing import EventLog
from agent_adapter.viewer import analyse_run

FACTS = {"prices": "prices: up 4% this year", "weather": "weather: sunny, 21 C"}
SEARCHES: list[str] = []

tools = Registry()
tools.add_many(board_tools())


@tools.tool
async def search(topic: str) -> str:
    """Search the web for a topic."""
    SEARCHES.append(topic)
    await asyncio.sleep(0.01)
    return FACTS[topic]


def tokens(request: LLMRequest) -> Usage:
    """About 4 characters a token, so the token numbers below have something to count."""
    sent = sum(len(m.content) for m in request.messages)
    return Usage(input_tokens=sent // 4, output_tokens=10)


def say(request: LLMRequest, text: str = "", call: ToolCall | None = None) -> LLMResponse:
    calls = (call,) if call else ()
    return LLMResponse(
        provider="host",
        model="demo",
        text=text,
        tool_calls=calls,
        usage=tokens(request),
        finish_reason="tool_calls" if calls else "stop",
        message=Message(role="assistant", content=text, tool_calls=calls),
    )


def researcher(name: str) -> CallableModel:
    """Stands in for a real model: claim the topic, search it, post the finding."""

    def reply(request: LLMRequest) -> LLMResponse:
        task = next(m.content for m in request.messages if m.content.startswith("research "))
        topic = task.split()[-1]
        last = next(m for m in reversed(request.messages) if "<shared-board>" not in m.content)
        if last.role == "user":
            claim = {"key": f"search:{topic}"}
            return say(request, call=ToolCall(name="board_claim", arguments=claim, id=f"{name}_1"))
        if last.name == "board_claim":
            if "it is yours" not in last.content:
                return say(request, f"skipped {topic}: {last.content.split()[0]} has it")
            return say(
                request, call=ToolCall(name="search", arguments={"topic": topic}, id=f"{name}_2")
            )
        if last.name == "search":
            post = {"kind": "finding", "text": last.content, "key": topic}
            return say(request, call=ToolCall(name="board_post", arguments=post, id=f"{name}_3"))
        return say(request, f"found {topic}")

    return CallableModel(reply, model=name)


def writer(request: LLMRequest) -> LLMResponse:
    """Writes from the board note the loop puts before each turn (marked as data)."""
    note = next(m.content for m in request.messages if "<shared-board>" in m.content)
    found = sorted(line.split("] ", 1)[1] for line in note.splitlines() if "[finding:" in line)
    return say(request, "Report: " + "; ".join(found))


# --8<-- [start:agents]
sink = EventLog()
executor = Executor(tools, sink=sink)
team = Registry()
for name in ("alpha", "beta", "gamma"):
    agent = LoopAgent(researcher(name), tools, board="auto")
    researches = agent_tool(
        agent, name=name, description="Researches a topic.", executor=executor, parallel_safe=True
    )
    team.add(researches)
author = LoopAgent(CallableModel(writer, model="writer"), Registry(), board="auto")
team.add(agent_tool(author, name="writer", description="Writes the report.", executor=executor))
# --8<-- [end:agents]


def supervisor(request: LLMRequest) -> LLMResponse:
    """Fans out to three researchers in one turn (gamma repeats alpha's topic), then fans in."""
    turns = sum(1 for m in request.messages if m.role == "assistant")
    if turns == 0:
        asks = (("alpha", "prices"), ("beta", "weather"), ("gamma", "prices"))
        calls = tuple(
            ToolCall(name=name, arguments={"task": f"research {topic}"}, id=f"call_{name}")
            for name, topic in asks
        )
        return LLMResponse(
            provider="host",
            model="demo",
            tool_calls=calls,
            usage=tokens(request),
            finish_reason="tool_calls",
            message=Message(role="assistant", tool_calls=calls),
        )
    if turns == 1:
        return say(
            request, call=ToolCall(name="writer", arguments={"task": "write it up"}, id="call_w")
        )
    return say(request, request.messages[-1].content)


async def main() -> None:
    # --8<-- [start:run]
    ctx = executor.context()
    ask = [Message(role="user", content="Brief me on Oslo")]
    boss = CallableModel(supervisor, model="boss")
    result = await run_tool_loop(boss, team, ask, executor=executor, ctx=ctx)
    report = analyse_run(sink.events)
    # --8<-- [end:run]
    print(result.final.text)
    print(f"searches run: {len(SEARCHES)} for 3 asks (a claim stopped the repeat)")
    print(f"board entries: {len(await board_of(ctx).read())}")
    join = report["joins"][0]
    print(
        f"parallel branches: {len(join['branches'])}, slowest waited 0 ms: "
        f"{min(b['waited_ms'] for b in join['branches']) == 0}"
    )
    print(f"contested claims: {len(report['claims']['contested'])}")
    used = {row["agent"]: row["input_tokens"] for row in report["agents"]}
    print("input tokens:", ", ".join(f"{name} {n}" for name, n in sorted(used.items())))


if __name__ == "__main__":
    asyncio.run(main())
