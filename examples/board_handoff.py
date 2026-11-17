"""Agents one after another on a shared board: a planner hands off, a writer gets only news."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
)
from agent_adapter.board import board_of, board_tools
from agent_adapter.exec import LoopAgent, agent_tool

NOTES: list[str] = []


def say(text: str = "", call: ToolCall | None = None) -> LLMResponse:
    calls = (call,) if call else ()
    return LLMResponse(
        provider="host",
        model="demo",
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        message=Message(role="assistant", content=text, tool_calls=calls),
    )


def planner(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: post the brief for the next agent, then stop."""
    if any(m.role == "tool" for m in request.messages):
        return say("brief posted")
    brief = "2 days in Oslo, museums first, 300 EUR in all"
    post = {"kind": "handoff", "text": brief, "key": "brief"}
    return say(call=ToolCall(name="board_post", arguments=post, id="p1"))


def writer(request: LLMRequest) -> LLMResponse:
    """Reads the brief from the board note the loop added (marked as data)."""
    notes = [m.content for m in request.messages if "<shared-board>" in m.content]
    NOTES.extend(notes)
    brief = next(line for line in notes[-1].splitlines() if "[handoff:brief]" in line)
    return say("Plan for " + brief.split("] ", 1)[1])


# --8<-- [start:handoff]
tools = Registry()
tools.add_many(board_tools(include=("post",)))
crew = Registry()
executor = Executor(crew)
plan = LoopAgent(CallableModel(planner, model="planner"), tools)
write = LoopAgent(CallableModel(writer, model="writer"), Registry(), board="auto", board_chars=600)
crew.add(agent_tool(plan, name="planner", description="Plans the trip.", executor=executor))
crew.add(agent_tool(write, name="writer", description="Writes the plan.", executor=executor))


async def main() -> None:
    ctx = executor.context()
    board = board_of(ctx)
    for n in range(40):  # what earlier agents of this run left on the board
        await board.post("finding", f"museum {n}: open 10-17, ticket {10 + n} EUR", by="scout")
    await executor.run("planner", {"task": "plan two days in Oslo"}, ctx)
    answer = await executor.run("writer", {"task": "write the plan"}, ctx)
    # --8<-- [end:handoff]
    entries = await board.read()
    print(answer)
    print(f"board: {len(entries)} entries, {sum(len(e.text) for e in entries)} characters")
    print(f"writer's note: {len(NOTES[0])} characters (cap 600), {len(NOTES)} note")


if __name__ == "__main__":
    asyncio.run(main())
