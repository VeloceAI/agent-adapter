"""A chat agent as a task agent, so `agent_tool` can run it as a sub-agent of another run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import aclosing

from agent_adapter.core.context import ToolContext
from agent_adapter.exec.executor import Executor
from agent_adapter.viewer.chat.contract import ChatAgent, ChatMessage, ChatSession
from agent_adapter.viewer.chat.invoke import reply_chunks


def chat_agent_task(
    agent: ChatAgent, executor: Executor
) -> Callable[[str, ToolContext], Awaitable[str]]:
    """`fn(task, ctx)` that runs `agent(message, session)` in the calling run; chunks are joined."""

    async def run(task: str, ctx: ToolContext) -> str:
        session = ChatSession(
            id=f"agent:{ctx.call_id}",
            history=(),
            state=dict(ctx.state),
            executor=executor,
            run_id=ctx.run_id,
        )
        async with aclosing(reply_chunks(agent, ChatMessage(text=task), session)) as chunks:
            return "".join([chunk async for chunk in chunks if isinstance(chunk, str)])

    return run
