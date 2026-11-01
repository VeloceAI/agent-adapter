"""Call a chat agent under the contract and yield its answer as text chunks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.viewer.chat.contract import ChatAgent, ChatMessage, ChatSession


def _on_loop(agent: object) -> bool:
    """True when calling `agent` only builds a coroutine or an async generator."""
    call = type(agent).__call__ if not inspect.isroutine(agent) else agent
    return is_async(agent) or inspect.isasyncgenfunction(agent) or inspect.isasyncgenfunction(call)


async def reply_chunks(
    agent: ChatAgent, message: ChatMessage, session: ChatSession
) -> AsyncGenerator[str, None]:
    """The agent's answer chunks; a sync agent runs in a worker thread.

    Raises:
        ToolUsageError: the agent returned something other than text or an async iterator
            of text.
        Exception: whatever the agent raised.
    """
    if _on_loop(agent):
        result: Any = agent(message, session)
    else:
        result = await asyncio.to_thread(agent, message, session)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, str):
        yield result
        return
    if not hasattr(result, "__aiter__"):
        raise ToolUsageError(
            f"a chat agent must return str or an async iterator of str, got {type(result).__name__}"
        )
    if hasattr(result, "aclose"):
        async with aclosing(result) as chunks:
            async for chunk in chunks:
                yield _text(chunk)
        return
    async for chunk in result:
        yield _text(chunk)


def _text(chunk: object) -> str:
    if not isinstance(chunk, str):
        raise ToolUsageError(f"a chat agent must stream str chunks, got {type(chunk).__name__}")
    return chunk
