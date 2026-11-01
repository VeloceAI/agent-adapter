"""Compress what Microsoft Agent Framework sends on every model call, with chat middleware."""

from typing import Any

from agent_framework import ChatContext, chat_middleware

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:agent_framework]
import agent_adapter.integrations.agent_framework  # noqa: E402, F401  registers its format


@chat_middleware
async def compress_messages(context: ChatContext, call_next: Any) -> None:
    """`Agent(client=..., middleware=[compress_messages])`: the session keeps the full history."""
    context.messages, _ = await compress_for(context.messages, policy, format="agent_framework")
    await call_next()


# --8<-- [end:agent_framework]
