"""Compress a LlamaIndex `ChatMessage` list before the LLM call; the list keeps its type."""

from typing import Any

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:llamaindex]
import agent_adapter.integrations.llamaindex  # noqa: E402, F401  registers its format


async def ask_llamaindex(llm: Any, messages: list[Any]) -> Any:
    """`messages` is a `list[ChatMessage]`; compress an agent's `chat_history=` the same way."""
    smaller, _ = await compress_for(messages, policy, format="llamaindex")
    return await llm.achat(smaller)


# --8<-- [end:llamaindex]
