"""Compress a Semantic Kernel `ChatHistory` copy before the chat service call."""

from typing import Any

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:semantic_kernel]
import agent_adapter.integrations.semantic_kernel  # noqa: E402, F401  registers its format


async def ask_semantic_kernel(service: Any, history: Any, settings: Any) -> Any:
    """The service gets a shorter copy; `history` itself keeps every message."""
    smaller, _ = await compress_for(history.messages, policy, format="semantic_kernel")
    shorter = history.model_copy(update={"messages": smaller})
    return await service.get_chat_message_contents(chat_history=shorter, settings=settings)


# --8<-- [end:semantic_kernel]
