"""Compress the chat messages DSPy builds for each LM call, with an adapter subclass."""

from typing import Any

import dspy

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for_sync

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:dspy]
class CompressingChatAdapter(dspy.ChatAdapter):
    """`dspy.configure(adapter=CompressingChatAdapter())`: history, demos and the new input."""

    def format(self, signature: Any, demos: list[Any], inputs: dict[str, Any]) -> list[Any]:
        messages = super().format(signature, demos, inputs)
        smaller, _ = compress_for_sync(messages, policy, format="openai")  # OpenAI chat dicts
        return smaller


# --8<-- [end:dspy]
