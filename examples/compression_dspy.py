"""Compress the prompt DSPy builds for each LM call, with an adapter subclass."""

from typing import Any

import dspy

from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.integrations.dspy import dspy_compress_prompt

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:dspy]
class CompressingChatAdapter(dspy.ChatAdapter):
    """`dspy.configure(adapter=CompressingChatAdapter())`: history, demos and the new input."""

    def format(self, signature: Any, demos: list[Any], inputs: dict[str, Any]) -> Any:
        prompt = super().format(signature, demos, inputs)
        smaller, _ = dspy_compress_prompt(prompt, policy)  # a Prompt, or chat dicts on DSPy 3.4
        return smaller


# --8<-- [end:dspy]
