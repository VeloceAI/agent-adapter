"""Compress a CrewAI agent's messages before each LLM call; they are OpenAI chat dicts."""

from typing import Any

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for_sync

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:crewai]
def compress_crewai_messages(context: Any) -> None:
    """`register_before_llm_call_hook(compress_crewai_messages)`, or decorate `@before_llm_call`."""
    smaller, _ = compress_for_sync(context.messages, policy, format="openai")
    context.messages[:] = smaller  # in place: CrewAI keeps this very list as the history


# --8<-- [end:crewai]
