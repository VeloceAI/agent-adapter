"""Compress the CAMEL `ChatAgent` memory records it builds each model call's context from."""

from typing import Any

from camel.memories import ContextRecord, ScoreBasedContextCreator

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for_sync

policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:camel]
import agent_adapter.integrations.camel  # noqa: E402, F401  registers its format


class CompressingContextCreator(ScoreBasedContextCreator):
    """`ChatAgent(memory=ChatHistoryMemory(CompressingContextCreator(counter, limit)))`."""

    def create_context(self, records: list[ContextRecord]) -> Any:
        memory = [record.memory_record for record in records]
        smaller, _ = compress_for_sync(memory, policy, format="camel")
        kept = [ContextRecord(memory_record=r, score=1.0, timestamp=r.timestamp) for r in smaller]
        return super().create_context(kept)  # the memory itself keeps every record


# --8<-- [end:camel]
