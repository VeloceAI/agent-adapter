"""ShrinkToolResults: replace big, old tool results with a short note (optionally saved)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from agent_adapter.context.recall import RECALL_TOOL_NAME, owner_scope, save_result
from agent_adapter.context.step import BaseCompressionStep, StepContext, check_count, check_ttl
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import CachePort
from agent_adapter.runtime.current import current_context

PREVIEW_CHARS = 200  # how much of a shortened result stays visible
DEFAULT_SAVE_SECONDS = 86_400.0  # a saved full result expires after a day


def placeholder(message: Message, tokens: int, result_id: str) -> str:
    """The note that replaces a tool result's text."""
    preview = message.content[:PREVIEW_CHARS]
    more = "..." if len(message.content) > PREVIEW_CHARS else ""
    if result_id:
        where = f'Call {RECALL_TOOL_NAME} with id "{result_id}" to read it in full.'
    else:
        where = "The full text is not available."
    return (
        f"[Tool result shortened to save context: it had about {tokens} tokens. {where}]\n"
        f"Start of the result: {preview}{more}"
    )


@dataclass(frozen=True, kw_only=True)
class ShrinkToolResults(BaseCompressionStep):
    """Shorten tool results over `bigger_than_tokens` older than `older_than_turns`, oldest first.

    A turn is one assistant message; the newest `older_than_turns` turns are left alone. With
    `save_to`, the full text is saved there for `save_for_seconds` (None: no expiry), under
    the run's thread or run id, and the note gives an id for `recall_tool`.
    """

    name: ClassVar[str] = "shrink_tool_results"
    older_than_turns: int = 2
    bigger_than_tokens: int = 5000
    save_to: CachePort | None = None
    save_for_seconds: float | None = DEFAULT_SAVE_SECONDS

    def __post_init__(self) -> None:
        check_count("older_than_turns", self.older_than_turns, minimum=0)
        check_count("bigger_than_tokens", self.bigger_than_tokens, minimum=1)
        if self.save_to is not None and not callable(getattr(self.save_to, "set", None)):
            raise ConfigError("save_to must be a CachePort")
        check_ttl("save_for_seconds", self.save_for_seconds)

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """`messages` with old big tool results shortened until under the target."""
        total = context.tokens(messages)
        if context.under_target(total):
            return messages
        out = list(messages)
        assistants = sum(1 for m in messages if m.role == "assistant")
        seen = 0  # assistant messages so far
        for unit in context.units(messages):
            seen += sum(1 for m in unit.messages if m.role == "assistant")
            if unit.protected or unit.kind not in ("tool_round", "tool_results"):
                continue
            if assistants - seen < self.older_than_turns:
                break  # this round and the later ones are recent
            for offset, message in enumerate(unit.messages):
                if context.under_target(total):
                    return out
                tokens = context.message_tokens(message)
                if message.role != "tool" or tokens <= self.bigger_than_tokens:
                    continue
                result_id = ""
                if self.save_to is not None:
                    scope = owner_scope(context.ctx or current_context())
                    result_id = await save_result(
                        self.save_to, message, scope=scope, ttl=self.save_for_seconds
                    )
                note = placeholder(message, tokens, result_id)
                short = message.model_copy(update={"content": note})
                short_tokens = context.message_tokens(short)
                if short_tokens < tokens:
                    out[unit.start + offset] = short
                    total += short_tokens - tokens
        return out
