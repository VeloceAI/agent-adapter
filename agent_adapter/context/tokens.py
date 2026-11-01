"""Token counting for compression: the user's counter, or a chars/4 estimate."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

logger = logging.getLogger(__name__)

TokenCounter = Callable[[list[Message]], int]
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4  # role and framing tokens every chat format adds per message


class _EstimateWarning:
    """Warn once per process that counts are estimates."""

    warned = False


def message_chars(message: Message) -> int:
    """Characters a provider sees for `message`: text, tool calls, and replayed native items.

    Provider items a port replays (`raw.items`: thinking, reasoning) count when they are
    bigger than the text; a converted native's thinking blocks count through `hidden_chars`.
    """
    chars = len(message.content) + len(message.name)
    for call in message.tool_calls:
        arguments = json.dumps(call.arguments, ensure_ascii=False, default=str, sort_keys=True)
        chars += len(call.name) + len(arguments)
    if message.raw is not None and message.raw.items:
        replayed = json.dumps(message.raw.items, ensure_ascii=False, default=str)
        chars = max(chars, len(replayed))
    hidden = getattr(message, "hidden_chars", 0)
    return chars + (hidden if isinstance(hidden, int) else 0)


def estimate_tokens(messages: Sequence[Message]) -> int:
    """About chars/4 per message plus a fixed per-message overhead."""
    return sum(
        math.ceil(message_chars(m) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS for m in messages
    )


class TokenMeter:
    """Counts with one counter and caches single-message counts while the messages live."""

    def __init__(self, count_tokens: TokenCounter | None = None) -> None:
        if count_tokens is not None and not callable(count_tokens):
            raise ConfigError("count_tokens must be a callable(list[Message]) -> int")
        if count_tokens is None and not _EstimateWarning.warned:
            _EstimateWarning.warned = True
            logger.warning(
                "context compression counts tokens with a chars/4 estimate; "
                "pass count_tokens= for exact counts"
            )
        self._count = count_tokens or estimate_tokens
        self._single: dict[int, tuple[Message, int]] = {}  # id -> (message kept alive, count)

    def total(self, messages: Sequence[Message]) -> int:
        """Tokens of the whole list.

        Raises:
            ConfigError: the counter returned something other than an int >= 0.
        """
        return self._checked(self._count(list(messages)))

    def of(self, message: Message) -> int:
        """Tokens of one message, cached by identity.

        Raises:
            ConfigError: the counter returned something other than an int >= 0.
        """
        cached = self._single.get(id(message))
        if cached is not None and cached[0] is message:
            return cached[1]
        tokens = self._checked(self._count([message]))
        self._single[id(message)] = (message, tokens)
        return tokens

    @staticmethod
    def _checked(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"count_tokens must return an int >= 0, got {type(value).__name__}")
        return value
