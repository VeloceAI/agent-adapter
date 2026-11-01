"""The chat agent contract: what an agent receives for one turn and what it may return."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import Message
from agent_adapter.exec.executor import Executor

ChatReply = str | AsyncIterator[str]


@dataclass(frozen=True)
class Attachment:
    """A file the user attached; `path` is a local copy the server stored."""

    name: str
    media_type: str
    size: int
    path: Path
    id: str = ""

    def read_bytes(self) -> bytes:
        """The file's bytes.

        Raises:
            OSError: the stored file cannot be read.
        """
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        """The file's text.

        Raises:
            OSError: the stored file cannot be read.
            UnicodeDecodeError: the file is not text in `encoding`.
        """
        return self.path.read_text(encoding=encoding)


@dataclass(frozen=True)
class ChatMessage:
    """The user's message of one turn."""

    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass
class ChatSession:
    """One conversation as the agent sees it during a turn; `state` is live and mutable."""

    id: str
    history: tuple[Message, ...]
    state: dict[str, Any]
    executor: Executor
    run_id: str

    def context(self) -> ToolContext:
        """A context in this turn's run: pass it as `ctx=` so tool calls show in the chat."""
        return self.executor.context(run_id=self.run_id, state=self.state)


@runtime_checkable
class ChatAgent(Protocol):
    """`agent(message, session)`: async or sync; returns text or an async iterator of chunks."""

    def __call__(
        self, message: ChatMessage, session: ChatSession
    ) -> ChatReply | Awaitable[ChatReply]: ...
