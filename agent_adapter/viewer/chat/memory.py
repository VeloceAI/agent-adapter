"""Cross-chat memory for one user: recall at a turn's start, save every N turns, view, delete."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, ConfigError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.memory import (
    GraphitiMemory,
    GuardedMemory,
    MemoryHit,
    MemoryRecord,
    MemoryScope,
    SQLiteMemory,
    ZepMemory,
)
from agent_adapter.memory.records import MAX_ID_CHARS, MAX_K, MAX_TEXT_CHARS
from agent_adapter.viewer.chat.contract import ChatSession
from agent_adapter.viewer.chat.targets import load_target

logger = logging.getLogger(__name__)

SavePolicy = Literal["summary", "facts", "none"]
SAVE_POLICIES: tuple[str, ...] = ("summary", "facts", "none")
MAX_SAVED = 20  # records one save may add
MAX_LISTED = 500  # records the Memory panel lists
MAX_SHOWN = 2_000  # characters of one recalled text sent to the page
SOURCE = "chat"


@runtime_checkable
class ChatMemoryWriter(Protocol):
    """An agent that can write memories: the texts to keep from `transcript`, per `policy`."""

    def save_memories(
        self, transcript: tuple[Message, ...], session: ChatSession, policy: str
    ) -> Sequence[str] | Awaitable[Sequence[str]]: ...


@dataclass(frozen=True)
class ChatMemory:
    """One user's memory in the chat host; recall and saves never cross users."""

    memory: GuardedMemory
    user_id: str
    save: SavePolicy = "none"
    every: int = 1
    k: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.memory, GuardedMemory):
            raise ConfigError(
                f"chat memory must be a memory backend from agent_adapter.memory, got "
                f"{type(self.memory).__name__}; wrap other stores in CallableMemory"
            )
        user = self.user_id
        if not isinstance(user, str) or not user.strip() or len(user) > MAX_ID_CHARS:
            raise ConfigError(f"chat memory needs a user id (1-{MAX_ID_CHARS} characters)")
        if self.save not in SAVE_POLICIES:
            raise ConfigError(f"save must be one of {list(SAVE_POLICIES)}, got {self.save!r}")
        for label, value, top in (("every", self.every, 10_000), ("k", self.k, MAX_K)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
                raise ConfigError(f"memory {label} must be an int from 1 to {top}, got {value!r}")

    @property
    def user_only(self) -> bool:
        """True for backends that keep one partition per user and no metadata."""
        flag = getattr(self.memory, "user_only", None)
        if isinstance(flag, bool):
            return flag
        return isinstance(self.memory, ZepMemory | GraphitiMemory)

    @property
    def scope(self) -> MemoryScope:
        """Every chat of this user."""
        return MemoryScope(user_id=self.user_id)

    def info(self) -> dict[str, Any]:
        """What the page shows: on, the user, the backend and the save policy."""
        return {
            "enabled": True,
            "user": self.user_id,
            "backend": self.memory.backend,
            "save": self.save,
            "every": self.every,
        }

    async def recall(self, text: str, session_id: str, ctx: ToolContext) -> list[MemoryHit]:
        """Records of this user's other chats relevant to `text`; [] when the backend fails."""
        try:
            hits = await self.memory.search(
                text[:MAX_TEXT_CHARS], scope=self.scope, k=min(MAX_K, self.k * 2), ctx=ctx
            )
        except AdapterError as exc:
            log_failure(logger, logging.WARNING, exc, "chat: memory recall failed; none used")
            return []
        others = [hit for hit in hits if hit.record.scope.session_id != session_id]
        return others[: self.k]

    def due(self, offered: int) -> bool:
        """Whether a save is due after `offered` finished turns since the last save."""
        return self.save != "none" and offered >= self.every

    async def save_texts(
        self, texts: Sequence[str], *, session_id: str, turn_id: str, ctx: ToolContext
    ) -> int:
        """Store `texts` in this user's scope for the chat; how many were stored.

        Raises:
            AdapterError: the backend refused or failed (the caller logs it).
        """
        kind = "summary" if self.save == "summary" else "fact"
        if self.user_only:  # Zep, Graphiti: one partition per user, no metadata
            scope, metadata = self.scope, {}
        else:
            scope = MemoryScope(user_id=self.user_id, session_id=session_id)
            metadata = {"kind": kind, "chat": session_id}
        records = [
            MemoryRecord(
                scope=scope,
                text=text[:MAX_TEXT_CHARS],
                metadata=metadata,
                source=turn_id[:MAX_ID_CHARS],
            )
            for text in clean_texts(texts)
        ]
        if not records:
            return 0
        stored = await self.memory.add(records, ctx=ctx)
        logger.info("chat: saved %d memory record(s) for chat %s", len(stored), session_id)
        return len(stored)

    def records(self) -> list[dict[str, Any]]:
        """This user's live records, newest first, as the Memory panel shows them.

        Raises:
            AdapterError: the backend failed.
        """
        found = self.memory.export_sync(self.scope)
        found.sort(key=lambda record: record.created_at.timestamp() if record.created_at else 0.0)
        return [record_view(record) for record in reversed(found[-MAX_LISTED:])]

    def delete(self, ids: Sequence[str]) -> int:
        """Delete these records, only those inside this user's scope; how many went.

        Raises:
            AdapterError: bad ids, or the backend failed.
        """
        count = self.memory.delete_sync(list(ids), scope=self.scope)
        logger.info("chat: deleted %d memory record(s) from the Memory panel", count)
        return count


async def written(
    writer: ChatMemoryWriter, transcript: tuple[Message, ...], session: ChatSession, policy: str
) -> list[str]:
    """The writer's texts, awaited when it is async.

    Raises:
        TypeError: the writer returned something other than a sequence of str.
    """
    result = writer.save_memories(transcript, session, policy)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, str) or not isinstance(result, Sequence):
        raise TypeError("save_memories must return a sequence of str")
    if not all(isinstance(text, str) for text in result):
        raise TypeError("save_memories must return a sequence of str")
    return list(result)


def clean_texts(texts: Sequence[str]) -> list[str]:
    """Stripped, non-empty, distinct texts, at most `MAX_SAVED`; "NONE" means nothing."""
    kept: list[str] = []
    for text in texts:
        line = text.strip()
        if line and line.upper() != "NONE" and line not in kept:
            kept.append(line)
    return kept[:MAX_SAVED]


def open_memory(spec: str) -> Any:
    """`sqlite:PATH` as a `SQLiteMemory`, else the `module:attribute` it names (`--memory`).

    Raises:
        ConfigError: an empty path, or a target that cannot be imported.
    """
    if spec.startswith("sqlite:"):
        path = spec.removeprefix("sqlite:")
        if not path:
            raise ConfigError("--memory sqlite: needs a file path, e.g. sqlite:memory.db")
        return SQLiteMemory(path)
    return load_target(spec, "--memory")


def record_view(record: MemoryRecord) -> dict[str, Any]:
    """A record as JSON for the page."""
    return {
        "id": record.id,
        "text": record.text[:MAX_SHOWN],
        "kind": str(record.metadata.get("kind", "")),
        "chat": record.scope.session_id,
        "created_at": record.created_at.timestamp() if record.created_at else None,
    }


def hit_view(hit: MemoryHit) -> dict[str, Any]:
    """A recalled hit as JSON for the page, with its score."""
    return {**record_view(hit.record), "score": round(hit.score, 4)}
