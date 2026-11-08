"""Memory in a chat turn: recall at its start and, when due, save after the answer."""

from __future__ import annotations

import asyncio
import logging

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import log_failure
from agent_adapter.core.messages import Message
from agent_adapter.viewer.chat.contract import ChatSession
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.memory import (
    ChatMemory,
    ChatMemoryWriter,
    hit_view,
    written,
)
from agent_adapter.viewer.chat.records import Session, Turn
from agent_adapter.viewer.chat.sessions import SessionStore

logger = logging.getLogger(__name__)


async def recall(
    memory: ChatMemory, view: ChatSession, turn: Turn, live: LiveTurn, ctx: ToolContext
) -> None:
    """Put the user's relevant records on `view.memories`, the turn and the live stream."""
    hits = await memory.recall(turn.text, view.id, ctx)
    view.memories = tuple(hits)
    turn.memories = [{"id": hit.record.id, "score": round(hit.score, 4)} for hit in hits]
    live.push("memory", recalled=[hit_view(hit) for hit in hits])


def pending_transcript(
    store: SessionStore, session: Session, turn: Turn, answer: str
) -> tuple[int, tuple[Message, ...]]:
    """Finished turns since the last save, this one included: (count, user/assistant messages)."""
    messages: list[Message] = []
    offered = 0
    with store.lock:
        index = session.turns.index(turn)
        for earlier in session.turns[session.memory_upto : index]:
            if earlier.status == "done":
                offered += 1
                messages.append(Message(role="user", content=earlier.text))
                messages.append(Message(role="assistant", content=earlier.answer))
    messages.append(Message(role="user", content=turn.text))
    messages.append(Message(role="assistant", content=answer))
    return offered + 1, tuple(messages)


async def save_if_due(
    memory: ChatMemory,
    writer: ChatMemoryWriter,
    store: SessionStore,
    session: Session,
    turn: Turn,
    live: LiveTurn,
    view: ChatSession,
    ctx: ToolContext,
) -> None:
    """Save memories of the turns since the last save when `every` is reached.

    A failure is logged and shown as a `memory` event; it never fails the turn.
    """
    offered, transcript = pending_transcript(store, session, turn, "".join(live.answer))
    if not memory.due(offered):
        return
    try:
        texts = await written(writer, transcript, view, memory.save)
        saved = await memory.save_texts(texts, session_id=session.id, turn_id=turn.id, ctx=ctx)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # memory is extra: the answer stands without it
        log_failure(logger, logging.WARNING, exc, "chat: saving memories of %s failed", turn.id)
        live.push("memory", saved=0, error=type(exc).__name__)
        return
    with store.lock:
        session.memory_upto = session.turns.index(turn) + 1
        turn.memory_saved = saved
    live.push("memory", saved=saved)
