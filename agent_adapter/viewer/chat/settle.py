"""Ending a turn: store its outcome, free its session and send the one end event, always."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.types import to_jsonable
from agent_adapter.hooks.builtin.logging import redact_secrets
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.records import Session, Turn
from agent_adapter.viewer.chat.sessions import SessionStore
from agent_adapter.viewer.chat.watch import TurnWatch

logger = logging.getLogger(__name__)

_NOT_SAVED = {"kind": "internal-error", "message": "the turn could not be saved"}


def _no_worker_threads(exc: BaseException) -> bool:
    """True for the executor's refusal at interpreter exit, not for a failure of the work."""
    return isinstance(exc, RuntimeError) and "cannot schedule new futures" in str(exc)


class Settler:
    """Writes a turn's outcome; every path frees the session and ends the event list."""

    def __init__(self, store: SessionStore, watch: TurnWatch, desk: ChatApprovals) -> None:
        self.store = store
        self.watch = watch
        self.desk = desk

    async def finish(
        self,
        live: LiveTurn,
        session: Session,
        turn: Turn,
        status: str,
        error: dict[str, str] | None,
    ) -> None:
        """On the loop: deny open approvals, snapshot state, then settle in a worker thread."""
        approvals: list[dict[str, Any]] = []
        try:
            approvals = self.desk.close_turn(live, status)
            with self.store.lock:
                session.saved_state = to_jsonable(session.state)
        except Exception as exc:  # the turn still ends: its approvals stay denied
            log_failure(logger, logging.ERROR, exc, "chat: closing turn %s failed", turn.id)
        args = (live, session, turn, status, error, approvals)
        try:
            await asyncio.to_thread(self.settle, *args)
            return
        except Exception as exc:
            if not _no_worker_threads(exc):
                log_failure(logger, logging.ERROR, exc, "chat: settling turn %s failed", turn.id)
                self.end_anyway(live, session, turn)
                return
            log_failure(logger, logging.WARNING, exc, "chat: turn %s settled on the loop", turn.id)
        try:
            self.settle(*args)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "chat: settling turn %s failed", turn.id)
            self.end_anyway(live, session, turn)

    def settle(
        self,
        live: LiveTurn,
        session: Session,
        turn: Turn,
        status: str,
        error: dict[str, str] | None,
        approvals: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write the turn's outcome, save the session and send the end event (any thread)."""
        with self.store.lock:
            turn.status = status  # type: ignore[assignment]
            turn.error = error
            turn.answer = "".join(live.answer)
            turn.ended = time.time()
            turn.tools = live.tool_list()
            turn.approvals = approvals or []
            turn.input_tokens, turn.output_tokens = live.input_tokens, live.output_tokens
            turn.cost_usd = round(live.cost_usd, 8)
            if session.running == turn.id:
                session.running = ""
            state = redact_secrets(session.saved_state)
            view = turn.view()
        self.store.save(session)
        self.watch.remove(live)
        live.push("state", state=state)
        if status == "done":
            live.push("done", status=status, turn=view)
        else:
            live.push("error", status=status, **(error or {}), turn=view)
        logger.info("chat: turn %s %s", turn.id, status)

    def end_anyway(self, live: LiveTurn, session: Session, turn: Turn) -> None:
        """Last resort when settling failed: mark the turn failed, free it, end its stream."""
        try:
            with self.store.lock:
                turn.status, turn.error, turn.ended = "failed", dict(_NOT_SAVED), time.time()
                if session.running == turn.id:
                    session.running = ""
                view = turn.view()
            self.watch.remove(live)
            live.push("error", status="failed", **_NOT_SAVED, turn=view)
        except Exception as exc:  # nothing is left to try; the reader ends at server close
            log_failure(logger, logging.ERROR, exc, "chat: turn %s could not end", turn.id)
