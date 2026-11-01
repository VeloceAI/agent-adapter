"""The turn host: each turn is a task on one background `LoopRunner` loop."""

from __future__ import annotations

import asyncio
import copy
import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import aclosing
from http import HTTPStatus
from typing import Any

from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.exec.executor import Executor
from agent_adapter.hooks.builtin.logging import redact_secrets
from agent_adapter.runtime.current import use_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.runtime.loop_runner import LoopRunner
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.contract import Attachment, ChatAgent, ChatMessage, ChatSession
from agent_adapter.viewer.chat.evalcases import EvalCases
from agent_adapter.viewer.chat.files import FileStore
from agent_adapter.viewer.chat.invoke import reply_chunks
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.records import ChatOptions, Refused, Session, Turn
from agent_adapter.viewer.chat.sessions import SessionStore, new_key
from agent_adapter.viewer.chat.settle import Settler
from agent_adapter.viewer.chat.turn_io import (
    checked_ids,
    checked_text,
    failure,
    history,
    replayed,
    unmask,
)
from agent_adapter.viewer.chat.watch import TurnWatch

logger = logging.getLogger(__name__)

MAX_RUNNING = 8  # turns running at once, over all sessions
MAX_LIVE = 200  # ended turns whose event lists stay in memory
START_TIMEOUT_S = 10.0
_STOPPED = {"kind": "stopped", "message": "the turn was stopped"}
_SERVER_STOPPED = {"kind": "interrupted", "message": "the server stopped during this turn"}


class ChatHost:
    """Starts, streams and stops turns; HTTP threads call it, turns run on its own loop."""

    def __init__(
        self,
        agent: ChatAgent,
        executor: Executor,
        *,
        options: ChatOptions,
        store: SessionStore,
        watch: TurnWatch,
        desk: ChatApprovals,
        refusal: Callable[[tuple[Attachment, ...]], str] | None = None,
    ) -> None:
        self.agent = agent
        self.executor = executor
        self.options = options
        self.store = store
        self.files = FileStore(store)
        self.evals = EvalCases(store.folder)
        self.watch = watch
        self.desk = desk
        self._settler = Settler(store, watch, desk)
        self._refusal = refusal
        self._runner = LoopRunner(name="agent_adapter-chat")
        self._lock = threading.Lock()
        self._live: OrderedDict[str, LiveTurn] = OrderedDict()
        self._tasks: set[asyncio.Task[None]] = set()
        self.closed = False
        self.on_close: list[Callable[[], object]] = []

    @property
    def mode(self) -> str:
        return self.options.mode

    def info(self) -> dict[str, Any]:
        return {"enabled": True, "mode": self.mode, "agent_name": self.options.agent_name}

    # --- turns (HTTP threads) -------------------------------------------------------------

    def start(
        self, session_id: str, text: Any, file_ids: Any, *, edit_of: str = ""
    ) -> dict[str, str]:
        """Start a turn (in a new branch when `edit_of` names a turn); its ids.

        Raises:
            Refused: bad input (400), unknown ids (404), a running turn or full session (409).
        """
        message = checked_text(text)
        ids = checked_ids(file_ids)
        session = self.store.get(session_id)
        return self._begin(session, message, ids, edit_of=edit_of)

    def replay(self, turn_id: str) -> dict[str, str]:
        """Run an ended turn's input again on the same history, in a new branch.

        Raises:
            Refused: no such turn (404) or the turn still runs (409).
        """
        session, turn = self.store.find_turn(turn_id)
        if turn.status == "running":
            raise Refused(HTTPStatus.CONFLICT, "the turn is still running")
        return self._begin(session, turn.text, list(turn.attachments), replay_of=turn_id)

    def _begin(
        self, session: Session, text: str, ids: list[str], *, edit_of: str = "", replay_of: str = ""
    ) -> dict[str, str]:
        """Check the turn, fork the branch an edit or replay runs in, then start the task."""
        if self.closed:
            raise Refused(HTTPStatus.SERVICE_UNAVAILABLE, "the server is stopping")
        if not text.strip() and not ids:
            raise Refused(HTTPStatus.BAD_REQUEST, "send text or an attachment")
        attachments = self.files.attachments(session, ids)
        refusal = self._refusal(attachments) if self._refusal is not None else ""
        if refusal:
            raise Refused(HTTPStatus.BAD_REQUEST, refusal)
        self._check_room()
        fork_at = edit_of or replay_of
        if fork_at:
            label = "edit" if edit_of else "replay"
            session = self.store.branch(session, fork_at, f"{session.title} ({label})")
            attachments = self.files.attachments(session, ids)
        try:
            turn, live = self._register(session, text, ids, edit_of=edit_of, replay_of=replay_of)
        except Refused:
            if fork_at:  # no empty branch is left behind
                self.store.discard(session)
            raise
        self.watch.add(live)
        self.store.save(session)
        try:
            self._runner.run(
                lambda: self._spawn(live, session, turn, attachments), timeout=START_TIMEOUT_S
            )
        except (ToolUsageError, TimeoutError) as exc:
            log_failure(logger, logging.ERROR, exc, "chat: turn %s did not start", turn.id)
            failed = {"kind": "internal-error", "message": "the turn did not start"}
            self._settler.settle(live, session, turn, "failed", failed)
            raise Refused(HTTPStatus.SERVICE_UNAVAILABLE, "the turn did not start") from exc
        logger.info("chat: turn %s started in %s (run %s)", turn.id, session.id, turn.run_id)
        return {"turn_id": turn.id, "run_id": turn.run_id, "session_id": session.id}

    def _register(
        self, session: Session, text: str, ids: list[str], *, edit_of: str, replay_of: str
    ) -> tuple[Turn, LiveTurn]:
        """Add the running turn to the session and its event list to the host.

        Raises:
            Refused: a turn runs in the session, too many run, or the session is full (409).
        """
        with self.store.lock, self._lock:
            if session.running:
                raise Refused(HTTPStatus.CONFLICT, "a turn is already running in this session")
            self._check_room(held=True)
            self.store.check_turn_room(session)
            turn = Turn(
                id=new_key("turn_"),
                session_id=session.id,
                run_id=new_id("run_"),
                text=text,
                attachments=ids,
                state_before=copy.deepcopy(session.saved_state),
                edit_of=edit_of,
                replay_of=replay_of,
            )
            session.turns.append(turn)
            session.running = turn.id
            live = LiveTurn(turn.id, session.id, turn.run_id)
            self._live[turn.id] = live
            self._forget_old()
        return turn, live

    def _check_room(self, *, held: bool = False) -> None:
        """Raises Refused (409) when `MAX_RUNNING` turns run already; `held`: caller has `_lock`."""
        if held:
            running = sum(1 for live in self._live.values() if not live.ended)
        else:
            with self._lock:
                running = sum(1 for live in self._live.values() if not live.ended)
        if running >= MAX_RUNNING:
            raise Refused(HTTPStatus.CONFLICT, f"at most {MAX_RUNNING} turns run at once")

    def stop(self, turn_id: str) -> dict[str, Any]:
        """Cancel a running turn; its pending approvals are denied.

        Raises:
            Refused: no such turn (404) or it is not running (409).
        """
        _, turn = self.store.find_turn(turn_id)
        with self._lock:
            live = self._live.get(turn_id)
        if live is None or live.ended or turn.status != "running":
            raise Refused(HTTPStatus.CONFLICT, "the turn is not running")
        try:
            self._runner.run(lambda: self._cancel(live, "stopped"), timeout=START_TIMEOUT_S)
        except (ToolUsageError, TimeoutError) as exc:
            log_failure(logger, logging.ERROR, exc, "chat: stopping turn %s failed", turn_id)
            raise Refused(HTTPStatus.SERVICE_UNAVAILABLE, "the turn could not be stopped") from exc
        return {"turn_id": turn_id, "stopping": True}

    def live(self, turn_id: str) -> LiveTurn:
        """The turn's event list; for a turn from an earlier start, its stored outcome.

        Raises:
            Refused: no such turn (404).
        """
        with self._lock:
            live = self._live.get(turn_id)
        if live is not None:
            return live
        _, turn = self.store.find_turn(turn_id)
        return replayed(turn)

    def set_state(self, session_id: str, state: Any) -> dict[str, Any]:
        """Replace a session's state between turns; masked values keep what they hid.

        Raises:
            Refused: not a JSON object (400), no such session (404) or a turn runs (409).
        """
        if not isinstance(state, dict):
            raise Refused(HTTPStatus.BAD_REQUEST, "state must be a JSON object")
        session = self.store.get(session_id)
        with self.store.lock:
            if session.running:
                raise Refused(HTTPStatus.CONFLICT, "the state cannot change while a turn runs")
            restored = unmask(state, session.saved_state)
            saved = copy.deepcopy(restored)  # both copies made before either field changes
            session.state, session.saved_state = restored, saved
        self.store.save(session)
        return dict(redact_secrets(session.saved_state))

    # --- the loop side --------------------------------------------------------------------

    async def _spawn(
        self, live: LiveTurn, session: Session, turn: Turn, attachments: tuple[Attachment, ...]
    ) -> None:
        task = asyncio.get_running_loop().create_task(
            self._drive(live, session, turn, attachments), name=f"chat-{turn.id}"
        )
        live.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _cancel(self, live: LiveTurn, reason: str) -> None:
        live.stop_reason = live.stop_reason or reason  # a task not made yet ends at its start
        if live.task is None or live.task.done():
            return
        self.desk.close_turn(live, reason)
        live.task.cancel()

    async def _drive(
        self, live: LiveTurn, session: Session, turn: Turn, attachments: tuple[Attachment, ...]
    ) -> None:
        view = ChatSession(
            id=session.id,
            history=history(session, turn),
            state=session.state,
            executor=self.executor,
            run_id=turn.run_id,
        )
        live.push("state", state=redact_secrets(session.saved_state))
        if live.stop_reason:  # stopped before the task ran
            await self._settler.finish(live, session, turn, "stopped", dict(_STOPPED))
            return
        status, error = "done", None
        limit = asyncio.timeout(self.options.turn_timeout)
        try:
            async with (
                limit,
                self.executor.run_scope(run_id=turn.run_id, state=session.state) as ctx,
            ):
                with use_context(ctx):
                    message = ChatMessage(turn.text, attachments)
                    async with aclosing(reply_chunks(self.agent, message, view)) as chunks:
                        async for chunk in chunks:
                            live.text(chunk)
        except TimeoutError as exc:
            if limit.expired():
                status, error = (
                    "timed-out",
                    {
                        "kind": "turn-timeout",
                        "message": f"the turn ran past {self.options.turn_timeout:g} s",
                    },
                )
            else:
                status, error = "failed", failure(exc, turn)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            status, error = "stopped", dict(_STOPPED)
        except Exception as exc:
            status, error = "failed", failure(exc, turn)
        # A stop denies pending approvals and cancels the task; whichever lands first, it stopped.
        if live.stop_reason and status == "failed":
            status, error = "stopped", dict(_STOPPED)
        await self._settler.finish(live, session, turn, status, error)

    def _forget_old(self) -> None:
        """Drop the oldest ended event lists past `MAX_LIVE`. Caller holds `_lock`."""
        ended = [key for key, live in self._live.items() if live.ended]
        for key in ended[: max(0, len(ended) - MAX_LIVE)]:
            del self._live[key]

    def _interrupt(self, live: LiveTurn) -> None:
        """End a turn whose task never settled: stored as stopped, readers get the end event."""
        try:
            session, turn = self.store.find_turn(live.turn_id)
            closed = self.desk.close_turn(live, "stopped")
            self._settler.settle(live, session, turn, "stopped", dict(_SERVER_STOPPED), closed)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "chat: ending turn %s failed", live.turn_id)
            stand_in = Turn(  # the stored turn was not reachable: the same shape, from `live`
                id=live.turn_id,
                session_id=live.session_id,
                run_id=live.run_id,
                text="",
                status="stopped",
                error=dict(_SERVER_STOPPED),
                answer="".join(live.answer),
                ended=time.time(),
            )
            live.push("error", status="stopped", **_SERVER_STOPPED, turn=stand_in.view())

    def close(self) -> None:
        """Stop running turns (their approvals are denied), wake readers, stop the loop."""
        if self.closed:
            return
        self.closed = True
        with self._lock:
            running = [live for live in self._live.values() if not live.ended]
        for live in running:
            try:
                self._runner.run(
                    functools.partial(self._cancel, live, "stopped"), timeout=START_TIMEOUT_S
                )
            except (ToolUsageError, TimeoutError) as exc:
                log_failure(
                    logger, logging.WARNING, exc, "chat: stopping turn %s failed", live.turn_id
                )
        self._runner.close(timeout=START_TIMEOUT_S)
        with self._lock:  # again: a turn may have registered while close() ran
            unsettled = [live for live in self._live.values() if not live.ended]
        for live in unsettled:
            self._interrupt(live)
        for callback in self.on_close:
            try:
                callback()
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "chat: a close step failed")
