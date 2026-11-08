"""The interactive approval desk: a tool call waits on a future that one UI decision resolves."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.approval.builtin import CallbackApproval
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalNotFound, log_failure
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.current import current_context
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.viewer.chat.approval_history import History, arguments_of, context_view
from agent_adapter.viewer.chat.live import LiveTurn, preview
from agent_adapter.viewer.chat.watch import TurnWatch

logger = logging.getLogger(__name__)

MAX_REASON = 500
MAX_KEPT = 1_000  # decided requests remembered, so a late second decision still gets 409
_ENDED = {
    "stopped": "the turn was stopped",
    "timed-out": "the turn timed out",
    "failed": "the turn failed",
}


@dataclass
class _Pending:
    approval_id: str
    live: LiveTurn
    call_id: str
    tool: str
    arguments: str
    hint: str
    future: asyncio.Future[ApprovalDecision]
    loop: asyncio.AbstractEventLoop
    status: str = "pending"  # approved | denied | timed-out | stopped
    reason: str = ""
    decided_by: str = ""
    decision: ApprovalDecision | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float = 0.0
    context: dict[str, Any] | None = None  # similar earlier decisions: shown, never applied

    def view(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "turn_id": self.live.turn_id,
            "session_id": self.live.session_id,
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "hint": self.hint,
            "status": self.status,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "context": self.context,
        }


class ChatApprovals:
    """One-use decisions bound to the turn and call that asked; every failure denies."""

    def __init__(
        self,
        watch: TurnWatch,
        *,
        reviewer: str,
        timeout: float,
        history: History | None = None,
    ) -> None:
        self.watch = watch
        self.reviewer = reviewer
        self.timeout = timeout
        self.history = history  # stored approval views of ended turns, for approval memory
        self._lock = threading.Lock()
        self._items: OrderedDict[str, _Pending] = OrderedDict()

    def port(self) -> CallbackApproval:
        """The ApprovalPort the host's executor uses."""
        return CallbackApproval(self.ask)

    async def ask(self, request: ApprovalRequest) -> ApprovalDecision:
        """Wait for the reviewer's decision; deny on timeout or any error (fail closed)."""
        try:
            return await self._wait(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_failure(
                logger, logging.ERROR, exc, "chat: approval %s failed; denied", request.approval_id
            )
            return _deny("the approval desk failed")

    async def _wait(self, request: ApprovalRequest) -> ApprovalDecision:
        live = self.watch.find(request.run_id)
        if live is None:
            logger.warning("chat: approval for run %s outside a chat turn; denied", request.run_id)
            return _deny("not asked in a chat turn")
        loop = asyncio.get_running_loop()
        arguments = preview(request.arguments)
        item = _Pending(
            approval_id=request.approval_id,
            live=live,
            call_id=request.call_id,
            tool=request.tool,
            arguments=arguments,
            hint=request.hint,
            future=loop.create_future(),
            loop=loop,
            context=self._context(request, arguments),
        )
        with self._lock:
            if request.approval_id in self._items:
                logger.warning("chat: approval id %s asked twice; denied", request.approval_id)
                return _deny("this approval was already asked")
            self._items[request.approval_id] = item
            self._prune()
        live.push("approval", **item.view())
        logger.info(
            "chat: approval %s for %s waits (turn %s)", item.approval_id, item.tool, live.turn_id
        )
        try:
            decision = await asyncio.wait_for(item.future, self.timeout)
        except TimeoutError:
            decision = self._expired(item)
        if item.status in ("timed-out", "stopped"):
            _announce_expired(request, item.status)
        return decision

    def _expired(self, item: _Pending) -> ApprovalDecision:
        with self._lock:
            if item.decision is not None:  # decided as the wait ran out: the reviewer counts
                return item.decision
            item.status = "timed-out"
            item.reason = f"no decision within {self.timeout:g} s"
        logger.info("chat: approval %s timed out; denied", item.approval_id)
        item.live.push("approval", **item.view())
        return _deny(item.reason)

    def decide(self, approval_id: str, *, approved: bool, reason: str = "") -> dict[str, Any]:
        """Record the reviewer's one decision and wake the waiting call (any thread).

        Raises:
            ApprovalNotFound: no request has this id.
            ApprovalAlreadyDecided: it was decided, timed out or stopped already.
        """
        text = str(reason or "")[:MAX_REASON]
        with self._lock:
            item = self._items.get(approval_id)
            if item is None:
                raise ApprovalNotFound(approval_id)
            if item.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=item.status)
            item.status = "approved" if approved else "denied"
            item.reason, item.decided_by = text, self.reviewer
            item.decided_at = time.time()
            item.decision = ApprovalDecision(
                approved=approved, reason=text, decided_by=self.reviewer, decided_at=item.decided_at
            )
        logger.info("chat: approval %s %s by %s", approval_id, item.status, self.reviewer)
        _resolve_soon(item, item.decision)
        item.live.push("approval", **item.view())
        return item.view()

    def close_turn(self, live: LiveTurn, status: str) -> list[dict[str, Any]]:
        """Deny the turn's still-pending requests as `status`; every request's view, in order."""
        ended: list[_Pending] = []
        with self._lock:
            items = [item for item in self._items.values() if item.live is live]
            for item in items:
                if item.status == "pending":
                    item.status = "timed-out" if status == "timed-out" else "stopped"
                    item.reason = _ENDED.get(status, "the turn ended")
                    ended.append(item)
        for item in ended:
            logger.info("chat: approval %s denied: %s", item.approval_id, item.reason)
            _resolve_soon(item, _deny(item.reason))
            live.push("approval", **item.view())
        return [item.view() for item in items]

    def _context(self, request: ApprovalRequest, arguments: str) -> dict[str, Any] | None:
        """Similar earlier decisions, compared on the masked preview; None when that fails."""
        try:
            with self._lock:
                views = [item.view() for item in reversed(self._items.values())]
            stored = list(self.history()) if self.history is not None else []
            shown = request.model_copy(update={"arguments": arguments_of(arguments)})
            return context_view(shown, [*views, *stored])
        except Exception as exc:  # advice only: the request is asked without it
            log_failure(logger, logging.WARNING, exc, "chat: approval memory failed")
            return None

    def _prune(self) -> None:
        """Forget the oldest decided requests past `MAX_KEPT`. Caller holds the lock."""
        while len(self._items) > MAX_KEPT:
            oldest = next(iter(self._items.values()))
            if oldest.status == "pending":
                break
            self._items.popitem(last=False)


def _announce_expired(request: ApprovalRequest, outcome: str) -> None:
    """Send `approval.expired` on the waiting call's sink: nobody decided in time."""
    ctx = current_context()
    if ctx is None or ctx.sink is None:
        return
    event = make_event(
        EventType.APPROVAL_EXPIRED,
        ctx,
        tool=request.tool,
        approval_id=request.approval_id,
        outcome=outcome,
    )
    emit(ctx.sink, event)


def _deny(reason: str) -> ApprovalDecision:
    return ApprovalDecision(
        approved=False, reason=reason, decided_by="chat", decided_at=time.time()
    )


def _resolve_soon(item: _Pending, decision: ApprovalDecision) -> None:
    """Resolve the waiting future on its own loop, from any thread."""

    def resolve() -> None:
        if not item.future.done():
            item.future.set_result(decision)

    try:
        item.loop.call_soon_threadsafe(resolve)
    except RuntimeError:  # the loop closed: nothing waits any more
        logger.debug("chat: approval %s: its loop is closed", item.approval_id)
