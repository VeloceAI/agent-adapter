"""One-use approval tokens carried in ADK confirmation payloads, and `resume_message`.

ADK's `confirmed` flag is a framework signal: it may add a gate, never approve. A call (or a
compiled flow's pause) is approved only by a token this process issued for the decision,
bound to the run, the call id, the tool and `request_digest` of the arguments, and taken once.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter.approval.store import request_digest
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations._expiring import ExpiringTable
from agent_adapter.integrations.adk._support import (
    INTERRUPT_PREFIX,
    REQUEST_CONFIRMATION,
    REQUEST_INPUT,
    TOKEN_KEY,
    genai_types,
)

logger = logging.getLogger(__name__)

TOKEN_TTL_S = 600.0  # from resume_message (or resolve_confirmations) to the resumed run
MAX_TOKENS = 4096
MAX_INTERRUPTS = 10_000


@dataclass(frozen=True)
class _Token:
    run_id: str
    call_id: str
    digest: str
    decision: ApprovalDecision


class ConfirmationTokens:
    """Secret, one-use tokens, each bound to one approval request and holding its decision."""

    def __init__(
        self,
        *,
        ttl: float = TOKEN_TTL_S,
        max_tokens: int = MAX_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tokens: ExpiringTable[str, _Token] = ExpiringTable(
            ttl=ttl, max_entries=max_tokens, clock=clock, label="confirmation token(s)"
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def issue(self, request: ApprovalRequest, decision: ApprovalDecision) -> str | None:
        """A new token for `decision` on `request`; None without run / call id or JSON args."""
        try:
            digest = request_digest(request)
        except ToolUsageError:
            digest = ""
        if not (request.run_id and request.call_id and digest):
            logger.warning("no confirmation token for %s (call %r)", request.tool, request.call_id)
            return None
        secret = secrets.token_urlsafe(24)  # a bearer secret: whoever holds it answers once
        token = _Token(request.run_id, request.call_id, digest, decision)
        for dropped in self._tokens.add(secret, token):
            logger.warning("confirmation tokens full: dropped call %s", dropped.call_id)
        return secret

    def take(self, secret: Any, request: ApprovalRequest) -> ApprovalDecision | None:
        """The decision of the live token `secret` if it was issued for `request`; used up."""
        if not isinstance(secret, str):
            return None
        token = self._tokens.pop(secret)
        if token is None:
            logger.warning(
                "confirmation token for %s (call %s) is unknown or expired", *_ids(request)
            )
            return None
        try:
            digest = request_digest(request)
        except ToolUsageError:
            digest = ""
        if (token.run_id, token.call_id, token.digest) != (request.run_id, request.call_id, digest):
            logger.warning(
                "confirmation token was issued for another call than %s (%s)", *_ids(request)
            )
            return None
        return token.decision

    def clear(self) -> None:
        """Forget every token (tests)."""
        self._tokens.clear()


TOKENS = ConfirmationTokens()


def _ids(request: ApprovalRequest) -> tuple[str, str]:
    return request.tool, request.call_id


def confirmation_parts(response: Any) -> tuple[bool, Any] | None:
    """`(confirmed, token)` of a ToolConfirmation (object or dict); None when malformed.

    The payload must be absent or exactly `{"aa_token": <str>}`: anything else (edited
    arguments, extra keys) is refused, so a payload cannot change what was approved.
    """
    if isinstance(response, Mapping):
        confirmed, payload = response.get("confirmed"), response.get("payload")
    else:
        confirmed, payload = (
            getattr(response, "confirmed", None),
            getattr(response, "payload", None),
        )
    if not isinstance(confirmed, bool):
        return None
    if payload is None:
        return confirmed, None
    if not isinstance(payload, Mapping) or set(payload) != {TOKEN_KEY}:
        return None
    return confirmed, payload[TOKEN_KEY]


def token_decision(
    response: Any, request: ApprovalRequest, tokens: ConfirmationTokens = TOKENS
) -> ApprovalDecision | None:
    """The adapter decision a confirmation response carries for `request`; None: ask the port."""
    parts = confirmation_parts(response)
    if parts is None:
        logger.warning("malformed confirmation for %s (call %s): ignored", *_ids(request))
        return None
    confirmed, secret = parts
    if secret is None:
        logger.warning(
            "confirmation for %s (call %s) carries no adapter token; the port decides",
            *_ids(request),
        )
        return None
    decision = tokens.take(secret, request)
    if decision is not None and decision.approved != confirmed:
        logger.warning("confirmation for %s (call %s) contradicts its token", *_ids(request))
        return None
    return decision


class _Interrupts:
    """The pending request of each interrupt this process raised (bounded, oldest dropped)."""

    def __init__(self, max_entries: int = MAX_INTERRUPTS) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, ApprovalRequest] = OrderedDict()

    def remember(self, interrupt_id: str, request: ApprovalRequest) -> None:
        with self._lock:
            self._pending[interrupt_id] = request
            self._pending.move_to_end(interrupt_id)
            while len(self._pending) > self._max:
                self._pending.popitem(last=False)

    def get(self, interrupt_id: str) -> ApprovalRequest | None:
        with self._lock:
            return self._pending.get(interrupt_id)


INTERRUPTS = _Interrupts()


def resume_message(interrupt_id: str, decision: ApprovalDecision) -> Any:
    """The user `types.Content` answering one interrupt with `decision`.

    For an interrupt a compiled flow raised in this process, the response carries a one-use
    token binding `decision` to the pending request; otherwise it carries none and the
    approval port decides when the run resumes (resolve tool confirmations of an `LlmAgent`
    with `resolve_confirmations`).

    Raises:
        ToolUsageError: `interrupt_id` is not a non-empty str, or `decision` not a decision.
    """
    if not isinstance(interrupt_id, str) or not interrupt_id:
        raise ToolUsageError(f"interrupt_id must be a non-empty str, got {interrupt_id!r}")
    if not isinstance(decision, ApprovalDecision):
        raise ToolUsageError(f"expected an ApprovalDecision, got {type(decision).__name__}")
    response: dict[str, Any] = {"confirmed": decision.approved}
    pending = INTERRUPTS.get(interrupt_id)
    if pending is not None:
        secret = TOKENS.issue(pending, decision)
        if secret is not None:
            response["payload"] = {TOKEN_KEY: secret}
    else:
        logger.info("interrupt %s is not known here; the approval port decides", interrupt_id)
    flow_interrupt = interrupt_id.startswith(INTERRUPT_PREFIX)
    return function_response(
        interrupt_id, REQUEST_INPUT if flow_interrupt else REQUEST_CONFIRMATION, response
    )


def function_response(interrupt_id: str, name: str, response: Mapping[str, Any]) -> Any:
    """A user `Content` with one `FunctionResponse` part."""
    types = genai_types()
    part = types.Part(
        function_response=types.FunctionResponse(
            id=interrupt_id, name=name, response=dict(response)
        )
    )
    return types.Content(role="user", parts=[part])
