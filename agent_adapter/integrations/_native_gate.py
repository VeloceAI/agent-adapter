"""Shared steps of the native approval gates: ask the port, one-use approval tokens.

Rule (security): a framework signal may only add a gate, never remove one. A call whose
`confirm` rule says yes gets `ctx.approval` only from a token this package issued after
its own port approved, bound to the gate, the native call id, the tool and the
`request_digest` of the validated arguments, and taken once. Gates evaluate a callable
`confirm` rule with a bare `ToolContext(run_id, call_id)`; when that answer differs from
the executor's (full context), the executor asks its own port again: fail closed.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.approval.batch import ask_each
from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.approval.store import request_digest
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalPending,
    IntegrationError,
    ToolArgsInvalid,
    ToolUsageError,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall
from agent_adapter.integrations._expiring import ExpiringTable
from agent_adapter.integrations.base import MODEL_VISIBLE_ERRORS, ToolRunner, model_visible_error
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

TOKEN_TTL_S = 60.0  # how long an approved call may take to reach the tool
MAX_TOKENS = 1024
NO_CALL_ID = "the framework gave this call no id, so its approval cannot be bound to it"


def pending_message(exc: ApprovalPending) -> str:
    """The model-visible text for a call whose approval is still pending."""
    return f"approval required: {exc.approval_id}"


def dump_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validated arguments as the executor shows them to approvers; None when invalid."""
    try:
        validated = spec.validate_args(arguments)
    except ToolArgsInvalid:
        return None
    dumped: dict[str, Any] = validated.model_dump(mode="json", by_alias=True, fallback=str)
    return dumped


def arguments_digest(tool: str, arguments: Mapping[str, Any]) -> str | None:
    """`request_digest` of a call of `tool` with these arguments; None when not plain JSON."""
    request = ApprovalRequest(
        approval_id="", run_id="", call_id="", tool=tool, arguments=dict(arguments)
    )
    try:
        return request_digest(request)
    except ToolUsageError:
        return None


def approval_request(
    tool: str, arguments: Mapping[str, Any], *, run_id: str, call_id: str
) -> ApprovalRequest:
    """A fresh approval request for one call."""
    return ApprovalRequest(
        approval_id=new_id("apr_"),
        run_id=run_id,
        call_id=call_id,
        tool=tool,
        arguments=dict(arguments),
        created_at=time.time(),
    )


def denial(reason: str, *, by: str) -> ApprovalDecision:
    """A denial decided by a bridge itself (fails closed)."""
    return ApprovalDecision(approved=False, reason=reason, decided_by=by, decided_at=time.time())


async def needs_approval(
    spec: ToolSpec, arguments: Mapping[str, Any], *, run_id: str, call_id: str
) -> bool:
    """Whether the confirm rule asks for approval of these (validated) arguments."""
    call = ToolCall(name=spec.name, arguments=dict(arguments), id=call_id)
    return await evaluate_confirm(spec.confirm, call, ToolContext(run_id=run_id, call_id=call_id))


async def gate_decision(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    port: ApprovalPort,
    integration: str,
    call_id: str,
    run_id: str,
    edits: bool = False,
) -> ApprovalDecision | None:
    """Ask `port` about one call of `spec`; None when no approval is needed.

    Invalid arguments also give None: the executor rejects them before approval. A call
    without an id is denied (its approval could not be bound to it). Without `edits`, an
    approval that edits the arguments becomes a denial.

    Raises:
        ApprovalPending: the port defers the decision.
    """
    dumped = dump_arguments(spec, arguments)
    if dumped is None or not await needs_approval(spec, dumped, run_id=run_id, call_id=call_id):
        return None
    if not call_id:
        logger.warning("%s: %s has no call id; denied", integration, spec.name)
        return denial(NO_CALL_ID, by=integration)
    decision = await port.request(
        approval_request(spec.name, dumped, run_id=run_id, call_id=call_id)
    )
    if decision.approved and decision.edited_args is not None and not edits:
        logger.warning(
            "%s: approval of %s (call %s) edits its arguments, which %s cannot apply; denied",
            integration,
            spec.name,
            call_id,
            integration,
        )
        return denial(f"argument edits are not supported by {integration}", by=integration)
    return decision


async def decide_calls(
    port: ApprovalPort,
    calls: Sequence[tuple[str, Mapping[str, Any], str]],
    registry: Registry,
    *,
    run_id: str,
    integration: str,
    tokens: TokenTable,
) -> list[ApprovalDecision]:
    """Decisions for paused `(tool, arguments, call_id)` calls, in order, via `ask_each`.

    Decided here without asking: no call id or invalid arguments (denied), a `confirm` rule
    that says no (approved). An approval that edits arguments is refused (denied): these
    frameworks cannot apply edits. Each approved registry call gets a one-use token, bound
    to `run_id`, its call id, tool and arguments, that the call takes when it runs.

    Raises:
        ApprovalPending: a decision is still open (every call was asked first).
    """
    planned = [
        await plan_call(registry, name, arguments, call_id, run_id, integration)
        for name, arguments, call_id in calls
    ]
    asked = iter(await ask_each(port, [p for p in planned if isinstance(p, ApprovalRequest)]))
    decisions: list[ApprovalDecision] = []
    for item in planned:
        if not isinstance(item, ApprovalRequest):
            decisions.append(item)
            continue
        decision = next(asked)
        if decision.approved and decision.edited_args is not None:
            logger.warning("%s: approval of %s edits its arguments; denied", integration, item.tool)
            decision = denial(f"argument edits are not supported by {integration}", by=integration)
        elif decision.approved and item.tool in registry:
            _grant(tokens, item, decision, integration)
        decisions.append(decision)
    return decisions


def _grant(
    tokens: TokenTable, request: ApprovalRequest, decision: ApprovalDecision, integration: str
) -> None:
    if not request.run_id:
        logger.warning(
            "%s: %s approved without a run id; the executor asks again when it runs",
            integration,
            request.tool,
        )
        return
    tokens.issue(
        request.tool, request.arguments, decision, call_id=request.call_id, run_id=request.run_id
    )


async def plan_call(
    registry: Registry,
    name: str,
    arguments: Mapping[str, Any],
    call_id: str,
    run_id: str,
    integration: str,
) -> ApprovalDecision | ApprovalRequest:
    """A decision made without asking (no call id, bad arguments, rule says no), or the request."""
    if not call_id:
        return denial(NO_CALL_ID, by=integration)
    if name not in registry:
        return approval_request(name, arguments, run_id=run_id, call_id=call_id)
    spec = registry.get(name)
    dumped = dump_arguments(spec, arguments)
    if dumped is None:
        return denial("invalid arguments", by=integration)
    if not await needs_approval(spec, dumped, run_id=run_id, call_id=call_id):
        return ApprovalDecision(approved=True, reason="not required", decided_by=integration)
    return approval_request(name, dumped, run_id=run_id, call_id=call_id)


def take_approval(
    tokens: TokenTable, spec: ToolSpec, arguments: Mapping[str, Any], ctx: ToolContext
) -> ApprovalDecision | None:
    """The decision of the token issued for exactly this call (run, call id, arguments)."""
    dumped = dump_arguments(spec, arguments)
    if dumped is None or not ctx.call_id or not ctx.run_id:
        return None
    token = tokens.take(spec.name, dumped, call_id=ctx.call_id, run_id=ctx.run_id)
    return token.decision if token is not None else None


async def run_bound(
    runner: ToolRunner, spec: ToolSpec, arguments: Mapping[str, Any], ctx: ToolContext
) -> Any:
    """Run one call; `MODEL_VISIBLE_ERRORS` become `{"error": …}`, everything else propagates."""
    try:
        return await runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        return model_visible_error(exc)


def check_framework(
    integration: str,
    version: str,
    supported: tuple[tuple[int, ...], tuple[int, ...]],
    relied_on: Mapping[str, bool],
) -> None:
    """Refuse a framework release outside the extra's range or missing what a bridge reads.

    `supported` is `(low, high)`, compared at `len(low)` parts: `((0, 2, 150), (0, 3, 0))`.

    Raises:
        IntegrationError: the version is outside `[low, high)` or a relied-on attribute is
            missing (the bridge's approval handling would no longer be sound).
    """
    low, high = supported
    found = _release(version, len(low))
    if not low <= found < high:
        raise IntegrationError(
            f"{integration} {version} is outside the supported range "
            f"{_dotted(low)} to below {_dotted(high)}",
            integration=integration,
        )
    missing = sorted(name for name, present in relied_on.items() if not present)
    if missing:
        raise IntegrationError(
            f"{integration} {version} lacks {', '.join(missing)}, which this bridge relies on",
            integration=integration,
        )


def _release(version: str, size: int) -> tuple[int, ...]:
    """`version`'s first `size` numeric parts, zero-padded: "0.2.150rc1" -> (0, 2, 150)."""
    parts: list[int] = []
    for piece in version.split(".")[:size]:
        digits = re.match(r"\d+", piece)
        if digits is None:
            break
        parts.append(int(digits.group()))
    return (*parts, *(0,) * (size - len(parts)))


def _dotted(release: tuple[int, ...]) -> str:
    """(0, 2, 150) -> "0.2.150"."""
    return ".".join(map(str, release))


def json_text(value: Any) -> Any:
    """`value` as JSON text, for frameworks that show a tool result to the model via `str()`."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class ApprovalToken:
    """One approved call: tool, call id, argument digest, the decision and its run."""

    tool: str
    call_id: str
    digest: str
    decision: ApprovalDecision
    run_id: str
    expires_at: float


class TokenTable:
    """One gate's one-use approval tokens: issued after its port approved, taken once."""

    def __init__(
        self,
        *,
        ttl: float = TOKEN_TTL_S,
        max_tokens: int = MAX_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tokens: ExpiringTable[int, ApprovalToken] = ExpiringTable(
            ttl=ttl, max_entries=max_tokens, clock=clock, label="approval token(s)"
        )
        self._keys = itertools.count()
        self.evicted = 0  # live tokens dropped so far to stay within `max_tokens`

    def __len__(self) -> int:
        return len(self._tokens)

    def issue(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        decision: ApprovalDecision,
        *,
        call_id: str,
        run_id: str,
    ) -> ApprovalToken | None:
        """Record an approved call; None (nothing stored) without a call id or JSON args."""
        digest = arguments_digest(tool, arguments)
        if not call_id or digest is None:
            logger.warning("no approval token for %s (call %r): cannot bind it", tool, call_id)
            return None
        token = ApprovalToken(tool, call_id, digest, decision, run_id, self._tokens.deadline())
        for dropped in self._tokens.add(next(self._keys), token, expires_at=token.expires_at):
            self.evicted += 1
            logger.warning(
                "approval tokens full: dropped %s (call %s)", dropped.tool, dropped.call_id
            )
        return token

    def take(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None,
        run_id: str | None = None,
    ) -> ApprovalToken | None:
        """Consume the live token for this tool, arguments, call id (`None`: any) and run."""
        digest = arguments_digest(tool, arguments)
        if digest is None or call_id == "":
            return None

        def matches(token: ApprovalToken) -> bool:
            return (
                token.tool == tool
                and token.digest == digest
                and (call_id is None or token.call_id == call_id)
                and (run_id is None or token.run_id == run_id)
            )

        return self._tokens.pop_first(matches)

    def clear(self) -> None:
        """Drop every token (the end of a framework step)."""
        dropped = self._tokens.clear()
        if dropped:
            logger.debug("dropping %d unused approval token(s)", dropped)
