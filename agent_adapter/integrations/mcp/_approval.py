"""Client-side approval over MCP 2026-07-28: an `input_required` form elicitation round trip."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.integrations.mcp._resume import (
    INVALID_STATE,
    STATE_TTL_S,
    SpentStates,
    record_client_decision,
    refuse,
)

if TYPE_CHECKING:
    import mcp.types as t
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server

    from agent_adapter.integrations.mcp._scope import PrincipalHook

logger = logging.getLogger(__name__)

APPROVAL_INPUT_KEY = "agent-adapter/approval"  # the key of our request in `inputRequests`
DECIDED_BY = "mcp-client"
STATE_VERSION = 2  # 2: carries `iat`
MAX_MESSAGE_ARGS = 2000  # characters of JSON arguments shown in the approval prompt

APPROVAL_FORM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean", "title": "Approve this call", "default": False},
        "reason": {"type": "string", "title": "Reason (optional)"},
    },
    "required": ["approved"],
}

_ACTION_REASONS = {
    "decline": "declined by the MCP client",
    "cancel": "cancelled by the MCP client",
}


def install_state_boundary(
    server: Server,
    *,
    audience: str,
    keys: Sequence[bytes | str] | None,
    principal: PrincipalHook | None,
) -> None:
    """Seal `requestState`: encrypted, bound to the tool, its arguments and the principal.

    Raises:
        ConfigError: a key is not bytes / str or shorter than 32 bytes.
    """
    from mcp.server.request_state import RequestStateBoundary, RequestStateSecurity

    try:
        security = RequestStateSecurity(
            keys=list(keys) if keys else [os.urandom(32)], bind_principal=principal
        )
    except (TypeError, ValueError) as exc:
        log_failure(logger, logging.ERROR, exc, "unusable request_state_keys")
        raise ConfigError(f"request_state_keys: {exc}") from exc
    server.middleware.append(RequestStateBoundary(security, default_audience=audience))


class ClientApproval:
    """Asks the client's user about `confirm` calls; the answer becomes `ctx.approval`.

    Each sealed state is single use: in this process (`SpentStates`) and, when the executor
    parks approvals in an `ApprovalStore`, on every worker sharing that store.
    """

    def __init__(self, store: ApprovalStore | None = None) -> None:
        self.store = store  # where decisions are recorded; else the executor's DeferredApproval
        self.spent = SpentStates()

    async def gate(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: t.CallToolRequestParams,
        spec: ToolSpec,
        args: Mapping[str, Any],
        tctx: ToolContext,
        store: ApprovalStore | None = None,
        *,
        resuming: bool = False,
    ) -> ToolContext | t.InputRequiredResult:
        """The context to run with, or the `input_required` result asking for approval.

        The confirm rule is asked here, before the executor's `before_tool` hooks and run
        guards (they run on the approved retry).

        Raises:
            ToolArgsInvalid: the arguments fail validation (nobody is asked).
            MCPError: an echoed `requestState` this server did not mint, or one already used.
        """
        if params.request_state is not None:
            return await self._resumed(params, spec, args, tctx, self.store or store)
        if resuming or not can_elicit(ctx):  # a resumed call: the port owning it decides
            return tctx
        validated = spec.validate_args(args)
        arguments = validated.model_dump(mode="json", by_alias=True, fallback=str)
        call = ToolCall(name=spec.name, arguments=arguments, id=tctx.call_id)
        if not await evaluate_confirm(spec.confirm, call, tctx):
            return tctx
        logger.info(
            "asking the MCP client to approve tool %s (run %s, call %s)",
            spec.name,
            tctx.run_id,
            tctx.call_id,
        )
        return _ask(spec, call, tctx)

    async def _resumed(
        self,
        params: t.CallToolRequestParams,
        spec: ToolSpec,
        args: Mapping[str, Any],
        tctx: ToolContext,
        store: ApprovalStore | None,
    ) -> ToolContext:
        state = _parse_state(params.request_state)
        decision = decision_from((params.input_responses or {}).get(APPROVAL_INPUT_KEY))
        arguments = spec.validate_args(args).model_dump(mode="json", by_alias=True, fallback=str)
        deadline = float(state["iat"]) + STATE_TTL_S
        if self.spent.clock() >= deadline:
            logger.warning("refusing an expired client-approval state for tool %s", spec.name)
            refuse(INVALID_STATE)
        principal = tctx.extras.get("principal")
        call = (state["run_id"], state["call_id"])
        if not self.spent.use(call, deadline, principal if isinstance(principal, str) else None):
            logger.warning("refusing a reused client-approval state for tool %s", spec.name)
            refuse(INVALID_STATE)
        if store is not None:
            await record_client_decision(
                store, spec, arguments, state["run_id"], state["call_id"], decision
            )
        logger.info(
            "MCP client %s tool %s (run %s, call %s)",
            "approved" if decision.approved else "denied",
            spec.name,
            state["run_id"],
            state["call_id"],
        )
        return dataclasses.replace(
            tctx, run_id=state["run_id"], call_id=state["call_id"], approval=decision
        )


def can_elicit(ctx: ServerRequestContext[Any, Any]) -> bool:
    """True on a 2026-07-28 request from a client that accepts form elicitation."""
    from mcp.types.version import MODERN_PROTOCOL_VERSIONS

    if ctx.protocol_version not in MODERN_PROTOCOL_VERSIONS:
        return False
    capabilities = ctx.session.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    # An empty `elicitation: {}` means form mode (spec back-compat rule).
    return elicitation is not None and (elicitation.form is not None or elicitation.url is None)


def _ask(spec: ToolSpec, call: ToolCall, tctx: ToolContext) -> t.InputRequiredResult:
    import mcp.types as t

    shown = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    if len(shown) > MAX_MESSAGE_ARGS:
        shown = shown[:MAX_MESSAGE_ARGS] + " ..."
    request = t.ElicitRequest(
        params=t.ElicitRequestFormParams(
            message=f"Approve the call to tool {spec.name!r} with arguments {shown}?",
            requested_schema=APPROVAL_FORM,
        )
    )
    state = {"v": STATE_VERSION, "run_id": tctx.run_id, "call_id": tctx.call_id, "iat": time.time()}
    return t.InputRequiredResult(
        input_requests={APPROVAL_INPUT_KEY: request},
        request_state=json.dumps(state, separators=(",", ":")),
    )


def _parse_state(raw: str | None) -> dict[str, Any]:
    """Our plaintext state; the boundary already verified it was minted for this call."""
    try:
        state = json.loads(raw or "")
        ids: dict[str, Any] = {key: state[key] for key in ("run_id", "call_id", "iat")}
        version = state.get("v")
        valid = (
            type(version) is int
            and version == STATE_VERSION
            and all(isinstance(ids[k], str) and ids[k] for k in ("run_id", "call_id"))
            and isinstance(ids["iat"], int | float)
            and not isinstance(ids["iat"], bool)
            and math.isfinite(ids["iat"])
        )
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        valid = False
    if not valid:
        logger.warning("MCP call carried a requestState that is not an approval state")
        refuse(INVALID_STATE)
    return ids


def decision_from(response: Any) -> ApprovalDecision:
    """The decision in the client's elicitation response; anything but an explicit yes denies."""
    from mcp.types import ElicitResult

    now = time.time()
    if not isinstance(response, ElicitResult):
        return ApprovalDecision(
            approved=False,
            reason="no approval response from the MCP client",
            decided_by=DECIDED_BY,
            decided_at=now,
        )
    content = response.content or {}
    reason = content.get("reason")
    reason = reason if isinstance(reason, str) else ""
    if response.action == "accept":
        approved = content.get("approved") is True
        return ApprovalDecision(
            approved=approved,
            reason=reason or ("" if approved else "not approved by the MCP client"),
            decided_by=DECIDED_BY,
            decided_at=now,
        )
    return ApprovalDecision(
        approved=False,
        reason=reason or _ACTION_REASONS.get(response.action, "denied by the MCP client"),
        decided_by=DECIDED_BY,
        decided_at=now,
    )
