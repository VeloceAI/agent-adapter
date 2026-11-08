"""`GET /budget`: the caller's own budgets (limit, spend, holds, window); approvers see any."""

from __future__ import annotations

import logging
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.server._payloads import client_key, refuse
from agent_adapter.server._state import ServiceResponse, State, json_response
from agent_adapter.server.auth import APPROVER_SCOPE, Principal, RequestInfo, principal_ref

logger = logging.getLogger(__name__)

# Scopes a caller can be charged under across runs; run and thread budgets are per run
# (see /runs/{run_id}/cost), and a global budget reports everyone's spend: approvers only.
SHOWN = ("principal", "user", "team", "global")


async def budgets(
    state: State, principal: Principal, request: RequestInfo, scope: str | None
) -> ServiceResponse:
    """The budgets that charge `principal`, optionally one `scope`; `?key=` is approver-only."""
    if scope is not None and scope not in SHOWN:
        refuse("bad-request", f"scope must be one of {', '.join(SHOWN)}")
    hooks = [
        hook
        for hook in state.executor.hooks
        if isinstance(hook, BudgetHook) and hook.scope in SHOWN and scope in (None, hook.scope)
    ]
    if not hooks:
        refuse("not-found", "no budget is configured for this scope")
    approver = APPROVER_SCOPE in principal.scopes
    key = client_key(request.query.get("key"), "key")
    if scope == "global" and not approver:
        refuse("forbidden", "only approvers may read the global budget")
    if key and scope is None:
        refuse("bad-request", "key needs a scope: GET /budget/{scope}?key=")
    if key and not approver:
        logger.warning("principal %s refused: budget of another key", principal_ref(principal.id))
        refuse("forbidden", "only approvers may read another key's budget")
    ctx = state.executor.context(extras={"principal": principal})
    shown: list[dict[str, Any]] = []
    for hook in hooks:
        if hook.scope == "global" and not approver:
            continue  # everyone's spend: not a caller's own scope
        try:
            override = key if key and hook.scope != "global" else None
            status = await hook.status(ctx, scope_id=override)
        except ConfigError:
            # A team key that does not resolve for this caller: no team budget charges them.
            logger.info("principal %s has no %s budget", principal_ref(principal.id), hook.scope)
            continue
        shown.append(status.view())
    return json_response(200, {"budgets": shown})
