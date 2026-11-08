"""Run one hook phase over a list of hooks, in registration order."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import (
    ApprovalDenied,
    ApprovalPending,
    BudgetExceeded,
    GuardrailBlocked,
    RunLimitExceeded,
    ToolUsageError,
    log_failure,
    qualified_name,
)
from agent_adapter.core.types import LedgerEntry
from agent_adapter.hooks.base import Hook, LLMInvocation, ShortCircuit, hook_name

logger = logging.getLogger(__name__)

SUBJECTS = ("tool", "llm", "node")
BEFORE = frozenset(f"before_{s}" for s in SUBJECTS)
AFTER = frozenset(f"after_{s}" for s in SUBJECTS)
ON_ERROR = frozenset(f"on_{s}_error" for s in SUBJECTS)

# Outcomes a policy hook raises on purpose (budget, guardrail, approval, limits): not hook bugs.
POLICY_OUTCOMES = (
    BudgetExceeded,
    GuardrailBlocked,
    ApprovalDenied,
    ApprovalPending,
    RunLimitExceeded,
)


async def run_before(hooks: Sequence[Hook], method: str, inv: Any) -> ShortCircuit | None:
    """Call `method` on each hook; the first ShortCircuit wins and stops the phase."""
    _check(method, BEFORE)
    for hook in hooks:
        result = await _call(hook, method, inv)
        if result is not None:
            return _short_circuit(hook, method, result)
    return None


async def run_after(hooks: Sequence[Hook], method: str, inv: Any, value: Any) -> Any:
    """Thread `value` through each hook; a None return keeps the current value."""
    _check(method, AFTER)
    for hook in hooks:
        result = await _call(hook, method, inv, value)
        if result is not None:
            value = result
    return value


async def run_on_error(
    hooks: Sequence[Hook], method: str, inv: Any, error: BaseException
) -> ShortCircuit | None:
    """Offer `error` to each hook; the first ShortCircuit recovers and stops the phase."""
    _check(method, ON_ERROR)
    for hook in hooks:
        result = await _call(hook, method, inv, error)
        if result is not None:
            return _short_circuit(hook, method, result)
    return None


async def run_end(hooks: Sequence[Hook], inv: LLMInvocation, billed: LedgerEntry | None) -> None:
    """`on_llm_end` on every hook; a failing hook is logged and the others still run."""
    for hook in hooks:
        try:
            await hook.on_llm_end(inv, billed)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.ERROR, exc, "hook %s failed in on_llm_end", hook_name(hook))


async def _call(hook: Hook, method: str, *args: Any) -> Any:
    try:
        return await getattr(hook, method)(*args)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        if isinstance(exc, POLICY_OUTCOMES):
            # The hook that decides logs the outcome and the executor logs the result; a third
            # line here would only repeat them.
            logger.debug(
                "hook %s stopped %s: %s", hook_name(hook), method, qualified_name(type(exc))
            )
        else:
            log_failure(logger, logging.ERROR, exc, "hook %s failed in %s", hook_name(hook), method)
        raise


def _short_circuit(hook: Hook, method: str, result: Any) -> ShortCircuit:
    if not isinstance(result, ShortCircuit):
        raise ToolUsageError(
            f"hook {hook_name(hook)!r}.{method} must return ShortCircuit or None, "
            f"got {type(result).__name__}"
        )
    return result


def _check(method: str, allowed: frozenset[str]) -> None:
    if method not in allowed:
        raise ToolUsageError(f"{method!r} is not one of {sorted(allowed)}")
