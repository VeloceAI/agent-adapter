"""The hooks and the confirm rule a policy adds; each wraps an existing building block."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import LLMResponse
from agent_adapter.core.tool import ConfirmRule
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.hooks.base import Hook, LLMInvocation, ShortCircuit, ToolInvocation
from agent_adapter.hooks.builtin.guards.base import ContentGuard
from agent_adapter.policy._scope import TeamOf, both, caller_matches, tool_matches
from agent_adapter.policy.condition import Condition, ConditionError
from agent_adapter.policy.errors import PolicyDenied
from agent_adapter.policy.model import Selector

if TYPE_CHECKING:
    from agent_adapter.policy.policy import Policy

logger = logging.getLogger(__name__)


class PolicyGate(Hook):
    """Refuses, before anything else runs, a tool the policy does not allow for this caller."""

    name = "policy"

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        allowed, reason = self.policy.access(inv.spec, inv.ctx)
        if allowed is not True:
            logger.info("policy denied tool %s (run %s)", inv.spec.name, inv.ctx.run_id)
            raise PolicyDenied(inv.spec.name, call_id=inv.ctx.call_id, reason=reason)
        return None


class ScopedGuard(Hook):
    """A content guard on the selected tools only; model calls only when it selects every tool."""

    name = "policy_guard"

    def __init__(self, guard: ContentGuard, selector: Selector, team_of: TeamOf) -> None:
        self.guard = guard
        self.selector = selector
        self.team_of = team_of
        self.everywhere = selector.tools == ["*"] and selector.kinds is None and not selector.tags

    def selects(self, inv: ToolInvocation) -> bool:
        """True when this call is one the guard watches."""
        found = both(
            tool_matches(self.selector, inv.spec.name, inv.spec),
            caller_matches(self.selector, inv.ctx, self.team_of),
        )
        return found is True

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        if self.guard.action == "approve":  # the policy's confirm rule asks instead
            return None
        if self.selects(inv):
            await self.guard.before_tool(inv)
        return None

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        return await self.guard.after_tool(inv, result) if self.selects(inv) else None

    async def before_llm(self, inv: LLMInvocation) -> ShortCircuit | None:
        if self._llm(inv.ctx):
            await self.guard.before_llm(inv)
        return None

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> LLMResponse | None:
        return await self.guard.after_llm(inv, response) if self._llm(inv.ctx) else None

    def _llm(self, ctx: ToolContext) -> bool:
        return self.everywhere and caller_matches(self.selector, ctx, self.team_of) is True


@dataclass(frozen=True)
class ConfirmItem:
    """One reason a call may need approval: always, a condition, or a guard that flags it."""

    where: str
    selector: Selector
    ask: bool | Condition | ContentGuard


class PolicyConfirm:
    """A `ToolSpec.confirm` rule: the code's own rule, or any policy item that asks."""

    def __init__(self, original: ConfirmRule, items: Sequence[ConfirmItem], team_of: TeamOf):
        self.original = original
        self.items = tuple(items)
        self.team_of = team_of

    def __repr__(self) -> str:
        return f"PolicyConfirm({[item.where for item in self.items]})"

    def includes(self, rule: object) -> bool:
        """True when `rule` (the code's rule, or a guard's `confirm`) is one this rule asks."""
        if rule is self.original or (callable(self.original) and self.original == rule):
            return True
        return any(
            isinstance(item.ask, ContentGuard) and item.ask.confirm == rule for item in self.items
        )

    def __eq__(self, other: object) -> bool:
        # A code guard with action="approve" checks `spec.confirm == guard.confirm` before it
        # lets a flagged call go on to approval; the wrapped rule must still answer yes.
        return other is self or self.includes(other)

    __hash__ = object.__hash__

    async def __call__(self, call: ToolCall, ctx: ToolContext) -> bool:
        if self.original is not False and await evaluate_confirm(self.original, call, ctx):
            return True
        for item in self.items:
            if caller_matches(item.selector, ctx, self.team_of) is True and await _asks(
                item, call, ctx
            ):
                return True
        return False


async def _asks(item: ConfirmItem, call: ToolCall, ctx: ToolContext) -> bool:
    if isinstance(item.ask, bool):
        return item.ask
    if isinstance(item.ask, ContentGuard):
        return await item.ask.confirm(call, ctx)
    try:
        return item.ask.evaluate(call.arguments)
    except ConditionError:
        # Undecidable (a missing or odd value): ask a person rather than let it through.
        logger.warning(
            "policy condition %s undecidable for tool %s; approval asked", item.where, call.name
        )
        return True
