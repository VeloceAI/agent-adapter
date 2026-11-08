"""`Explanation`: which policy rules apply to one tool, as short lines for people and the viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.hooks.builtin.guards.base import ContentGuard
from agent_adapter.memory.tools import FORGET, RECALL, REMEMBER
from agent_adapter.policy._scope import Tri, both, caller_matches, describe_selector, tool_matches
from agent_adapter.policy.condition import Condition
from agent_adapter.policy.model import Selector

if TYPE_CHECKING:
    from agent_adapter.policy.policy import Policy

MEMORY_TOOLS = (REMEMBER, RECALL, FORGET)


@dataclass(frozen=True)
class Explanation:
    """The rules for one tool: `allowed` is None when it depends on the caller."""

    tool: str
    env: str
    allowed: Tri
    lines: tuple[str, ...]

    def __str__(self) -> str:
        return "\n".join(self.lines)


def explain(policy: Policy, tool: str | ToolSpec, ctx: ToolContext | None) -> Explanation:
    """The lines for `tool` under `policy`, for `ctx` (or any caller when None)."""
    name, spec = (tool.name, tool) if isinstance(tool, ToolSpec) else (tool, None)
    allowed, reason = policy.access(tool, ctx)
    word = {True: "allowed", False: "denied", None: "allowed for some callers"}[allowed]
    lines = [f"{word} ({reason})"]
    lines += _approvals(policy, name, spec, ctx)
    for where, rule, hook in policy.guards:
        if _applies(tool_matches(rule, name, spec), rule, ctx, policy) is not False:
            stages = ", ".join(sorted(hook.guard.stages))
            lines.append(f"guard {rule.guard}: {hook.guard.action} on {stages} ({where})")
    for where, budget, _ in policy.budgets:
        window = f" per {budget.per}" if budget.per else ""
        lines.append(
            f"budget {budget.name}: {budget.max_usd:g} USD per {budget.scope}{window}, "
            f"{budget.on_exceed} ({where})"
        )
    if policy.rates is not None:
        for kind, rate in policy.rates.rates.items():
            if spec is None or spec.kind == kind or kind == "llm":
                lines.append(f"rate limit {kind}: {rate.calls} per {rate.per:g} s")
    if policy.spec.limits and policy.spec.limits.max_calls:
        lines.append(f"run limit: {policy.spec.limits.max_calls} calls (limits)")
    if name in MEMORY_TOOLS and policy.spec.memory:
        memory = policy.spec.memory
        kept = f"{memory.retention_days:g} days" if memory.retention_days else "as the backend"
        lines.append(f"memory: save {memory.save}; kept {kept} (memory)")
    return Explanation(name, policy.env, allowed, tuple(lines))


def _approvals(
    policy: Policy, name: str, spec: ToolSpec | None, ctx: ToolContext | None
) -> list[str]:
    lines: list[str] = []
    if spec is not None and spec.confirm is True:
        lines.append("approval: always (set in code; a policy cannot remove it)")
    elif spec is not None and spec.confirm is not False:
        lines.append("approval: by the code's own rule")
    for item in policy.confirm_items(name, spec):
        found = _applies(tool_matches(item.selector, name, spec), item.selector, ctx, policy)
        if found is False:
            continue
        if isinstance(item.ask, Condition):
            what = f"when {item.ask.shown()}"
        elif isinstance(item.ask, ContentGuard):
            what = f"when guard {item.ask.name} flags the arguments"
        else:
            what = "always"
        who = f" [{describe_selector(item.selector)}]" if found is None else ""
        lines.append(f"approval: {what} ({item.where}){who}")
    return lines


def _applies(found: Tri, rule: Selector, ctx: ToolContext | None, policy: Policy) -> Tri:
    return both(found, caller_matches(rule, ctx, policy.team_of))
