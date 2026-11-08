"""Who and what an entry applies to: tool globs, kinds, tags, teams and users."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase

from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.hooks.builtin.budgets import principal_of
from agent_adapter.policy.model import Selector

TeamOf = Callable[[ToolContext], str]
Tri = bool | None  # None: decided by the caller (team or user), or by a spec not given


def team_from(key: str) -> TeamOf:
    """`ctx.extras[key]`, else the principal's `claims[key]`; "" when neither is a string."""

    def team_of(ctx: ToolContext) -> str:
        value = ctx.extras.get(key)
        if isinstance(value, str) and value:
            return value
        principal = ctx.extras.get("principal")
        claims = (
            principal.get("claims")
            if isinstance(principal, Mapping)
            else getattr(principal, "claims", None)
        )
        found = claims.get(key) if isinstance(claims, Mapping) else None
        return found if isinstance(found, str) else ""

    return team_of


def tool_matches(selector: Selector, name: str, spec: ToolSpec | None) -> Tri:
    """Whether the tool part matches: False, True, or None when kinds/tags need a spec."""
    if not any(fnmatchcase(name, pattern) for pattern in selector.tools):
        return False
    if selector.kinds is None and selector.tags is None:
        return True
    if spec is None:
        return None
    if selector.kinds is not None and spec.kind not in selector.kinds:
        return False
    return selector.tags is None or bool(spec.tags & set(selector.tags))


def caller_matches(selector: Selector, ctx: ToolContext | None, team_of: TeamOf) -> Tri:
    """Whether the team and user part matches `ctx`; None without a context to ask."""
    if selector.teams is None and selector.users is None:
        return True
    if ctx is None:
        return None
    if selector.teams is not None and team_of(ctx) not in selector.teams:
        return False
    return selector.users is None or principal_of(ctx) in selector.users


def both(first: Tri, second: Tri) -> Tri:
    """Three-valued and: False wins, then None."""
    if first is False or second is False:
        return False
    return None if first is None or second is None else True


def describe_selector(selector: Selector) -> str:
    """`tools refund_*, teams support` for explanations."""
    parts = [f"tools {', '.join(selector.tools)}"]
    for label in ("kinds", "tags", "teams"):
        values = getattr(selector, label)
        if values is not None:
            parts.append(f"{label} {', '.join(values)}")
    if selector.users is not None:  # user ids may be emails: counted, never shown
        parts.append(f"{len(selector.users)} listed user(s)")
    return "; ".join(parts)
