"""`Policy`: one declarative file applied to an executor, so every surface obeys it."""

from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.guards import RunLimits
from agent_adapter.hooks.base import Hook
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets.store import BudgetStore
from agent_adapter.hooks.builtin.guards.base import ContentGuard
from agent_adapter.hooks.builtin.rate_limit import RateLimitHook
from agent_adapter.memory import GuardedMemory
from agent_adapter.memory.tools import REMEMBER
from agent_adapter.policy._build import Stores, build_budget, build_guard, build_rates
from agent_adapter.policy._hooks import ConfirmItem, PolicyConfirm, PolicyGate, ScopedGuard
from agent_adapter.policy._scope import TeamOf, Tri, both, caller_matches, team_from, tool_matches
from agent_adapter.policy.errors import PolicyError
from agent_adapter.policy.explain import Explanation, explain
from agent_adapter.policy.model import BudgetRule, PolicySpec, Selector

logger = logging.getLogger(__name__)

ENV_VAR = "AA_ENV"
_SERIALS = itertools.count(1)
# Per memory backend: its retention as built in code, and each applied policy's bound.
_RETENTION: weakref.WeakKeyDictionary[
    GuardedMemory, tuple[float | None, dict[int, tuple[float, bool]]]
] = weakref.WeakKeyDictionary()
E = TypeVar("E")


class Policy:
    """A validated policy for one environment, ready to `apply` to an executor.

    Build it with `load_policy`. Everything it adds is stricter than the code: approvals are
    only ever added, and the two relaxations need `relax:` plus `allow_relax=True`.

    Raises:
        PolicyError: an unknown environment, an unapproved relaxation, or a bad guard/store.
    """

    def __init__(
        self,
        spec: PolicySpec,
        *,
        env: str | None = None,
        source: str = "",
        allow_relax: bool = False,
        guards: Mapping[str, ContentGuard] | None = None,
        stores: Mapping[str, BudgetStore] | None = None,
        team_of: TeamOf | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.spec = spec
        self.source = source
        self.env = os.environ.get(ENV_VAR, "") if env is None else env
        self._check_env()
        if spec.relax and not allow_relax:
            raise PolicyError(
                "this policy relaxes code defaults: load it with allow_relax=True",
                source=source,
                key="relax",
            )
        self.relax = frozenset(spec.relax)
        self.team_of: TeamOf = team_of if team_of is not None else team_from(spec.team_key)
        self.rules = _active([(f"rules[{i}]", r) for i, r in enumerate(spec.rules)], self.env)
        self.guards = [
            (
                where,
                rule,
                ScopedGuard(build_guard(rule, where, guards or {}, source), rule, self.team_of),
            )
            for where, rule in _active(
                [(f"guards[{i}]", g) for i, g in enumerate(spec.guards)], self.env
            )
        ]
        made = Stores(stores or {}, base_dir, source)
        self.budgets: list[tuple[str, BudgetRule, BudgetHook]] = [
            (where, rule, build_budget(rule, where, made, self.team_of))
            for where, rule in _active(
                [(f"budgets[{i}]", b) for i, b in enumerate(spec.budgets)], self.env
            )
        ]
        self.rates: RateLimitHook | None = build_rates(
            _active([(f"rate_limits[{i}]", r) for i, r in enumerate(spec.rate_limits)], self.env),
            source,
        )
        self.gate = PolicyGate(self)
        self._serial = next(_SERIALS)  # a stable key that is never reused, unlike id()
        self._effective: weakref.WeakKeyDictionary[ToolSpec, ToolSpec] = weakref.WeakKeyDictionary()

    def __repr__(self) -> str:
        return f"Policy(source={self.source!r}, env={self.env!r})"

    def _check_env(self) -> None:
        known = self.spec.environments
        if known is not None and self.env not in known:
            shown = repr(self.env) if self.env else f"none ({ENV_VAR} is not set)"
            raise PolicyError(
                f"the environment is {shown}; this policy needs one of {known}",
                source=self.source,
                key="environments",
            )

    @property
    def default(self) -> str:
        """`allow` or `deny` for tools no rule names, in this environment."""
        chosen = self.spec.default
        return chosen if isinstance(chosen, str) else chosen.get(self.env, "deny")

    # --- applying ---------------------------------------------------------------------------

    def hooks(self) -> tuple[Hook, ...]:
        """The hooks this policy adds, the access gate first."""
        added: list[Hook] = [self.gate, *(hook for _, _, hook in self.guards)]
        added += [hook for _, _, hook in self.budgets]
        if self.rates is not None:
            added.append(self.rates)
        return tuple(added)

    def apply(self, executor: Executor) -> Executor:
        """A new executor running every call under this policy; `executor` is unchanged.

        Applying a second policy keeps the first: both must allow a call, either may ask.
        """
        current = executor.policy
        parts = current.parts if isinstance(current, PolicyChain) else (current,)
        if self in parts:
            return executor
        chained = self if current is None else PolicyChain((*parts, self))
        logger.info("policy %s applied (env %r)", self.source or "(in code)", self.env)
        return executor.replace(policy=chained)

    def limits(self, current: RunLimits) -> RunLimits:
        """`current` with the policy's `max_calls`: lower only, unless relaxed."""
        wanted = self.spec.limits.max_calls if self.spec.limits else None
        if wanted is None or current.max_calls == wanted:
            return current
        lower = current.max_calls is None or wanted < current.max_calls
        if lower or "limits.max_calls" in self.relax:
            return dataclasses.replace(current, max_calls=wanted)
        return current

    def effective(self, spec: ToolSpec) -> ToolSpec:
        """`spec` with the policy's approvals added to its own `confirm` (never removed)."""
        cached = self._effective.get(spec)
        if cached is not None:
            return cached
        items = self.confirm_items(spec.name, spec)
        found = spec
        if items and spec.confirm is not True:
            found = dataclasses.replace(
                spec, confirm=PolicyConfirm(spec.confirm, items, self.team_of)
            )
        self._effective[spec] = found
        return found

    def describe(self, spec: ToolSpec) -> list[str]:
        """The rules that may apply to `spec`, one short line each (the viewer's Tools tab)."""
        return list(self.explain(spec).lines)

    def explain(self, tool: str | ToolSpec, ctx: ToolContext | None = None) -> Explanation:
        """Which rules apply to `tool` for `ctx`; without a context, caller rules say "some"."""
        return explain(self, tool, ctx)

    def tools(self, registry: Registry) -> Registry:
        """The tools of `registry` a model may be shown: those not denied to every caller."""
        keep = [spec.name for spec in registry if self.access(spec, None)[0] is not False]
        return registry.subset(names=keep) if keep else Registry()

    def apply_memory(self, memory: E) -> E:
        """Shorten `memory`'s retention to the policy's (longer only when relaxed); returns it.

        Raises:
            PolicyError: `memory` is not one of agent_adapter.memory's backends.
        """
        days = self.spec.memory.retention_days if self.spec.memory else None
        if days is None:
            return memory
        if not isinstance(memory, GuardedMemory):
            raise PolicyError("apply_memory needs an agent_adapter.memory backend", key="memory")
        code, bounds = _RETENTION.setdefault(memory, (memory.retention_days, {}))
        bounds[self._serial] = (float(days), "memory.retention_days" in self.relax)
        # Every applied policy bounds the backend; the code's own value holds unless one of
        # them relaxes it, so a relaxation never loosens another policy's bound.
        limits = [bound for bound, _ in bounds.values()]
        if code is not None and not any(relaxed for _, relaxed in bounds.values()):
            limits.append(code)
        memory.retention_days = min(limits)
        logger.info("memory retention set to %g days by policy", memory.retention_days)
        return memory

    def check_sandbox(self, sandbox: E) -> E:
        """Refuse a sandbox the policy does not accept; returns it unchanged.

        Raises:
            PolicyError: not isolated, or network access, where the policy forbids it.
        """
        rule = self.spec.sandbox
        if rule is None:
            return sandbox
        if rule.require_isolated and getattr(sandbox, "isolated", False) is not True:
            raise PolicyError("the sandbox is not isolated from the host", key="sandbox")
        if rule.deny_network and getattr(sandbox, "network", True) is not False:
            raise PolicyError("the sandbox allows network access", key="sandbox")
        return sandbox

    # --- deciding ---------------------------------------------------------------------------

    def access(self, tool: str | ToolSpec, ctx: ToolContext | None) -> tuple[Tri, str]:
        """Whether `tool` may run for `ctx`, and why; None when it depends on the caller."""
        name, spec = (tool.name, tool) if isinstance(tool, ToolSpec) else (tool, None)
        save = self.spec.memory.save if self.spec.memory else "allow"
        if name == REMEMBER and save == "deny":
            return False, "memory.save is deny"
        allowed: Tri = self.default == "allow"
        reason = f"default {self.default}"
        maybe_denied = False
        for where, rule in self.rules:
            if rule.allow is None:
                continue
            found = self._matches(rule, name, spec, ctx)
            if found is False:
                continue
            if rule.allow is False:
                if found is True:
                    return False, rule.reason or f"{where} denies it"
                maybe_denied = True
            elif found is True:
                allowed, reason = True, f"{where} allows it"
            elif allowed is False:
                allowed, reason = None, f"{where} allows some callers"
        if allowed is True and maybe_denied:
            return None, "a rule denies some callers"
        return allowed, reason

    def confirm_items(self, name: str, spec: ToolSpec | None) -> list[ConfirmItem]:
        """The policy's reasons this tool may need approval (callers checked per call)."""
        items = [
            ConfirmItem(where, rule, rule.confirm)
            for where, rule in self.rules
            if rule.confirm is not None and tool_matches(rule, name, spec) is not False
        ]
        items += [
            ConfirmItem(where, rule, hook.guard)
            for where, rule, hook in self.guards
            if hook.guard.action == "approve" and tool_matches(rule, name, spec) is not False
        ]
        if name == REMEMBER and self.spec.memory and self.spec.memory.save == "confirm":
            items.append(ConfirmItem("memory.save", Selector(tools=[REMEMBER]), True))
        return items

    def _matches(
        self, rule: Selector, name: str, spec: ToolSpec | None, ctx: ToolContext | None
    ) -> Tri:
        return both(tool_matches(rule, name, spec), caller_matches(rule, ctx, self.team_of))


class PolicyChain:
    """Several policies at once: each rewrites the spec in turn, so all of them apply."""

    def __init__(self, parts: tuple[Any, ...]) -> None:
        self.parts = parts

    def hooks(self) -> tuple[Hook, ...]:
        """Every policy's hooks, in the order they were applied."""
        return tuple(hook for part in self.parts for hook in part.hooks())

    def limits(self, current: RunLimits) -> RunLimits:
        """The strictest `max_calls` any policy gives; a relaxation raises only the code's own."""
        found = [part.limits(current).max_calls for part in self.parts]
        capped = [value for value in found if value is not None]
        wanted = min(capped) if capped else None
        if wanted == current.max_calls:
            return current
        return dataclasses.replace(current, max_calls=wanted)

    def effective(self, spec: ToolSpec) -> ToolSpec:
        """`spec` through every policy in order."""
        for part in self.parts:
            spec = part.effective(spec)
        return spec

    def describe(self, spec: ToolSpec) -> list[str]:
        """Every policy's lines for `spec`."""
        return [line for part in self.parts for line in part.describe(spec)]


def _active(entries: list[tuple[str, Any]], env: str) -> list[tuple[str, Any]]:
    """The entries for `env`: those without `envs`, or listing it."""
    return [(where, rule) for where, rule in entries if rule.envs is None or env in rule.envs]
