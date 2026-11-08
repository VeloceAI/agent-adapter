"""Policy: one file that sets approvals, budgets, guardrails, limits and access per tool and env."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_adapter.hooks.builtin.budgets.store import BudgetStore
from agent_adapter.hooks.builtin.guards.base import ContentGuard
from agent_adapter.policy._load import read_file, validate
from agent_adapter.policy._scope import TeamOf, team_from
from agent_adapter.policy.condition import Condition, ConditionError
from agent_adapter.policy.errors import PolicyDenied, PolicyError
from agent_adapter.policy.explain import Explanation
from agent_adapter.policy.model import (
    BudgetRule,
    GuardRule,
    LimitsRule,
    MemoryRule,
    PolicySpec,
    RateRule,
    SandboxRule,
    ToolRule,
)
from agent_adapter.policy.policy import ENV_VAR, Policy, PolicyChain

PolicySource = str | os.PathLike[str] | Mapping[str, Any] | PolicySpec


def load_policy(
    source: PolicySource,
    *,
    env: str | None = None,
    allow_relax: bool = False,
    guards: Mapping[str, ContentGuard] | None = None,
    stores: Mapping[str, BudgetStore] | None = None,
    team_of: TeamOf | None = None,
) -> Policy:
    """A validated `Policy` from a file path (.yaml, .yml, .toml, .json), a dict or a PolicySpec.

    Args:
        env: The environment; default `AA_ENV` (empty when unset).
        allow_relax: Accept the relaxations the policy lists under `relax`.
        guards: Content guards built in code, named by `guard:` entries.
        stores: Budget stores built in code (Redis, SQL), named by `store:` entries.
        team_of: The caller's team; default `ctx.extras[team_key]`, then principal claims.

    Raises:
        PolicyError: a malformed, unknown or relaxing setting, with the file, line and key.
        MissingDependency: a YAML file without pyyaml.
    """
    if isinstance(source, Mapping | PolicySpec):
        spec, label, base = validate(source), "", None
    elif isinstance(source, str | os.PathLike):
        path = Path(source)
        data, lines = read_file(path)
        spec, label, base = validate(data, str(path), lines), str(path), path.resolve().parent
    else:
        raise PolicyError("a policy is a file path, a mapping or a PolicySpec")
    return Policy(
        spec,
        env=env,
        source=label,
        allow_relax=allow_relax,
        guards=guards,
        stores=stores,
        team_of=team_of,
        base_dir=base,
    )


__all__ = [
    "ENV_VAR",
    "BudgetRule",
    "Condition",
    "ConditionError",
    "Explanation",
    "GuardRule",
    "LimitsRule",
    "MemoryRule",
    "Policy",
    "PolicyChain",
    "PolicyDenied",
    "PolicyError",
    "PolicySource",
    "PolicySpec",
    "RateRule",
    "SandboxRule",
    "TeamOf",
    "ToolRule",
    "load_policy",
    "team_from",
]
