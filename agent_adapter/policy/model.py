"""The policy schema: pydantic models that refuse unknown keys and loose values."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_adapter.hooks.builtin import BudgetScope, OnExceed
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards.base import Action
from agent_adapter.policy.condition import Condition, ConditionError

Relaxation = Literal["limits.max_calls", "memory.retention_days"]
Access = Literal["allow", "deny"]
MAX_NAME = 200
BUILTIN_GUARDS = ("pii", "presidio")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _names(value: Any) -> Any:
    """A single string means a list of one."""
    return [value] if isinstance(value, str) else value


def _check_names(values: list[str] | None, what: str) -> list[str] | None:
    if values is None:
        return None
    if not values:
        raise ValueError(f"{what} must not be empty (leave the key out to match all)")
    for value in values:
        if not value or len(value) > MAX_NAME:
            raise ValueError(f"each {what} entry is a non-empty string of at most {MAX_NAME}")
    return values


class Selector(_Strict):
    """Which calls an entry applies to; every given list must match (globs for `tools`)."""

    tools: list[str] = Field(default_factory=lambda: ["*"])
    kinds: list[str] | None = None
    tags: list[str] | None = None
    envs: list[str] | None = None
    teams: list[str] | None = None
    users: list[str] | None = None

    split_names = field_validator(
        "tools", "kinds", "tags", "envs", "teams", "users", mode="before"
    )(_names)

    @field_validator("tools", "kinds", "tags", "envs", "teams", "users")
    @classmethod
    def _non_empty(cls, value: list[str] | None, info: Any) -> list[str] | None:
        return _check_names(value, info.field_name)


class ToolRule(Selector):
    """Allow or deny tools, or ask for approval always (`true`) or on a condition."""

    allow: bool | None = None
    confirm: Literal[True] | Condition | None = None
    reason: str = Field(default="", max_length=MAX_NAME)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("confirm", mode="before")
    @classmethod
    def _confirm(cls, value: Any) -> Any:
        if value is False:
            raise ValueError("confirm: false cannot remove an approval; leave the key out")
        if value is not None and not isinstance(value, bool | str | Condition):
            raise ValueError("confirm is true or a condition string")
        if isinstance(value, str):
            try:
                return Condition(value)
            except ConditionError as exc:
                raise ValueError(str(exc)) from None
        return value

    @model_validator(mode="after")
    def _does_something(self) -> ToolRule:
        if self.allow is None and self.confirm is None:
            raise ValueError("a rule needs allow or confirm")
        return self


class GuardRule(Selector):
    """A content guard on the selected tools' arguments or results (and model calls)."""

    guard: str = Field(min_length=1, max_length=MAX_NAME)
    on: list[Stage] | None = None
    action: Action | None = None
    entities: list[str] | None = None

    split_lists = field_validator("on", "entities", mode="before")(_names)

    @model_validator(mode="after")
    def _model_stages_need_every_tool(self) -> GuardRule:
        narrowed = self.tools != ["*"] or self.kinds is not None or self.tags is not None
        if narrowed and self.on and any(stage.startswith("llm_") for stage in self.on):
            raise ValueError(
                "llm_request and llm_response guard every model call: "
                "drop tools, kinds and tags, or those stages"
            )
        return self


class BudgetRule(_Strict):
    """A `BudgetHook`: a spend limit per scope, optionally per UTC day or month."""

    max_usd: float = Field(ge=0, allow_inf_nan=False)
    name: str = Field(default="budget", min_length=1, max_length=MAX_NAME)
    scope: BudgetScope = "run"
    per: Literal["day", "month"] | None = None
    on_exceed: OnExceed = "stop"
    store: str = Field(default="memory", min_length=1, max_length=1000)
    envs: list[str] | None = None

    split_envs = field_validator("envs", mode="before")(_names)


class RateRule(_Strict):
    """A `RateLimitHook` bucket: at most `calls` per `per` seconds for one tool kind (or llm)."""

    kind: str = Field(min_length=1, max_length=MAX_NAME)
    calls: int = Field(ge=1)
    per: float = Field(gt=0, allow_inf_nan=False)
    envs: list[str] | None = None

    split_envs = field_validator("envs", mode="before")(_names)


class LimitsRule(_Strict):
    """Run limits: at most `max_calls` distinct calls per run."""

    max_calls: int | None = Field(default=None, ge=1)


class MemoryRule(_Strict):
    """How long memories are kept, and whether the `remember` tool may save, with approval."""

    retention_days: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    save: Literal["allow", "confirm", "deny"] = "allow"


class SandboxRule(_Strict):
    """What a code sandbox must be: isolated from the host, and without network access."""

    require_isolated: bool = False
    deny_network: bool = False


class PolicySpec(_Strict):
    """A whole policy, as written in the file."""

    version: Literal[1] = 1
    default: Access | dict[str, Access] = "allow"
    environments: list[str] | None = None
    team_key: str = Field(default="team", min_length=1, max_length=MAX_NAME)
    rules: list[ToolRule] = Field(default_factory=list)
    guards: list[GuardRule] = Field(default_factory=list)
    budgets: list[BudgetRule] = Field(default_factory=list)
    rate_limits: list[RateRule] = Field(default_factory=list)
    limits: LimitsRule | None = None
    memory: MemoryRule | None = None
    sandbox: SandboxRule | None = None
    relax: list[Relaxation] = Field(default_factory=list)

    split_envs = field_validator("environments", mode="before")(_names)

    @field_validator("environments")
    @classmethod
    def _environments(cls, value: list[str] | None) -> list[str] | None:
        checked = _check_names(value, "environments")
        if checked is not None and len(set(checked)) != len(checked):
            raise ValueError("environments must not repeat")
        return checked

    @model_validator(mode="after")
    def _known_envs(self) -> PolicySpec:
        used = [(f"rules[{i}]", r.envs) for i, r in enumerate(self.rules)]
        used += [(f"guards[{i}]", g.envs) for i, g in enumerate(self.guards)]
        used += [(f"budgets[{i}]", b.envs) for i, b in enumerate(self.budgets)]
        used += [(f"rate_limits[{i}]", r.envs) for i, r in enumerate(self.rate_limits)]
        if isinstance(self.default, dict):
            used.append(("default", list(self.default)))
        named = [(where, envs) for where, envs in used if envs]
        if named and self.environments is None:
            raise ValueError(f"{named[0][0]} names environments: list them in `environments`")
        known = set(self.environments or ())
        for where, envs in named:
            unknown = sorted(set(envs or ()) - known)
            if unknown:
                raise ValueError(f"{where} names unknown environments {unknown}")
        return self
