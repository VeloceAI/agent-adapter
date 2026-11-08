"""Turn policy entries into the existing hooks: guards, budgets and rate limits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agent_adapter.core.errors import AdapterError
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets.sqlite_store import SQLiteBudgetStore
from agent_adapter.hooks.builtin.budgets.store import BudgetStore
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards.base import DEFAULT_STAGES, ContentGuard
from agent_adapter.hooks.builtin.guards.pii import PII_KINDS, PIIGuard
from agent_adapter.hooks.builtin.guards.presidio import PresidioGuard
from agent_adapter.hooks.builtin.rate_limit import Rate, RateLimitHook
from agent_adapter.policy._scope import TeamOf
from agent_adapter.policy.errors import PolicyError
from agent_adapter.policy.model import BudgetRule, GuardRule, RateRule

SQLITE = "sqlite:"


def build_guard(
    rule: GuardRule, where: str, custom: Mapping[str, ContentGuard], source: str
) -> ContentGuard:
    """The guard an entry names: `pii`, `presidio`, or one passed in `guards=`.

    Raises:
        PolicyError: an unknown guard, options on a passed guard, or settings it refuses.
    """
    given = rule.on is not None or rule.action is not None or rule.entities is not None
    if rule.guard in custom:
        guard = custom[rule.guard]
        if not isinstance(guard, ContentGuard):
            raise PolicyError("guards= values must be ContentGuard hooks", source=source, key=where)
        if given:
            raise PolicyError(
                f"guard {rule.guard!r} was built in code: set on, action and entities there",
                source=source,
                key=where,
            )
        return guard
    action = rule.action or "block"
    narrowed = rule.tools != ["*"] or rule.kinds is not None or rule.tags is not None
    stages: tuple[Stage, ...] = tuple(rule.on) if rule.on is not None else _stages(action, narrowed)
    try:
        if rule.guard == "pii":
            return PIIGuard(kinds=rule.entities or PII_KINDS, on=stages, action=action)
        if rule.guard == "presidio":
            return PresidioGuard(entities=rule.entities, on=stages, action=action)
    except AdapterError as exc:
        raise PolicyError(str(exc), source=source, key=where) from exc
    known = sorted({"pii", "presidio", *custom})
    raise PolicyError(
        f"unknown guard {rule.guard!r}; use one of {known} or pass it in guards=",
        source=source,
        key=f"{where}.guard",
    )


def _stages(action: str, narrowed: bool) -> tuple[Stage, ...]:
    """The default stages; a guard on some tools only never claims to watch model calls."""
    if action == "approve":
        return ("args",)
    return tuple(s for s in DEFAULT_STAGES if not narrowed or not s.startswith("llm_"))


class Stores:
    """Budget stores by name: `memory`, `sqlite:PATH`, or one passed in `stores=`; one each."""

    def __init__(self, given: Mapping[str, BudgetStore], base: Path | None, source: str) -> None:
        self.given = dict(given)
        self.base = base
        self.source = source
        self._made: dict[str, BudgetStore] = {}

    def get(self, name: str, where: str) -> BudgetStore | None:
        """The store for `name` (None: the hook's own counters in this process).

        Raises:
            PolicyError: an unknown name, or a SQLite file that cannot be opened.
        """
        if name == "memory":
            return None
        if name in self.given:
            store = self.given[name]
            if not isinstance(store, BudgetStore):
                raise PolicyError(
                    "stores= values must be BudgetStores", source=self.source, key=where
                )
            return store
        if not name.startswith(SQLITE):
            raise PolicyError(
                f"unknown budget store {name!r}; use memory, sqlite:PATH or pass it in stores=",
                source=self.source,
                key=where,
            )
        if name not in self._made:
            path = Path(name[len(SQLITE) :])
            if self.base is not None and not path.is_absolute():
                path = self.base / path  # relative to the policy file, not the working folder
            try:
                self._made[name] = SQLiteBudgetStore(path)
            except AdapterError as exc:
                raise PolicyError(str(exc), source=self.source, key=where) from exc
        return self._made[name]


def build_budget(rule: BudgetRule, where: str, stores: Stores, team_of: TeamOf) -> BudgetHook:
    """A `BudgetHook` for one entry; team budgets count by the policy's team.

    Raises:
        PolicyError: a bad store, or settings the hook refuses.
    """
    store = stores.get(rule.store, f"{where}.store")
    try:
        return BudgetHook(
            rule.max_usd,
            on_exceed=rule.on_exceed,
            scope=rule.scope,
            per=rule.per,
            scope_key=team_of if rule.scope == "team" else None,
            store=store,
            budget=rule.name,
        )
    except AdapterError as exc:
        raise PolicyError(str(exc), source=stores.source, key=where) from exc


def build_rates(entries: list[tuple[str, RateRule]], source: str) -> RateLimitHook | None:
    """One `RateLimitHook` for every active entry, or None without any.

    Raises:
        PolicyError: two entries for one kind in this environment.
    """
    rates: dict[str, Rate] = {}
    for where, rule in entries:
        if rule.kind in rates:
            raise PolicyError(
                f"a second rate limit for kind {rule.kind!r} in this environment",
                source=source,
                key=where,
            )
        rates[rule.kind] = Rate(rule.calls, rule.per)
    return RateLimitHook(rates) if rates else None
