"""Validation of `BudgetHook` settings: refuse a bad configuration before any call."""

from __future__ import annotations

import math
from typing import Any, Literal, get_args

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.budgets._keys import PERIODS
from agent_adapter.hooks.builtin.budgets.store import BudgetStore

OnExceed = Literal["warn", "stop"]
BudgetScope = Literal["run", "thread", "global", "principal", "user", "team"]


def check_settings(
    max_usd: Any,
    on_exceed: Any,
    scope: Any,
    per: Any,
    scope_key: Any,
    store: Any,
    budget: Any,
    estimate: Any,
) -> None:
    """Refuse a bad configuration up front.

    Raises:
        ConfigError: the first invalid setting.
    """
    if (
        isinstance(max_usd, bool)
        or not isinstance(max_usd, int | float)
        or not math.isfinite(max_usd)
        or max_usd < 0
    ):
        raise ConfigError(f"max_usd must be a finite number >= 0, got {max_usd!r}")
    if on_exceed not in get_args(OnExceed):
        raise ConfigError(f"on_exceed must be 'warn' or 'stop', got {on_exceed!r}")
    if scope not in get_args(BudgetScope):
        raise ConfigError(f"scope must be one of {list(get_args(BudgetScope))}, got {scope!r}")
    if scope == "team" and not callable(scope_key):
        raise ConfigError("scope='team' needs scope_key=lambda ctx: <team id>")
    if scope != "team" and scope_key is not None:
        raise ConfigError(f"scope_key only applies to scope='team', not {scope!r}")
    if per is not None and per not in PERIODS:
        raise ConfigError(f"per must be None, 'day' or 'month', got {per!r}")
    if store is not None and not isinstance(store, BudgetStore):
        raise ConfigError(f"store must be a BudgetStore, got {type(store).__name__}")
    if not isinstance(budget, str) or not budget:
        raise ConfigError(f"budget must be a non-empty str, got {budget!r}")
    if estimate is not None and not callable(estimate):
        if isinstance(estimate, bool) or not isinstance(estimate, int | float):
            raise ConfigError(f"estimate_usd must be a number or a callable, got {estimate!r}")
        if not math.isfinite(estimate) or estimate < 0:
            raise ConfigError(f"estimate_usd must be a finite number >= 0, got {estimate!r}")
