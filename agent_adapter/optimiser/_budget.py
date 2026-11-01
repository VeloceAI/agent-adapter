"""Input checks and the run's budget meter: spend, case runs and wall time against limits."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from agent_adapter.core.errors import ConfigError, OptimiserError
from agent_adapter.core.ports import LedgerPort
from agent_adapter.hooks.base import Hook
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.optimiser.model import (
    EvalCase,
    HarnessTree,
    OptimiserBudget,
    OptimiserConfig,
    SurfaceCaps,
)
from agent_adapter.optimiser.surface import Validates, validate_tree

_now: Callable[[], float] = time.monotonic  # module-level so tests can drive the wall clock


class Stopped(OptimiserError):
    """A budget limit ended the run early (internal: the run still reports)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"budget stop: {reason}")
        self.reason = reason


def check_config(config: OptimiserConfig) -> None:
    """Reject values the loop cannot work with.

    Raises:
        ConfigError: every problem, listed together.
    """
    problems = [
        f"{name} must be >= {low}, got {value}"
        for name, value, low in (
            ("k", config.k, 1),
            ("rollout_concurrency", config.rollout_concurrency, 1),
            ("evidence_chars", config.evidence_chars, 1_000),
            ("obs_trunc", config.obs_trunc, 100),
            ("max_lessons", config.max_lessons, 0),
            ("min_gain", config.min_gain, 0),
        )
        if value < low
    ]
    if config.case_timeout_s is not None and config.case_timeout_s <= 0:
        problems.append(f"case_timeout_s must be > 0, got {config.case_timeout_s}")
    for name, value in config.budget.model_dump().items():
        if value is not None and value < 0:
            problems.append(f"budget.{name} must be >= 0, got {value}")
    if problems:
        raise ConfigError("; ".join(problems))


def check_host(
    host: Validates, tree: HarnessTree, caps: SurfaceCaps, cases: Sequence[EvalCase]
) -> None:
    """Reject a host the loop cannot start with.

    Raises:
        OptimiserError: no cases, duplicate case ids, or an invalid surface.
    """
    if not cases:
        raise OptimiserError("the host has no eval cases")
    if len({case.id for case in cases}) != len(cases):
        raise OptimiserError("the host's eval cases have duplicate ids")
    problems = validate_tree(tree, caps, host)
    if problems:
        raise OptimiserError("the host's own surface is invalid: " + "; ".join(problems))


class Meter:
    """What the run has used (including a resumed run's earlier usage) against its limits."""

    def __init__(
        self,
        budget: OptimiserBudget,
        ledger: LedgerPort,
        *,
        restored_usd: float,
        restored_wall: float,
    ) -> None:
        self.budget = budget
        self.ledger = ledger
        self.restored_usd = restored_usd
        self.restored_wall = restored_wall
        self.started = _now()

    def spent(self) -> float:
        return self.restored_usd + self.ledger.total_usd()

    def wall(self) -> float:
        return self.restored_wall + (_now() - self.started)

    def guards(self) -> list[Hook]:
        """A `BudgetHook` over the remaining USD (global scope of the run's own ledger)."""
        if self.budget.max_usd is None:
            return []
        return [BudgetHook(max(self.budget.max_usd - self.restored_usd, 0.0), scope="global")]

    def wall_left(self) -> float | None:
        """Seconds left under `max_wall_s` (never negative), or None without that limit."""
        if self.budget.max_wall_s is None:
            return None
        return max(self.budget.max_wall_s - self.wall(), 0.0)

    def over_usd(self) -> bool:
        return self.budget.max_usd is not None and self.spent() >= self.budget.max_usd

    def gate(self, *, done_case_runs: int, case_runs: int = 0) -> None:
        """Stop before a stage that a limit no longer allows.

        Raises:
            Stopped: with the name of the limit.
        """
        limit = self.budget.max_case_runs
        if limit is not None and done_case_runs + case_runs > limit:
            raise Stopped("max_case_runs")
        if self.over_usd():
            raise Stopped("max_usd")
        if self.budget.max_wall_s is not None and self.wall() >= self.budget.max_wall_s:
            raise Stopped("max_wall_s")
