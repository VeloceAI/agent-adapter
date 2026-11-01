"""BudgetHook: stop (or warn) before a tool or model call once spend reaches a limit."""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Literal, get_args

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded, ConfigError
from agent_adapter.hooks.base import Hook, LLMInvocation, ToolInvocation
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

OnExceed = Literal["warn", "stop"]
BudgetScope = Literal["run", "thread", "global"]

# Runs remembered as already warned; bounded so a long-lived hook does not grow forever.
MAX_WARNED_RUNS = 10_000


class BudgetHook(Hook):
    """Reads the ledger total for `scope` before each tool and LLM call."""

    name = "budget"

    def __init__(
        self, max_usd: float, *, on_exceed: OnExceed = "stop", scope: BudgetScope = "run"
    ) -> None:
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
        self.max_usd = max_usd
        self.on_exceed: OnExceed = on_exceed
        self.scope: BudgetScope = scope
        self._lock = threading.Lock()
        self._warned: OrderedDict[str, None] = OrderedDict()
        self._max_warned_runs = MAX_WARNED_RUNS
        self._no_ledger_warned = False

    async def before_tool(self, inv: ToolInvocation) -> None:
        self._check(inv.ctx)

    async def before_llm(self, inv: LLMInvocation) -> None:
        self._check(inv.ctx)

    def _check(self, ctx: ToolContext) -> None:
        """Compare the scope's spend with the limit.

        Raises:
            BudgetExceeded: spend is at or over the limit and `on_exceed="stop"`.
        """
        ledger = ctx.ledger
        if ledger is None:
            self._warn_no_ledger()
            return
        scope = self.scope
        if scope == "thread" and not ctx.thread_id:
            scope = "run"
        if scope == "run":
            spent = ledger.total_usd(ctx.run_id)
        elif scope == "thread":
            spent = ledger.total_usd(thread_id=ctx.thread_id)
        else:
            spent = ledger.total_usd()
        if spent < self.max_usd:
            return
        if self.on_exceed == "stop":
            logger.warning(
                "%s budget $%.4f reached (spent $%.4f, run %s): stopping",
                scope,
                self.max_usd,
                spent,
                ctx.run_id,
            )
            raise BudgetExceeded(self.max_usd, spent, scope=scope, run_id=ctx.run_id)
        if self._first_warning(ctx.run_id):
            logger.warning(
                "%s budget $%.4f reached (spent $%.4f, run %s): continuing",
                scope,
                self.max_usd,
                spent,
                ctx.run_id,
            )
            event = make_event(
                EventType.BUDGET_WARNING, ctx, limit_usd=self.max_usd, spent_usd=spent, scope=scope
            )
            emit(ctx.sink, event)

    def config(self) -> dict[str, Any]:
        return {"max_usd": self.max_usd, "on_exceed": self.on_exceed, "scope": self.scope}

    @classmethod
    def config_schema(cls) -> dict[str, Any] | None:
        return {
            "type": "object",
            "properties": {
                "max_usd": {"type": "number", "minimum": 0},
                "on_exceed": {"enum": list(get_args(OnExceed))},
                "scope": {"enum": list(get_args(BudgetScope))},
            },
            "required": ["max_usd"],
            "additionalProperties": False,
        }

    def with_config(self, config: Mapping[str, Any]) -> BudgetHook:
        """A new hook with `config` merged over this one's.

        Raises:
            ConfigError: unknown keys or invalid values.
        """
        unknown = sorted(set(config) - set(self.config()))
        if unknown:
            raise ConfigError(f"unknown BudgetHook config keys: {unknown}")
        merged = {**self.config(), **config}
        return BudgetHook(merged["max_usd"], on_exceed=merged["on_exceed"], scope=merged["scope"])

    def _first_warning(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._warned:
                return False
            self._warned[run_id] = None
            while len(self._warned) > self._max_warned_runs:
                self._warned.popitem(last=False)
            return True

    def _warn_no_ledger(self) -> None:
        with self._lock:
            if self._no_ledger_warned:
                return
            self._no_ledger_warned = True
        logger.warning("BudgetHook has no ledger in the tool context; budget not enforced")
