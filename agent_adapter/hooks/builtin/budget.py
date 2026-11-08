"""BudgetHook: stop (or warn) before a tool or model call once spend reaches a limit."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, get_args

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded, ConfigError, log_failure
from agent_adapter.core.types import LedgerEntry
from agent_adapter.hooks.base import Hook, LLMInvocation, ToolInvocation
from agent_adapter.hooks.builtin.budgets._holds import Holds
from agent_adapter.hooks.builtin.budgets._keys import (
    KEEP_AFTER_S,
    PERIODS,
    Per,
    Window,
    counter_key,
    principal_of,
    user_differs,
    window_at,
)
from agent_adapter.hooks.builtin.budgets._pricing import (
    PriceTable,
    billed_usd,
    estimate_usd,
    prices_of,
)
from agent_adapter.hooks.builtin.budgets._settings import BudgetScope as BudgetScope
from agent_adapter.hooks.builtin.budgets._settings import OnExceed as OnExceed
from agent_adapter.hooks.builtin.budgets._settings import check_settings
from agent_adapter.hooks.builtin.budgets.status import BudgetStatus
from agent_adapter.hooks.builtin.budgets.store import (
    BudgetStore,
    BudgetUsage,
    MemoryBudgetStore,
    admits,
)
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

ScopeKey = Callable[[ToolContext], str]
Estimate = float | Callable[[LLMInvocation], float]

# Runs remembered as already warned; bounded so a long-lived hook does not grow forever.
MAX_WARNED_RUNS = 10_000
MAX_HELD = 10_000  # model calls holding a reservation at once; older ones free by `hold_s`
TUNABLE_SCOPES = ("run", "thread", "global")  # user and team budgets keep their scope
DEFAULT_HOLD_S = 900.0
_COUNTED = frozenset({"principal", "user", "team"})


class BudgetHook(Hook):
    """Caps spend per run, thread, user, team or everything, optionally per UTC day or month.

    Raises:
        ConfigError: invalid limits, scope, window, store or estimate; `scope="team"`
            without `scope_key`, or `scope_key` with another scope.
    """

    name = "budget"

    def __init__(
        self,
        max_usd: float,
        *,
        on_exceed: OnExceed = "stop",
        scope: BudgetScope = "run",
        per: Per | None = None,
        scope_key: ScopeKey | None = None,
        store: BudgetStore | None = None,
        budget: str = "budget",
        estimate_usd: Estimate | None = None,
        prices: PriceTable | None = None,
        hold_s: float = DEFAULT_HOLD_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        check_settings(max_usd, on_exceed, scope, per, scope_key, store, budget, estimate_usd)
        if isinstance(hold_s, bool) or not isinstance(hold_s, int | float) or not hold_s > 0:
            raise ConfigError(f"hold_s must be a number > 0, got {hold_s!r}")
        self.max_usd = max_usd
        self.on_exceed: OnExceed = on_exceed
        self.scope: BudgetScope = scope
        self.per: Per | None = per
        self.scope_key = scope_key
        self.budget = budget
        self.estimate_usd = estimate_usd
        self.prices = prices
        self.hold_s = float(hold_s)
        self._clock = clock
        self._given_store = store
        self.counted = store is not None or per is not None or scope in _COUNTED
        self.store: BudgetStore | None = (
            store if store is not None or not self.counted else MemoryBudgetStore(clock=clock)
        )
        self._lock = threading.Lock()
        self._warned: OrderedDict[str, None] = OrderedDict()
        self._max_warned_runs = MAX_WARNED_RUNS
        self._holds = Holds(max_open=MAX_HELD)
        self._no_ledger_warned = False
        self._unpriced_warned = False
        self._identity_warned = False

    # --- hook phases ------------------------------------------------------------------

    async def before_tool(self, inv: ToolInvocation) -> None:
        if self.store is None:
            self._check(inv.ctx)
            return
        scope, scope_id = self.scope_of(inv.ctx)
        window = window_at(self.per, self._clock())
        key = counter_key(self.budget, _key_scope(scope), scope_id, window)
        usage = await self.store.usage(key)
        self._judge(inv.ctx, scope, window, usage, 0.0)

    async def before_llm(self, inv: LLMInvocation) -> None:
        if self.store is None:
            self._check(inv.ctx)
            return
        now = self._clock()
        scope, scope_id = self.scope_of(inv.ctx)
        window = window_at(self.per, now)
        key = counter_key(self.budget, _key_scope(scope), scope_id, window)
        amount = self._estimate(inv, now)
        rid = new_id("bud_")
        expire_at = None if window.end is None else window.end + KEEP_AFTER_S
        limit = self.max_usd if self.on_exceed == "stop" else None
        reserved = await self.store.reserve(
            key, rid, amount, limit_usd=limit, hold_s=self.hold_s, expire_at=expire_at
        )
        if not reserved.accepted:
            self._stop(inv.ctx, scope, window, reserved.usage.committed_usd)
        self._holds.keep(inv, key, rid, expire_at)
        self._judge(inv.ctx, scope, window, reserved.usage, amount)  # warns; a hold fits
        event = make_event(
            EventType.BUDGET_RESERVED,
            inv.ctx,
            scope=scope,
            limit_usd=self.max_usd,
            spent_usd=reserved.usage.committed_usd,
            reserved_usd=amount,
            window=window.label,
        )
        emit(inv.ctx.sink, event)

    async def on_llm_end(self, inv: LLMInvocation, billed: LedgerEntry | None) -> None:
        """Settle the hold with what the call billed (as the ledger prices it), or release it."""
        hold = self._holds.take(inv)
        store = self.store
        if hold is None or store is None:
            return
        try:
            if billed is None:  # nothing billed here: refused, or a nested call holds the cost
                await store.release(hold.key, hold.reservation_id)
                return
            cost = billed_usd(prices_of(self.prices, inv.ctx.ledger), billed)
            if cost is None:
                self._warn_unpriced(billed.model)
                cost = 0.0  # as the ledger: an unknown price counts as 0
            await store.settle(hold.key, hold.reservation_id, cost, expire_at=hold.expire_at)
        except Exception as exc:
            # The call has ended; its hold stays counted until `hold_s` passes.
            log_failure(logger, logging.ERROR, exc, "budget settle failed (run %s)", inv.ctx.run_id)

    # --- scopes and status -------------------------------------------------------------

    def scope_of(self, ctx: ToolContext) -> tuple[str, str]:
        """The effective scope and the id spend is counted under for `ctx`.

        Raises:
            ConfigError: `scope_key` raised or returned something other than a non-empty str.
        """
        if self.scope == "thread":
            return ("thread", ctx.thread_id) if ctx.thread_id else ("run", ctx.run_id)
        if self.scope == "run":
            return "run", ctx.run_id
        if self.scope == "global":
            return "global", "*"
        if self.scope == "team":
            return "team", self._team(ctx)
        if user_differs(ctx):
            self._warn_identity()
        return self.scope, principal_of(ctx)

    async def status(self, ctx: ToolContext, *, scope_id: str | None = None) -> BudgetStatus:
        """The budget of `ctx`'s scope, or of `scope_id` when given (an approver's view).

        Raises:
            ConfigError: as `scope_of`.
        """
        if scope_id is None:
            scope, found = self.scope_of(ctx)
        else:
            scope, found = self.scope, scope_id
        window = window_at(self.per, self._clock())
        if self.store is not None:
            key = counter_key(self.budget, _key_scope(scope), found, window)
            usage = await self.store.usage(key)
        else:
            usage = BudgetUsage(_ledger_spent(ctx, scope, found))
        return BudgetStatus(
            name=self.budget,
            scope=scope,
            scope_id=found,
            window=window.label,
            limit_usd=self.max_usd,
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd,
            resets_at=window.end,
            on_exceed=self.on_exceed,
        )

    # --- tuning ------------------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        found: dict[str, Any] = {"max_usd": self.max_usd, "on_exceed": self.on_exceed}
        if self.scope in TUNABLE_SCOPES:  # a user or team budget's scope is not tunable
            found["scope"] = self.scope
        if self.per is not None:  # absent by default, so 1.0 configs read back unchanged
            found["per"] = self.per
        return found

    @classmethod
    def config_schema(cls) -> dict[str, Any] | None:
        return {
            "type": "object",
            "properties": {
                "max_usd": {"type": "number", "minimum": 0},
                "on_exceed": {"enum": list(get_args(OnExceed))},
                "scope": {"enum": list(TUNABLE_SCOPES)},
                "per": {"enum": [None, *PERIODS]},
            },
            "required": ["max_usd"],
            "additionalProperties": False,
        }

    def with_config(self, config: Mapping[str, Any]) -> BudgetHook:
        """A new hook with `config` merged over this one's (store and keys carried over).

        Raises:
            ConfigError: unknown keys or invalid values.
        """
        unknown = sorted(set(config) - {"max_usd", "on_exceed", "scope", "per"})
        if unknown:
            raise ConfigError(f"unknown BudgetHook config keys: {unknown}")
        if "scope" in config and (
            self.scope not in TUNABLE_SCOPES or config["scope"] not in TUNABLE_SCOPES
        ):
            raise ConfigError(f"scope is tunable only among {list(TUNABLE_SCOPES)}")
        merged = {"per": None, "scope": self.scope, **self.config(), **config}
        return BudgetHook(
            merged["max_usd"],
            on_exceed=merged["on_exceed"],
            scope=merged["scope"],
            per=merged["per"],
            scope_key=self.scope_key,
            store=self._carried_store(merged["per"], merged["scope"]),
            budget=self.budget,
            estimate_usd=self.estimate_usd,
            prices=self.prices,
            hold_s=self.hold_s,
            clock=self._clock,
        )

    # --- internals ---------------------------------------------------------------------

    def _carried_store(self, per: Any, scope: Any) -> BudgetStore | None:
        """The given store; else the default one while the new config still counts."""
        if self._given_store is not None:
            return self._given_store
        return self.store if per is not None or scope in _COUNTED else None

    def _check(self, ctx: ToolContext) -> None:
        """Compare the ledger's spend for the scope with the limit.

        Raises:
            BudgetExceeded: spend is at or over the limit and `on_exceed="stop"`.
        """
        if ctx.ledger is None:
            self._warn_no_ledger()
            return
        scope, scope_id = self.scope_of(ctx)
        spent = _ledger_spent(ctx, scope, scope_id)
        self._judge(ctx, scope, window_at(None, 0.0), BudgetUsage(spent), 0.0)

    def _judge(
        self, ctx: ToolContext, scope: str, window: Window, usage: BudgetUsage, amount: float
    ) -> None:
        """Stop or warn when `amount` does not fit.

        Raises:
            BudgetExceeded: it does not fit and `on_exceed="stop"`.
        """
        if admits(usage, amount, self.max_usd):
            return
        spent = usage.committed_usd
        if self.on_exceed == "stop":
            self._stop(ctx, scope, window, spent)
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

    def _stop(self, ctx: ToolContext, scope: str, window: Window, spent: float) -> None:
        """Log, emit `budget.exceeded` and refuse the call.

        Raises:
            BudgetExceeded: always.
        """
        logger.warning(
            "%s budget $%.4f reached (spent $%.4f, run %s): stopping",
            scope,
            self.max_usd,
            spent,
            ctx.run_id,
        )
        event = make_event(
            EventType.BUDGET_EXCEEDED,
            ctx,
            scope=scope,
            limit_usd=self.max_usd,
            spent_usd=spent,
            window=window.label,
        )
        emit(ctx.sink, event)
        raise BudgetExceeded(self.max_usd, spent, scope=scope, run_id=ctx.run_id)

    def _estimate(self, inv: LLMInvocation, now: float) -> float:
        given = self.estimate_usd
        if given is None:
            prices = prices_of(self.prices, inv.ctx.ledger)
            return estimate_usd(prices, inv.model, inv.request, now)
        value = given(inv) if callable(given) else given
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"estimate_usd returned {type(value).__name__}, not a number")
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"estimate_usd must be a finite number >= 0, got {value!r}")
        return float(value)

    def _team(self, ctx: ToolContext) -> str:
        assert self.scope_key is not None  # checked in __init__
        try:
            found = self.scope_key(ctx)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "budget scope_key failed (run %s)", ctx.run_id)
            raise ConfigError("budget scope_key failed; the call is refused") from exc
        if not isinstance(found, str) or not found:
            raise ConfigError("budget scope_key must return a non-empty str; the call is refused")
        return found

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

    def _warn_identity(self) -> None:
        with self._lock:
            if self._identity_warned:
                return
            self._identity_warned = True
        logger.warning("budget: ctx.user_id differs from the principal; charging the principal")

    def _warn_unpriced(self, model: str) -> None:
        with self._lock:
            if self._unpriced_warned:
                return
            self._unpriced_warned = True
        logger.warning("budget has no price for model %s: its calls count as $0", model)


def _key_scope(scope: str) -> str:
    """`user` and `principal` name one identity, so they share counters."""
    return "principal" if scope == "user" else scope


def _ledger_spent(ctx: ToolContext, scope: str, scope_id: str) -> float:
    ledger = ctx.ledger
    if ledger is None:
        return 0.0
    if scope == "run":
        return ledger.total_usd(scope_id)
    if scope == "thread":
        return ledger.total_usd(thread_id=scope_id)
    return ledger.total_usd()
