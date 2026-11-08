"""Regression tests from the budget review: holds end with the call, charges match the ledger."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import warnings
from typing import Any

import pytest

from agent_adapter.core.errors import ApprovalPending, BudgetExceeded, ConfigError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message, StreamDone, TextDelta
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.executor import Executor
from agent_adapter.hooks import Hook, ShortCircuit
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets import MemoryBudgetStore, SQLAlchemyBudgetStore, _holds
from agent_adapter.models.callable import CallableModel
from agent_adapter.models.fallback import FallbackModel
from tests.hooks.builtin.budgets.helpers import (
    COST,
    DAY,
    PRICES,
    FrozenClock,
    Who,
    answer,
    ask,
    bill,
    ctx,
)

ASK = LLMRequest(messages=(Message(role="user", content="hi"),))
STREAM = ASK.model_copy(update={"stream": True})


async def _unused(request: LLMRequest) -> LLMResponse:
    raise AssertionError("the stream is used")


def reply() -> LLMResponse:
    return answer(model="house")


def setup(*hooks: Hook, port: Any = None) -> tuple[Executor, Any, Ledger]:
    async def house(request: LLMRequest) -> LLMResponse:
        return reply()

    ledger = Ledger(prices=PRICES)
    executor = Executor(ledger=ledger, hooks=list(hooks))
    model = executor.model(port or CallableModel(house, provider="host", model="house"))
    return executor, model, ledger


def user_hook(**kw: Any) -> BudgetHook:
    kw.setdefault("estimate_usd", 0.3)
    return BudgetHook(1.0, scope="user", **kw)


async def held(hook: BudgetHook, executor: Executor) -> float:
    return (await hook.status(executor.context(user_id="ana"))).reserved_usd


# --- H1: no lock or I/O runs from a finalizer --------------------------------------------


def test_a_gc_inside_keep_cannot_deadlock(monkeypatch):
    holds = _holds.Holds(max_open=10)
    old = ask(ctx(user="ana"))
    holds.keep(old, "k", "r0", None)
    cycle: list[Any] = [ask(ctx(user="bo"))]
    cycle.append(cycle)  # an abandoned invocation in a reference cycle
    holds.keep(cycle[0], "k", "r1", None)
    del cycle
    real = _holds.Hold

    def collecting(*args: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()  # what a GC triggered by this allocation would do
        return real(*args)

    monkeypatch.setattr(_holds, "Hold", collecting)
    done = threading.Event()

    def keep() -> None:
        holds.keep(ask(ctx(user="cy")), "k", "r2", None)
        done.set()

    worker = threading.Thread(target=keep, daemon=True)
    worker.start()
    assert done.wait(5), "keep() deadlocked under a GC"
    assert len(holds) == 3  # a dropped call keeps its hold until on_llm_end or hold_s


def test_holds_register_no_weakref_finalizers():
    assert not hasattr(_holds.Holds, "_abandoned")


# --- M1: every end of a call releases its hold, without a GC ------------------------------


async def test_a_later_hook_refusing_releases_without_gc():
    user = user_hook()
    team = BudgetHook(0.0, scope="team", scope_key=lambda c: "desk")
    executor, model, _ = setup(user, team)
    gc.disable()
    try:
        with pytest.raises(BudgetExceeded):
            await model.bind(executor.context(user_id="ana")).generate(ASK)
        assert await held(user, executor) == 0.0
    finally:
        gc.enable()


async def test_a_cancelled_call_releases_its_hold():
    started = asyncio.Event()

    async def slow(request: LLMRequest) -> LLMResponse:
        started.set()
        await asyncio.sleep(30)
        return reply()

    user = user_hook()
    executor, model, _ = setup(user, port=CallableModel(slow, provider="host", model="house"))
    task = asyncio.create_task(model.bind(executor.context(user_id="ana")).generate(ASK))
    await started.wait()
    assert await held(user, executor) == pytest.approx(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await held(user, executor) == 0.0


async def test_approval_pending_from_the_port_releases_its_hold():
    async def waits(request: LLMRequest) -> LLMResponse:
        raise ApprovalPending("apr_1")

    user = user_hook()
    executor, model, _ = setup(user, port=CallableModel(waits, provider="host", model="house"))
    with pytest.raises(ApprovalPending):
        await model.bind(executor.context(user_id="ana")).generate(ASK)
    assert await held(user, executor) == 0.0


async def test_a_consumer_that_breaks_out_of_a_stream_releases_its_hold():
    async def streams(request: LLMRequest):
        yield TextDelta(text="o")
        await asyncio.sleep(30)
        yield StreamDone(response=reply())

    user = user_hook()
    port = CallableModel(_unused, stream_fn=streams, provider="host", model="house")
    executor, model, _ = setup(user, port=port)
    events = model.bind(executor.context(user_id="ana")).stream(
        ASK.model_copy(update={"stream": True})
    )
    async for _ in events:
        break
    await events.aclose()
    assert await held(user, executor) == 0.0


async def test_a_completed_stream_settles_the_real_cost():
    async def streams(request: LLMRequest):
        yield TextDelta(text="o")
        yield StreamDone(response=reply())

    user = user_hook()
    port = CallableModel(_unused, stream_fn=streams, provider="host", model="house")
    executor, model, ledger = setup(user, port=port)
    async for _ in model.bind(executor.context(user_id="ana")).stream(
        ASK.model_copy(update={"stream": True})
    ):
        pass
    status = await user.status(executor.context(user_id="ana"))
    assert (status.spent_usd, status.reserved_usd) == (pytest.approx(ledger.total_usd()), 0.0)


# --- M2 / M3: the budget charges what the ledger bills ------------------------------------


async def test_nested_instrumented_models_are_charged_once():
    async def house(request: LLMRequest) -> LLMResponse:
        return reply()

    user = user_hook()
    ledger = Ledger(prices=PRICES)
    executor = Executor(ledger=ledger, hooks=[user])
    inner = executor.model(CallableModel(house, provider="host", model="house"))
    outer = executor.model(FallbackModel([inner]))
    await outer.bind(executor.context(user_id="ana")).generate(ASK)
    status = await user.status(executor.context(user_id="ana"))
    assert ledger.total_usd() == pytest.approx(COST)
    assert (status.spent_usd, status.reserved_usd) == (pytest.approx(COST), 0.0)


class Canned(Hook):
    async def before_llm(self, inv: Any) -> ShortCircuit:
        return ShortCircuit(reply())


class Recovers(Hook):
    async def on_llm_error(self, inv: Any, error: BaseException) -> ShortCircuit:
        return ShortCircuit(reply())


async def test_a_short_circuit_reply_after_the_budget_costs_nothing():
    user = user_hook()
    executor, model, ledger = setup(user, Canned())
    await model.bind(executor.context(user_id="ana")).generate(ASK)
    status = await user.status(executor.context(user_id="ana"))
    assert ledger.total_usd() == 0.0
    assert (status.spent_usd, status.reserved_usd) == (0.0, 0.0)


@pytest.mark.parametrize("order", ["recover_first", "budget_first"])
async def test_a_recovered_failure_costs_nothing(order):
    async def broken(request: LLMRequest) -> LLMResponse:
        raise RuntimeError("down")

    user = user_hook()
    hooks = [Recovers(), user] if order == "recover_first" else [user, Recovers()]
    executor, model, ledger = setup(
        *hooks, port=CallableModel(broken, provider="host", model="house")
    )
    await model.bind(executor.context(user_id="ana")).generate(ASK)
    status = await user.status(executor.context(user_id="ana"))
    assert (ledger.total_usd(), status.spent_usd, status.reserved_usd) == (0.0, 0.0, 0.0)


# --- L5, L9, I1 ---------------------------------------------------------------------------


def test_only_run_thread_and_global_scopes_are_tunable():
    schema = BudgetHook.config_schema()
    assert schema["properties"]["scope"]["enum"] == ["run", "thread", "global"]
    assert BudgetHook(1.0).with_config({"scope": "thread"}).scope == "thread"
    for hook, scope in ((BudgetHook(1.0), "team"), (BudgetHook(1.0, scope="user"), "run")):
        with pytest.raises(ConfigError, match="tunable"):
            hook.with_config({"scope": scope})


async def test_a_hold_made_before_midnight_settles_on_its_own_day():
    clock = FrozenClock(1_800_057_600.0 - 1)  # 23:59:59 UTC
    hook = BudgetHook(0.5, scope="user", per="day", prices=PRICES, clock=clock)
    inv = ask(ctx(user="ana"))
    await hook.before_llm(inv)
    clock.now += 2  # past midnight
    await hook.on_llm_end(inv, bill(inv))
    today = await hook.status(ctx(user="ana"))
    assert (today.window, today.spent_usd, today.reserved_usd) == ("day:2027-01-16", 0.0, 0.0)
    clock.now -= 2
    yesterday = await hook.status(ctx(user="ana"))
    assert yesterday.spent_usd == pytest.approx(COST)
    clock.now += DAY  # yesterday's counter is not today's


async def test_the_principal_wins_over_a_different_user_id(caplog):
    hook = BudgetHook(1.0, scope="user", prices=PRICES)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            inv = ask(ctx(user="mallory", principal=Who("ana")))
            await hook.before_llm(inv)
            await hook.on_llm_end(inv, bill(inv))
    assert (await hook.status(ctx(principal=Who("ana")))).spent_usd == pytest.approx(2 * COST)
    assert (await hook.status(ctx(user="mallory"))).spent_usd == 0.0
    assert caplog.text.count("differs from the principal") == 1
    assert "mallory" not in caplog.text


# --- L4: Redis caches its scripts ---------------------------------------------------------


async def test_redis_loads_a_script_once_then_calls_it_by_sha():
    from agent_adapter.runtime.loop_local import LoopLocal
    from tests.hooks.builtin.budgets.stores import fake_redis

    client = fake_redis()
    calls: list[str] = []

    class Counting:
        async def evalsha(self, *args: Any) -> Any:
            calls.append("evalsha")
            return await client.evalsha(*args)

        async def eval(self, *args: Any) -> Any:
            calls.append("eval")
            return await client.eval(*args)

    from agent_adapter.hooks.builtin.budgets import RedisBudgetStore

    store = RedisBudgetStore(Counting)
    for _ in range(2):
        await store.usage("k")
    assert calls == ["evalsha", "eval", "evalsha"]
    await LoopLocal.aclose_all()


# --- M4: Postgres under REPEATABLE READ (contract, needs AA_PG_URL) -----------------------


@pytest.mark.contract
async def test_postgres_repeatable_read_engines_never_overspend():
    url = os.environ.get("AA_PG_URL", "")
    if not url:
        pytest.skip("set AA_PG_URL (postgresql+asyncpg://...) to run against Postgres")
    from sqlalchemy.ext.asyncio import create_async_engine

    from agent_adapter.runtime.loop_local import LoopLocal

    def engine() -> Any:
        return create_async_engine(url, isolation_level="REPEATABLE READ")

    stores = [SQLAlchemyBudgetStore(engine, table="aa_budget_rr") for _ in range(2)]
    key = f"rr|{os.getpid()}"

    async def one(i: int) -> bool:
        reserved = await stores[i % 2].reserve(
            key, f"r{i}", 0.1, limit_usd=1.0, hold_s=60, expire_at=None
        )
        return reserved.accepted

    results = await asyncio.gather(*(one(i) for i in range(30)), return_exceptions=True)
    accepted = sum(r is True for r in results)
    assert accepted <= 10
    assert (await stores[0].usage(key)).reserved_usd <= 1.0 + 1e-9
    await LoopLocal.aclose_all()


def test_memory_store_refuses_rather_than_forgets():
    assert MemoryBudgetStore(max_keys=1).max_keys == 1
