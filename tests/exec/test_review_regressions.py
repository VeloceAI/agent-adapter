"""Regression tests for the WP02 review (fix round 1)."""

import asyncio

import pytest

from agent_adapter import (
    ApprovalDecision,
    ApprovalPending,
    ApprovalRequest,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.cache.keys import digest
from agent_adapter.errors import RunLimitExceeded, ToolUsageError
from agent_adapter.exec import Executor, RunLimits, Step
from agent_adapter.hooks import Hook
from tests.exec.helpers import DeferredFake, GraphBubbleUp, check_events, make
from tests.fakes.approval import ScriptedApproval
from tests.fakes.cache import DictCache

YES = ApprovalDecision(approved=True)


async def test_global_cache_keys_carry_the_schema_digest():
    reg = Registry()

    @reg.tool(cache_ttl=60, cache_scope="global")
    async def rates(currency: str) -> float:
        """Exchange rate."""
        return 1.0

    cache = DictCache()
    await Executor(reg, cache=cache).run("rates", {"currency": "EUR"})
    schema = digest(reg.get("rates").json_schema())[:8]
    assert cache.calls[-1] == (
        "set",
        f"aa:tool:v1:global:rates@{schema}:{digest({'currency': 'EUR'})}",
    )


async def test_a_reused_call_id_does_not_bypass_the_guards():
    reg = Registry()
    ran: list[str] = []

    @reg.tool(once_per_run=True, parallel_safe=True)
    async def a(x: int = 0) -> str:
        """Once per run."""
        ran.append("a")
        return "a"

    @reg.tool(parallel_safe=True)
    async def b(x: int = 0) -> str:
        """Plain."""
        ran.append("b")
        return "b"

    ex = Executor(reg, limits=RunLimits(max_calls=2))
    ctx = ex.context(run_id="r")
    assert await ex.run("a", {}, ctx, call_id="c1") == "a"
    assert await ex.run("a", {}, ctx, call_id="c1") == "a"  # exact re-entry: not counted
    assert await ex.run("b", {}, ctx, call_id="c1") == "b"  # same id, other tool: counted
    with pytest.raises(RunLimitExceeded, match="max_calls"):
        await ex.run("b", {"x": 1}, ctx, call_id="c1")  # same id, other args: a new call
    with pytest.raises(RunLimitExceeded, match="once_per_run"):
        await ex.run("a", {"x": 1}, ctx, call_id="c1")
    assert ran == ["a", "a", "b"]


async def test_duplicate_call_ids_in_one_batch_are_rejected():
    reg = Registry()

    @reg.tool
    async def t() -> int:
        """Exclusive."""
        return 1

    ex = Executor(reg)
    calls = [ToolCall(name="t", id="same"), ToolCall(name="t", id="same")]
    with pytest.raises(ToolUsageError, match="twice"):
        await ex.run_many(calls)
    with pytest.raises(ToolUsageError, match="twice"):
        ex.as_completed(calls)
    assert len(await ex.run_many([ToolCall(name="t"), ToolCall(name="t")])) == 2  # "" is fine


async def test_a_fan_out_inside_a_tool_still_respects_per_kind_caps():
    reg = Registry()
    running = peak = 0

    @reg.tool(kind="http", parallel_safe=True)
    async def fetch(n: int) -> int:
        """Capped at one at a time per run."""
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.002)
        running -= 1
        return n

    ex = Executor(reg, limits=RunLimits(per_kind={"http": 1}))

    @reg.tool(kind="orchestrator")
    async def orchestrate(ctx: ToolContext) -> list[int]:
        """Fans out five http calls in its own run."""
        calls = [ToolCall(name="fetch", arguments={"n": i}) for i in range(5)]
        return [r.value for r in await ex.run_many(calls, ctx)]

    value = await asyncio.wait_for(ex.run("orchestrate", ctx=ex.context(run_id="r")), timeout=5)
    assert value == [0, 1, 2, 3, 4] and peak == 1


async def test_plan_step_call_ids_are_stable_so_a_resumed_plan_runs_a_step_once():
    reg = Registry()
    paid: list[int] = []

    @reg.tool(confirm=True)
    async def pay(amount: int) -> int:
        """Needs approval."""
        paid.append(amount)
        return amount

    port = DeferredFake()
    ex = Executor(reg, approval=port, limits=RunLimits(max_calls=1))
    ctx = ex.context(run_id="r")
    steps = [Step(id="pay", tool="pay", args={"amount": 5})]
    first = await ex.run_plan(steps, ctx)
    assert first["pay"].status == "pending" and first["pay"].call_id == "r:pay"
    port.decide("r", "r:pay", YES)
    second = await ex.run_plan(steps, ctx)
    assert second["pay"].ok and paid == [5]
    third = await ex.run_plan(steps, ctx)  # same step again: re-entry, max_calls not hit
    assert third["pay"].status == "pending"  # the fake asks again; no new call counted


async def test_invalid_args_and_denials_do_not_consume_once_per_run():
    reg = Registry()

    @reg.tool(once_per_run=True, confirm=True)
    async def submit(n: int) -> int:
        """Once per run, with approval."""
        return n

    port = ScriptedApproval([ApprovalDecision(approved=False), YES])
    ex = Executor(reg, approval=port, limits=RunLimits(max_calls=1))
    ctx = ex.context(run_id="r")
    assert (await ex.run_call(ToolCall(name="submit", arguments={"n": "x"}), ctx)).error.kind == (
        "tool-args-invalid"
    )
    denied = await ex.run_call(ToolCall(name="submit", arguments={"n": 1}), ctx)
    assert denied.error.kind == "approval-denied"
    assert await ex.run("submit", {"n": 2}, ctx) == 2
    with pytest.raises(RunLimitExceeded):
        await ex.run("submit", {"n": 3}, ctx)


async def test_approval_pending_from_a_nested_call_skips_on_tool_error_hooks():
    reg = Registry()
    offered: list[BaseException] = []

    class Recover(Hook):
        async def on_tool_error(self, inv, error):
            offered.append(error)

    ex = Executor(reg, approval=DeferredFake(), hooks=[Recover()])

    @reg.tool(confirm=True, parallel_safe=True)
    async def inner() -> str:
        """Needs approval."""
        return "inner"

    @reg.tool(parallel_safe=True)
    async def outer(ctx: ToolContext) -> str:
        """Calls inner."""
        return await ex.run("inner", ctx=ctx)

    result = await ex.run_call(ToolCall(name="outer"), ex.context(run_id="r"))
    assert result.status == "pending" and offered == []


async def test_approval_events_use_one_id_across_pending_and_resume():
    reg = Registry()

    @reg.tool(confirm=True)
    async def act() -> str:
        """Needs approval."""
        return "done"

    class Store:
        """Like DeferredApproval: the record id is the request's approval_id."""

        def __init__(self) -> None:
            self.decided = False

        async def request(self, request: ApprovalRequest) -> ApprovalDecision:
            if not self.decided:
                self.decided = True
                raise ApprovalPending(request.approval_id, run_id=request.run_id)
            return YES

    ex, sink, *_ = make(reg, approval=Store())
    ctx = ex.context(run_id="r")
    assert (await ex.run_call(ToolCall(name="act", id="c"), ctx)).status == "pending"
    assert (await ex.run_call(ToolCall(name="act", id="c"), ctx)).ok
    types = check_events(sink)
    assert types.count("approval.requested") == 1 and types.count("approval.decided") == 1
    ids = {e.data["approval_id"] for e in sink.events if "approval_id" in e.data}
    assert len(ids) == 1


async def test_an_outer_timeout_wins_over_errors_raised_while_cancelling():
    async def stubborn():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise ValueError("raised while being cancelled") from None

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await Executor().parallel(stubborn, stubborn)


async def test_control_flow_inside_a_tools_exception_group_is_unwrapped():
    reg = Registry()

    @reg.tool
    async def grouped() -> None:
        """Its own TaskGroup raises an interrupt."""

        async def interrupt() -> None:
            raise GraphBubbleUp("pause")

        async with asyncio.TaskGroup() as group:
            group.create_task(interrupt())

    with pytest.raises(GraphBubbleUp):
        await Executor(reg).run_call(ToolCall(name="grouped"))
    with pytest.raises(GraphBubbleUp):
        await Executor(reg).run_many([ToolCall(name="grouped")])


async def test_registry_and_hooks_are_read_only():
    ex = Executor(Registry())
    with pytest.raises(AttributeError):
        ex.registry = Registry()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ex.hooks = ()  # type: ignore[misc]
