import asyncio
import threading

import pytest

from agent_adapter import Registry, ToolCall, ToolContext, run_sync
from agent_adapter.errors import RunLimitExceeded, ToolUsageError
from agent_adapter.exec import Backoff, Executor, RunLimits, Step
from tests.exec.helpers import check_events, make
from tests.fakes.approval import ScriptedApproval
from tests.fakes.cache import DictCache
from tests.fakes.hooks import RecordingHook
from tests.fakes.sinks import RecordingLedger, RecordingSink

registry = Registry()


@registry.tool(once_per_run=True, parallel_safe=True)
async def once(x: int) -> int:
    """Only once per run."""
    await asyncio.sleep(0.001)
    return x


def test_context_carries_the_executor_ports():
    sink, ledger, cache = RecordingSink(), RecordingLedger(), DictCache()
    ex = Executor(sink=sink, ledger=ledger, cache=cache)
    ctx = ex.context(thread_id="t", state={"user": "u"}, extras={"k": 1})
    assert ctx.run_id.startswith("run_") and ctx.call_id == "" and ctx.thread_id == "t"
    assert (ctx.sink, ctx.ledger, ctx.cache) == (ex.sink, ledger, cache)
    assert ex.sink.sinks == (sink,)  # the executor's hub, so sinks added later reach ctx
    assert ctx.state == {"user": "u"} and ctx.extras == {"k": 1}
    assert ex.context(run_id="mine").run_id == "mine"
    assert ex.context(user_id="ann").user_id == "ann" and ctx.user_id == ""


def test_public_configuration_is_read_only():
    limits, backoff = RunLimits(max_calls=3), Backoff(base_s=1)
    port = ScriptedApproval([])
    ex = Executor(registry, approval=port, limits=limits, backoff=backoff, namespace="ns")
    assert (ex.registry, ex.approval, ex.limits, ex.backoff, ex.namespace) == (
        registry,
        port,
        limits,
        backoff,
        "ns",
    )
    assert ex.hooks == () and Executor().limits == RunLimits() and Executor().backoff == Backoff()
    with pytest.raises(AttributeError):
        ex.limits = RunLimits()  # type: ignore[misc]


async def test_run_scope_emits_run_events_and_ends_the_run():
    ex, sink, *_ = make(registry)
    async with ex.run_scope(run_id="r") as ctx:
        await ex.run("once", {"x": 1}, ctx)
        assert len(ex._settings.runs) == 1
    assert check_events(sink) == ["run.started", "tool.started", "tool.finished", "run.finished"]
    assert sink.events[-1].data == {"ok": True} and len(ex._settings.runs) == 0
    async with ex.run_scope(run_id="r") as ctx:  # ended, so once_per_run starts over
        await ex.run("once", {"x": 1}, ctx)


async def test_run_scope_reports_failure():
    ex, sink, *_ = make(registry)
    with pytest.raises(KeyError):
        async with ex.run_scope():
            raise KeyError("boom")
    assert sink.events[-1].data == {"ok": False}


async def test_end_run_resets_guards():
    ex = Executor(registry)
    ctx = ex.context(run_id="r")
    await ex.run("once", {"x": 1}, ctx)
    with pytest.raises(RunLimitExceeded):
        await ex.run("once", {"x": 2}, ctx)
    ex.end_run("r")
    assert await ex.run("once", {"x": 3}, ctx) == 3


async def test_registry_keyword_overrides_the_executor_registry():
    other = Registry()

    @other.tool
    async def only_here() -> str:
        """Lives in another registry."""
        return "here"

    ex = Executor(registry)
    assert (await ex.run_call(ToolCall(name="only_here"))).error.kind == "tool-not-found"
    assert (await ex.run_call(ToolCall(name="only_here"), registry=other)).value == "here"
    [result] = await ex.run_many([ToolCall(name="only_here")], registry=other)
    assert result.value == "here"
    assert [
        r.value async for r in ex.as_completed([ToolCall(name="only_here")], registry=other)
    ] == ["here"]
    plan = await ex.run_plan([Step(id="s", tool="only_here")], registry=other)
    assert plan["s"].value == "here"
    assert (await ex.run_plan([Step(id="s", tool="only_here")]))["s"].error.kind == (
        "tool-not-found"
    )


async def test_replace_builds_a_new_executor_and_never_mutates_self():
    hook, sink = RecordingHook(), RecordingSink()
    ex = Executor(registry, hooks=[hook], limits=RunLimits(max_calls=5))
    other = ex.replace(sink=sink, approval=ScriptedApproval([]))
    assert other is not ex and other.registry is registry and other.hooks == (hook,)
    assert other.limits == ex.limits and other.sink.sinks == (sink,) and ex.sink.sinks == ()
    assert ex.approval is None
    with pytest.raises(ToolUsageError, match="unknown arguments"):
        ex.replace(colour="red")


async def test_replaced_executors_share_run_state():
    ex = Executor(registry)
    other = ex.replace(sink=RecordingSink())
    ctx = ex.context(run_id="shared")
    await ex.run("once", {"x": 1}, ctx)
    with pytest.raises(RunLimitExceeded):
        await other.run("once", {"x": 2}, ctx)
    fresh = ex.replace(max_tracked_runs=5)  # a different table size means a new table
    assert await fresh.run("once", {"x": 3}, ctx) == 3


async def test_nested_call_inside_a_non_parallel_safe_tool_does_not_deadlock():
    reg = Registry()
    ex = Executor(reg)

    @reg.tool
    async def parent(ctx: ToolContext) -> list[str]:
        """Exclusive; fans out to exclusive children of the same run."""
        results = await ex.run_many(
            [ToolCall(name="child", arguments={"n": i}) for i in range(3)], ctx
        )
        return [r.value for r in results]

    @reg.tool
    async def child(n: int, ctx: ToolContext) -> str:
        """Exclusive leaf."""
        return f"{n}:{ctx.run_id}"

    async with ex.run_scope(run_id="r") as ctx:
        value = await asyncio.wait_for(ex.run("parent", ctx=ctx), timeout=2)
    assert value == ["0:r", "1:r", "2:r"]


def test_two_concurrent_run_sync_calls_share_one_executor():
    reg = Registry()
    entered = threading.Barrier(2, timeout=5)

    @reg.tool
    async def exclusive(n: int) -> int:
        """Not parallel-safe: its run lock lives on each thread's own loop."""
        await asyncio.sleep(0.01)
        return n

    ex = Executor(reg, limits=RunLimits(max_calls=3))
    results: dict[int, list[str]] = {}

    def worker(index: int) -> None:
        entered.wait()
        ctx = ToolContext(run_id="shared-run")
        calls = [
            ToolCall(name="exclusive", arguments={"n": n}, id=f"t{index}-{n}") for n in range(3)
        ]
        results[index] = [r.status for r in run_sync(ex.run_many(calls, ctx), timeout=10)]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    statuses = results[0] + results[1]
    assert (
        statuses.count("ok") == 3 and statuses.count("error") == 3
    )  # max_calls counted across loops


async def test_an_explicit_empty_registry_is_not_replaced_by_the_default():
    ex = Executor(registry)
    result = await ex.run_call(ToolCall(name="once", arguments={"x": 1}), registry=Registry())
    assert result.error.kind == "tool-not-found"


async def test_as_completed_with_a_caller_run_keeps_the_run():
    ex = Executor(registry)
    ctx = ex.context(run_id="kept")
    assert [
        r.value async for r in ex.as_completed([ToolCall(name="once", arguments={"x": 1})], ctx)
    ] == [1]
    assert len(ex._settings.runs) == 1
