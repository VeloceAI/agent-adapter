import asyncio
import logging

import pytest

from agent_adapter import Registry, ToolCall, ToolContext, ToolError, ToolResult
from agent_adapter.errors import (
    ModelRateLimited,
    ToolArgsInvalid,
    ToolFailed,
    ToolNotFound,
    ToolTimeout,
    ToolUsageError,
)
from agent_adapter.exec import Executor
from agent_adapter.hooks import Hook, ShortCircuit
from tests.exec.helpers import GraphBubbleUp, check_events, make
from tests.fakes.hooks import RaisingHook, RecordingHook
from tests.fakes.sinks import RecordingSink

registry = Registry()


@registry.tool(kind="math", parallel_safe=True)
async def add(a: int, b: int = 1) -> int:
    """Add two numbers."""
    return a + b


@registry.tool
def shout(text: str) -> str:
    """Upper-case text (sync: runs in a worker thread)."""
    return text.upper()


async def test_ok_call_returns_value_events_and_ledger_entry():
    ex, sink, ledger, _ = make(registry)
    ctx = ex.context(run_id="run_1", thread_id="th")
    result = await ex.run_call(ToolCall(name="add", arguments={"a": 2}, id="c1"), ctx)
    assert result == ToolResult(call_id="c1", name="add", status="ok", value=3, attempts=1)
    assert check_events(sink) == ["tool.started", "tool.finished"]
    started, finished = sink.events
    assert started.data == {"tool": "add", "kind": "math"}
    assert started.run_id == "run_1" and started.call_id == "c1" and started.parent_id == ""
    assert finished.data["attempts"] == 1 and finished.data["cached"] is False
    [entry] = ledger.entries("run_1")
    assert entry.model_dump(
        include={"kind", "call_id", "entry_id", "stage", "provider", "model"}
    ) == {
        "kind": "tool",
        "call_id": "c1",
        "entry_id": "c1",
        "stage": "add",
        "provider": "tool",
        "model": "add",
    }
    assert entry.cost_usd == 0.0 and entry.ok and entry.attempts == 1 and entry.thread_id == "th"
    assert not entry.cache_hit and entry.error_kind == ""
    assert entry.started_at <= entry.ended_at


async def test_run_returns_the_value_and_sync_tools_work():
    ex, *_ = make(registry)
    assert await ex.run("add", {"a": 1, "b": 5}) == 6
    assert await ex.run("shout", {"text": "hi"}) == "HI"


async def test_latency_comes_from_the_clock():
    reg = Registry()
    ex, sink, _, clock = make(reg)

    @reg.tool
    async def tick() -> int:
        """Advance time."""
        clock.advance(0.25)
        return 1

    result = await ex.run_call(ToolCall(name="tick"))
    assert result.latency_ms == 250.0 and sink.events[-1].data["latency_ms"] == 250.0


async def test_call_id_prefers_the_call_then_the_keyword():
    ex, *_ = make(registry)
    assert (
        await ex.run_call(ToolCall(name="add", arguments={"a": 1}, id="mine"))
    ).call_id == "mine"
    seen = []

    class Spy(Hook):
        async def before_tool(self, inv):
            seen.append(inv.ctx.call_id)

    ex = Executor(registry, hooks=[Spy()])
    await ex.run("add", {"a": 1}, call_id="kw")
    await ex.run("add", {"a": 1})
    assert seen[0] == "kw" and seen[1].startswith("call_")


async def test_unknown_tool_is_a_tool_not_found_result():
    ex, sink, ledger, _ = make(registry)
    result = await ex.run_call(ToolCall(name="nope"))
    assert result.status == "error" and result.error.kind == "tool-not-found"
    assert result.attempts == 0 and isinstance(result.exception, ToolNotFound)
    assert check_events(sink) == ["tool.started", "tool.failed"]
    assert sink.events[0].data == {"tool": "nope", "kind": ""}
    assert [e.error_kind for e in ledger.entries()] == ["tool-not-found"]
    with pytest.raises(ToolNotFound):
        await ex.run("nope")


async def test_executor_without_registry_looks_in_the_default_registry():
    result = await Executor().run_call(ToolCall(name="add_not_in_the_default_registry"))
    assert result.error.kind == "tool-not-found"
    assert "no tool named 'add_not_in_the_default_registry'" in result.error.message


async def test_run_spec_runs_an_unregistered_spec():
    ex = Executor()
    assert await ex.run_spec(registry.get("add"), {"a": 4}) == 5


async def test_invalid_args_skip_hooks_and_the_tool():
    hook = RecordingHook()
    ex, _, ledger, _ = make(registry, hooks=[hook])
    result = await ex.run_call(ToolCall(name="add", arguments={"a": "x"}))
    assert result.error.kind == "tool-args-invalid" and result.attempts == 0
    assert hook.calls == [] and not ledger.entries()[0].ok
    with pytest.raises(ToolArgsInvalid):
        await ex.run("add", {"b": 1})


async def test_hooks_see_validated_json_arguments_in_order():
    hook = RecordingHook()
    ex, *_ = make(registry, hooks=[hook])
    await ex.run("add", {"a": 2})
    assert hook.phases() == ["before_tool", "after_tool"]
    inv = hook.calls[0][1][0]
    assert inv.call.arguments == {"a": 2, "b": 1} and inv.spec.name == "add"
    assert inv.ctx.call_id == inv.call.id


async def test_before_tool_short_circuit_skips_the_tool():
    calls = []
    reg = Registry()

    @reg.tool(cache_ttl=60)
    async def costly(q: str) -> str:
        """Expensive."""
        calls.append(q)
        return q

    hook = RecordingHook({"before_tool": ShortCircuit("canned")})
    ex, sink, ledger, _ = make(reg, hooks=[hook])
    result = await ex.run_call(ToolCall(name="costly", arguments={"q": "x"}))
    assert result.value == "canned" and result.attempts == 0 and calls == []
    assert hook.phases() == ["before_tool", "after_tool"]
    assert ledger.entries()[0].attempts == 0
    assert check_events(sink) == ["tool.started", "tool.finished"]


async def test_after_tool_may_replace_the_result():
    class Doubler(Hook):
        async def after_tool(self, inv, result):
            return ToolResult(result.call_id, result.name, "ok", value=result.value * 2, attempts=1)

    ex = Executor(registry, hooks=[Doubler()])
    assert await ex.run("add", {"a": 1}) == 4


async def test_after_tool_must_return_a_tool_result():
    hook = RecordingHook({"after_tool": "oops"})
    ex = Executor(registry, hooks=[hook])
    result = await ex.run_call(ToolCall(name="add", arguments={"a": 1}))
    assert result.error.kind == "tool-usage-error"


async def test_a_raising_hook_fails_the_call_with_its_exception():
    error = KeyError("hook")
    ex = Executor(registry, hooks=[RaisingHook(error, phases=["before_tool"])])
    result = await ex.run_call(ToolCall(name="add", arguments={"a": 1}))
    assert result.exception is error and result.error.kind == "internal-error"
    with pytest.raises(KeyError):
        await ex.run("add", {"a": 1})


def test_hooks_must_be_hook_instances():
    with pytest.raises(ToolUsageError):
        Executor(hooks=[object()])  # type: ignore[list-item]


def _failing(reg, failures, **policy):
    state = {"n": 0}

    @reg.tool(**policy)
    async def flaky() -> str:
        """Fails a few times."""
        state["n"] += 1
        if failures:
            raise failures.pop(0)
        return "ok"

    return state


async def test_retries_with_backoff_delays_and_retry_events():
    reg = Registry()
    state = _failing(reg, [ValueError("a"), ConnectionError("b")], retries=2)
    ex, sink, ledger, clock = make(reg)
    result = await ex.run_call(ToolCall(name="flaky"))
    assert result.ok and result.attempts == 3 and state["n"] == 3
    assert clock.sleeps == [0.5, 1.0]
    assert check_events(sink) == ["tool.started", "tool.retry", "tool.retry", "tool.finished"]
    assert [e.data["attempt"] for e in sink.of_type("tool.retry")] == [1, 2]
    assert [e.data["delay_s"] for e in sink.of_type("tool.retry")] == [0.5, 1.0]
    assert ledger.entries()[0].attempts == 3


async def test_exhausted_foreign_errors_become_tool_failed_with_the_cause(caplog):
    reg = Registry()
    _failing(reg, [ValueError("a"), ValueError("b")], retries=1)
    ex, sink, ledger, _ = make(reg)
    with caplog.at_level(logging.ERROR, logger="agent_adapter.exec.pipeline"):
        result = await ex.run_call(ToolCall(name="flaky"))
    error = result.exception
    assert isinstance(error, ToolFailed) and error.attempts == 2
    assert isinstance(error.__cause__, ValueError) and error.cause_type == "ValueError"
    assert result.error.details == {"cause_type": "ValueError"} and result.attempts == 2
    assert sink.events[-1].data["error_kind"] == "tool-failed"
    assert ledger.entries()[0].error_kind == "tool-failed" and "tool flaky failed" in caplog.text


async def test_non_retryable_adapter_errors_are_raised_as_is_without_retry():
    reg = Registry()
    state = _failing(reg, [ToolError("bad query", details={"field": "q"})], retries=3)
    ex, *_ = make(reg)
    with pytest.raises(ToolError, match="bad query"):
        await ex.run("flaky")
    assert state["n"] == 1


async def test_retryable_adapter_errors_that_run_out_become_tool_failed():
    reg = Registry()
    _failing(reg, [ModelRateLimited("slow", provider="p")], retries=0)
    ex, *_ = make(reg)
    result = await ex.run_call(ToolCall(name="flaky"))
    assert isinstance(result.exception, ToolFailed)
    assert isinstance(result.exception.__cause__, ModelRateLimited)


async def test_timeouts_raise_tool_timeout_and_are_retried():
    reg = Registry()
    attempts = []

    @reg.tool(timeout=0.01, retries=1)
    async def hang() -> str:
        """Never returns in time."""
        attempts.append(1)
        await asyncio.sleep(1)
        return "late"

    ex, sink, _, _ = make(reg)
    result = await ex.run_call(ToolCall(name="hang"))
    assert isinstance(result.exception, ToolTimeout) and len(attempts) == 2
    assert result.exception.timeout == 0.01 and result.exception.call_id == result.call_id
    assert result.error.kind == "tool-timeout" and result.error.retryable
    assert sink.of_type("tool.retry")[0].data["error_kind"] == "tool-timeout"


async def test_on_tool_error_may_recover():
    class Fallback(Hook):
        async def on_tool_error(self, inv, error):
            assert isinstance(error, ToolFailed)
            return ShortCircuit("fallback")

    reg = Registry()
    _failing(reg, [ValueError("a")], cache_ttl=60)
    ex, sink, _, _ = make(reg, hooks=[Fallback()])
    result = await ex.run_call(ToolCall(name="flaky"))
    assert result.ok and result.value == "fallback" and result.attempts == 1
    assert sink.types()[-1] == "tool.finished"


@pytest.mark.parametrize("signal", [GraphBubbleUp("interrupt"), asyncio.CancelledError()])
async def test_control_flow_passes_through_without_hooks_ledger_or_failure(signal):
    reg = Registry()
    _failing(reg, [signal], retries=3)
    hook = RecordingHook()
    ex, sink, ledger, _ = make(reg, hooks=[hook])
    with pytest.raises(type(signal)):
        await ex.run_call(ToolCall(name="flaky"))
    assert hook.phases() == ["before_tool"] and ledger.entries() == []
    assert check_events(sink) == ["tool.started"]


async def test_empty_run_id_means_a_fresh_run_per_call():
    reg = Registry()
    _failing(reg, [], once_per_run=True)
    ex, sink, *_ = make(reg)
    ctx = ToolContext()
    assert await ex.run("flaky", ctx=ctx) == "ok"
    assert await ex.run("flaky", ctx=ctx) == "ok"  # once_per_run, but each call is its own run
    run_ids = {e.run_id for e in sink.events}
    assert len(run_ids) == 2 and all(r.startswith("run_") for r in run_ids)
    assert len(ex._settings.runs) == 0  # executor-made runs are ended after the call


async def test_caller_context_ports_win_and_executor_fills_the_gaps():
    own = RecordingSink()
    ex, sink, ledger, _ = make(registry)
    await ex.run("add", {"a": 1}, ToolContext(run_id="r", sink=own))
    assert own.types() == ["tool.started", "tool.finished"] and sink.events == []
    assert len(ledger.entries("r")) == 1


async def test_tools_get_a_child_context_with_the_executor_ports():
    reg = Registry()
    seen = []

    @reg.tool
    async def peek(ctx: ToolContext) -> str:
        """Look at the context."""
        seen.append(ctx)
        ctx.emit("progress", pct=50)
        return ctx.call_id

    ex, sink, ledger, _ = make(reg)
    call_id = await ex.run("peek", ctx=ex.context(run_id="r", extras={"k": 1}))
    ctx = seen[0]
    assert ctx.call_id == call_id and ctx.run_id == "r" and ctx.extras == {"k": 1}
    assert ctx.ledger is ledger and ctx.sink is ex.sink and ex.sink.sinks == (sink,)
    assert sink.of_type("custom.progress")[0].call_id == call_id


async def test_nested_calls_record_their_parent():
    reg = Registry()
    ex, sink, *_ = make(reg)

    @reg.tool
    async def outer(ctx: ToolContext) -> int:
        """Calls inner."""
        return await ex.run("inner", ctx=ctx)

    @reg.tool
    async def inner() -> int:
        """Leaf."""
        return 7

    assert await ex.run("outer", ctx=ex.context(run_id="r")) == 7
    outer_id = sink.events[0].call_id
    inner_events = [e for e in sink.events if e.data.get("tool") == "inner"]
    assert inner_events and all(e.parent_id == outer_id for e in inner_events)


async def test_a_raising_ledger_does_not_fail_the_call(caplog):
    class Broken:
        def record(self, entry):
            raise RuntimeError("disk full")

    ex = Executor(registry, ledger=Broken())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="agent_adapter.exec.pipeline"):
        assert await ex.run("add", {"a": 1}) == 2
    assert "ledger.record failed" in caplog.text


async def test_a_falsy_context_port_is_still_used():
    class EmptyLedger(list):
        def record(self, entry):
            self.append(entry)

    mine = EmptyLedger()
    ex, _, theirs, _ = make(registry)
    await ex.run("add", {"a": 1}, ToolContext(run_id="r", ledger=mine))  # type: ignore[arg-type]
    assert len(mine) == 1 and theirs.entries() == []
