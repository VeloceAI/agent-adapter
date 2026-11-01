"""Executor activity becomes AG-UI events: tool calls, results, errors, trace, run outcomes."""

import asyncio
import json
import logging
import threading

import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import EventType

from agent_adapter.core.errors import ApprovalPending, ToolError
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.integrations.ag_ui import AGUIBridge, AGUIRun
from agent_adapter.integrations.ag_ui._channel import EventChannel
from agent_adapter.tracing.sinks import EventLog
from tests.fakes.models import ScriptedModel, calls
from tests.integrations.ag_ui.support import Shop, collect, interrupts, run_input, types

pytestmark = pytest.mark.contract


async def test_a_call_streams_start_args_end_result():
    async def agent(run: AGUIRun):
        return (await run.call("total", {"order_id": "o-1"}, call_id="c1")).value

    events = await collect(Shop().bridge(trace=False), run_input("r"), agent)
    assert types(events) == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "RUN_FINISHED",
    ]
    started, args, _, result, finished = events[1:]
    assert started.tool_call_name == "total" and started.tool_call_id == "c1"
    assert json.loads(args.delta) == {"order_id": "o-1"}
    assert result.content == "120.0" and result.tool_call_id == "c1"
    assert events[0].protocol_version == "1.0"
    assert finished.result == 120.0 and finished.outcome is None


async def test_failures_and_unknown_tools_close_their_calls_with_an_error():
    shop = Shop()

    @shop.registry.tool
    def broken() -> str:
        """Always fails."""
        raise ToolError("down")

    async def agent(run: AGUIRun):
        await run.call("broken", call_id="b1")
        await run.call("nope", call_id="n1")
        await run.call("total", {"order_id": 5, "extra": 1}, call_id="x1")

    events = await collect(shop.bridge(trace=False), run_input("r"), agent)
    results = {
        e.tool_call_id: json.loads(e.content) for e in events if e.type.value == "TOOL_CALL_RESULT"
    }
    assert results["b1"]["error"]["kind"] == "tool-error"
    assert results["n1"]["error"]["kind"] == "tool-not-found"
    assert results["x1"]["error"]["kind"] == "tool-args-invalid"
    starts = [e.tool_call_id for e in events if e.type.value == "TOOL_CALL_START"]
    assert starts == ["b1", "n1", "x1"]


async def test_trace_events_become_custom_without_content():
    shop = Shop()
    log = EventLog()
    bridge = AGUIBridge(Executor(shop.registry, sink=log))

    async def agent(run: AGUIRun):
        run.ctx.emit("note", arguments="secret", n=1)
        return None

    events = await collect(bridge, run_input("r"), agent)
    (custom,) = [e for e in events if e.type.value == "CUSTOM"]
    assert custom.name == "agent_adapter.trace"
    assert custom.value["type"] == "custom.note" and custom.value["data"] == {"n": 1}
    assert [e.type for e in log.events] == ["custom.note"]  # the executor's own sink still gets it


async def test_text_and_custom_events():
    async def agent(run: AGUIRun):
        run.text("hi")
        run.text("")
        run.custom("app.progress", {"pct": 50})

    events = await collect(Shop().bridge(), run_input("r"), agent)
    assert types(events)[1:-1] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_END",
        "CUSTOM",
    ]


async def test_agent_errors_end_the_run_with_run_error(caplog):
    async def agent(run: AGUIRun):
        raise RuntimeError("secret detail")

    with caplog.at_level(logging.ERROR):
        events = await collect(Shop().bridge(), run_input("r"), agent)
    error = events[-1]
    assert error.type == EventType.RUN_ERROR and "secret" not in error.message
    assert "secret detail" not in caplog.text


async def test_pending_outside_a_tool_call_is_a_run_error():
    async def agent(run: AGUIRun):
        raise ApprovalPending("apr_1", tool="x")

    events = await collect(Shop().bridge(), run_input("r"), agent)
    assert events[-1].type == EventType.RUN_ERROR and events[-1].code == "approval-pending"


async def test_run_spec_raising_pending_still_interrupts():
    shop = Shop()

    async def agent(run: AGUIRun):
        spec = shop.registry.get("refund")
        await run.executor.run_spec(spec, {"order_id": "o", "amount": 1.0}, run.ctx, call_id="c9")

    events = await collect(shop.bridge(), run_input("r"), agent)
    (found,) = interrupts(events)
    assert found.tool_call_id == "c9" and found.reason == "tool_call"


async def test_run_tool_loop_streams_calls_and_pauses():
    shop = Shop()
    model = ScriptedModel(
        [calls(("total", {"order_id": "o-1"}), ("refund", {"order_id": "o-1", "amount": 1.0}))]
    )

    async def agent(run: AGUIRun):
        result = await run_tool_loop(
            model, shop.registry, run.messages(), executor=run.executor, ctx=run.ctx
        )
        return result.finish

    events = await collect(shop.bridge(trace=False), run_input("r"), agent)
    names = [e.tool_call_name for e in events if e.type.value == "TOOL_CALL_START"]
    assert names == ["total", "refund"]
    assert len(interrupts(events)) == 1


async def test_closing_the_stream_cancels_the_agent():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def agent(run: AGUIRun):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stream = Shop().bridge().stream(run_input("r"), agent)
    assert (await anext(stream)).type == EventType.RUN_STARTED
    await started.wait()
    await stream.aclose()
    assert cancelled.is_set()


async def test_channel_bounds_trace_events_and_accepts_other_threads(caplog):
    channel = EventChannel(max_events=1)
    channel.put("a", droppable=True)
    with caplog.at_level(logging.WARNING):
        channel.put("b", droppable=True)
    channel.put("c")
    thread = threading.Thread(target=channel.put, args=("d",))
    thread.start()
    thread.join()
    await asyncio.sleep(0)
    channel.close()
    channel.put("late")
    assert [item async for item in channel] == ["a", "c", "d"]
    assert channel.dropped == 1 and "dropping trace events" in caplog.text
