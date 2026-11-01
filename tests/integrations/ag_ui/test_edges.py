"""Edge cases: replaced results, failing sinks, unknown arguments, bounded tables."""

import json
import logging

import pytest

pytest.importorskip("ag_ui")

from agent_adapter.approval import AutoApprove
from agent_adapter.core.types import ErrorInfo, ToolResult
from agent_adapter.exec import Executor
from agent_adapter.hooks import Hook
from agent_adapter.integrations.ag_ui import AGUIBridge, AGUIInputInvalid, AGUIRun, PendingCall
from agent_adapter.integrations.ag_ui._channel import EventChannel
from agent_adapter.integrations.ag_ui._events import ToolEvents
from agent_adapter.integrations.ag_ui._input import MAX_MESSAGES
from agent_adapter.integrations.ag_ui._interrupts import InterruptTable
from agent_adapter.tracing.events import make_event
from tests.integrations.ag_ui.support import (
    Shop,
    approve,
    collect,
    refund_agent,
    run_input,
)

pytestmark = pytest.mark.contract


class _Fails(Hook):
    async def after_tool(self, inv, result):
        return ToolResult(
            inv.call.id, inv.spec.name, "error", error=ErrorInfo(kind="x", message="m")
        )


async def test_a_result_replaced_by_an_error_is_sent_as_one():
    shop = Shop()
    bridge = AGUIBridge(Executor(shop.registry, hooks=[_Fails()]), trace=False)

    async def agent(run: AGUIRun):
        await run.call("total", {"order_id": "o"}, call_id="c1")

    events = await collect(bridge, run_input("r"), agent)
    (result,) = [e for e in events if e.type.value == "TOOL_CALL_RESULT"]
    assert json.loads(result.content) == {"error": {"kind": "x", "message": ""}}


async def test_a_failing_event_conversion_is_logged_not_raised(caplog, monkeypatch):
    events = ToolEvents(EventChannel())
    monkeypatch.setattr(events, "_on", lambda event: 1 / 0)
    with caplog.at_level(logging.WARNING):
        events.emit(make_event("custom.x"))
    assert "not sent" in caplog.text


async def test_a_second_result_for_a_call_is_dropped():
    channel = EventChannel()
    events = ToolEvents(channel)
    events.denied("c1", "t", RuntimeError("x"))
    events.denied("c1", "t", RuntimeError("y"))
    channel.close()
    sent = [e.type.value async for e in channel]
    assert sent == ["TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_RESULT"]
    channel.close()  # closing twice is harmless


@pytest.mark.parametrize(
    ("tool", "arguments", "kind"),
    [
        ("refund", None, "approval-denied"),
        ("not_registered", {"x": 1}, "approval-denied"),
        ("refund", {"order_id": 1}, "approval-denied"),
    ],
)
async def test_unresumable_interrupts_deny(tool, arguments, kind):
    shop = Shop()
    bridge = shop.bridge(AutoApprove())
    bridge.interrupts.open(
        PendingCall("int_1", "t-1", "", "run_x", "c1", tool, arguments, expires_at=1.0)
    )
    events = await collect(bridge, run_input("r", resume=[approve("int_1")]), refund_agent)
    (result,) = [e for e in events if e.type.value == "TOOL_CALL_RESULT"]
    assert json.loads(result.content)["error"]["kind"] == kind
    assert shop.refunds == []


async def test_a_pause_without_known_arguments_still_interrupts(caplog):
    shop = Shop()

    async def agent(run: AGUIRun):
        run.events.emit(make_event("tool.pending", call_id="c1", tool="refund", approval_id="a"))

    with caplog.at_level(logging.WARNING):
        events = await collect(shop.bridge(), run_input("r"), agent)
    assert events[-1].outcome.interrupts[0].tool_call_id == "c1"
    assert "unknown arguments" in caplog.text


def test_a_full_interrupt_table_drops_the_oldest(caplog):
    table = InterruptTable(max_interrupts=1)
    with caplog.at_level(logging.WARNING):
        table.open(PendingCall("a", "t", "", "r", "c1", "x"))
        table.open(PendingCall("b", "t", "", "r", "c2", "x"))
    assert len(table) == 1 and "dropped interrupt" in caplog.text
    assert table.take("a", thread_id="t", principal="") is None


def test_too_many_messages_are_refused():
    from agent_adapter.integrations.ag_ui import parse_input

    messages = [{"id": str(n), "role": "user", "content": "x"} for n in range(MAX_MESSAGES + 1)]
    with pytest.raises(AGUIInputInvalid, match="messages"):
        parse_input({"threadId": "t", "runId": "r", "messages": messages})


def test_one_principal_over_its_cap_drops_only_its_own_interrupts(caplog):
    table = InterruptTable(max_interrupts=10, max_per_principal=2)
    table.open(PendingCall("bob-1", "t", "bob", "r", "c0", "x"))
    with caplog.at_level(logging.WARNING):
        for n in range(5):
            table.open(PendingCall(f"eve-{n}", "t", "eve", "r", f"c{n}", "x"))
    assert len(table) == 3 and "dropped interrupt" in caplog.text
    assert table.take("bob-1", thread_id="t", principal="bob") is not None
    assert table.take("eve-2", thread_id="t", principal="eve") is None
    assert table.take("eve-4", thread_id="t", principal="eve") is not None


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_a_bad_per_principal_cap_is_refused(cap):
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="max_per_principal"):
        InterruptTable(max_per_principal=cap)


def test_an_interrupt_expiring_while_the_cap_is_enforced_is_not_counted_twice():
    ticks = iter([0.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0])
    table = InterruptTable(ttl=10, max_per_principal=1, clock=lambda: next(ticks, 99.0))
    table.open(PendingCall("a", "t", "ana", "r", "c1", "x"))
    table.open(PendingCall("b", "t", "ana", "r", "c2", "x"))  # "a" expires mid-enforcement
    assert table.take("b", thread_id="t", principal="ana") is not None


async def test_the_overflow_warning_is_logged_once(caplog):
    channel = EventChannel(max_events=1)
    channel.put("a", droppable=True)
    with caplog.at_level(logging.WARNING):
        for item in ("b", "c", "d"):
            channel.put(item, droppable=True)
    assert channel.dropped == 3 and caplog.text.count("dropping trace events") == 1
