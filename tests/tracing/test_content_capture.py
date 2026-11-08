"""Opt-in content on native events: only sinks that ask get it, and it is built only for them."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel

from agent_adapter.approval.builtin import AutoApprove
from agent_adapter.core.errors import AdapterError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import Event, ToolCall
from agent_adapter.evals import cases_from_runs
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.flow import Flow, FlowRunner
from agent_adapter.hooks.builtin import _content as safe
from agent_adapter.hooks.builtin.guards import PIIGuard
from agent_adapter.models.callable import CallableModel
from agent_adapter.tracing._content import content_data, emit_with_content, wants_content
from agent_adapter.tracing.sinks import (
    CONTENT_KEYS,
    CallbackEventSink,
    EventLog,
    FanoutSink,
    LoggingSink,
    SinkHub,
)
from agent_adapter.viewer import Recorder

registry = Registry()


@registry.tool
def order_status(order_id: str, api_key: str = "") -> dict[str, str]:
    """Where an order is."""
    return {"status": "shipped", "token": "t-123", "note": "key sk-live-abcdefghijklmnopqrstuv"}


@registry.tool(confirm=True)
def refund(card: str) -> str:
    """Refund to a card (needs approval)."""
    return f"refunded card {card}"


@registry.tool
def email_user(to: str) -> str:
    """Send an email."""
    return "sent"


def scripted(request: LLMRequest) -> LLMResponse:
    """Calls the tool the user names once, then answers."""
    asked = next(m.content for m in reversed(request.messages) if m.role == "user")
    if not any(m.role == "tool" for m in request.messages):
        tool, args = (
            ("refund", {"card": "4111 1111 1111 1111"})
            if "refund" in asked
            else ("order_status", {"order_id": "1042", "api_key": "k"})
        )
        call = ToolCall(name=tool, arguments=args, id="c1")
        message = Message(role="assistant", tool_calls=(call,))
        return LLMResponse(provider="local", model="m", tool_calls=(call,), message=message)
    message = Message(role="assistant", content="Order 1042 shipped.")
    return LLMResponse(provider="local", model="m", text=message.content, message=message)


class Wanting(EventLog):
    """An in-memory sink that asks for content and keeps what it may read."""

    wants_content = True
    wants_confirm_content = False

    def emit(self, event: Event) -> None:
        super().emit(dataclasses.replace(event, data=dict(content_data(event, self))))


def content_keys(events: list[Event]) -> set[str]:
    return {key for event in events for key in event.data if key in CONTENT_KEYS}


def first(events: list[Event], kind: str) -> Event:
    return next(e for e in events if e.type == kind)


async def loop(
    executor: Executor, asked: str = "Where is order 1042?", history: tuple[Message, ...] = ()
) -> None:
    messages = [*history, Message(role="user", content=asked)]
    await run_tool_loop(CallableModel(scripted), registry, messages, executor=executor)


async def test_only_the_asking_sink_gets_content_masked_after_the_guards() -> None:
    plain, wanting = EventLog(), Wanting()
    executor = Executor(registry, sink=FanoutSink(plain, CallbackEventSink(lambda e: None)))
    executor.add_sink(wanting)
    await loop(executor)
    assert content_keys(plain.events) == set()
    assert "arguments" not in first(wanting.events, "tool.started").data
    finished = first(wanting.events, "tool.finished").data
    assert finished["arguments"] == {"order_id": "1042", "api_key": "***"}
    assert finished["result"]["token"] == "***"
    assert finished["result"]["note"] == "key [SECRET]"  # a secret inside text, too
    llm = first(wanting.events, "llm.finished").data
    assert llm["input_messages"] == [{"role": "user", "content": "Where is order 1042?"}]
    assert llm["output_messages"][0]["tool_calls"] == [{"id": "c1", "name": "order_status"}]
    assert "input_messages" not in first(wanting.events, "llm.started").data
    assert len(plain.events) == len(wanting.events)


async def test_a_guard_blocked_payload_never_reaches_a_content_sink() -> None:
    wanting = Wanting()
    executor = Executor(registry, sink=wanting, hooks=[PIIGuard(on=("args",))])
    with pytest.raises(AdapterError):
        await executor.run("email_user", {"to": "alice@example.com"})
    assert content_keys(wanting.events) == set()
    assert [e.type for e in wanting.events][-1] == "tool.failed"


async def test_content_is_never_built_when_no_sink_asks() -> None:
    log = EventLog()
    executor = Executor(registry, sink=SinkHub(log))
    flow = Flow("f", state=Order).node("a", lambda s: {"answer": "x"}).end("a")
    with (
        mock.patch("agent_adapter.models._llm_call.message_dicts") as messages,
        mock.patch("agent_adapter.exec._call_content.safe_value") as values,
        mock.patch("agent_adapter.flow._drive.safe_value") as states,
    ):
        await loop(executor)
        await FlowRunner(executor).run(flow, {"question": "q"})
    assert not (messages.called or values.called or states.called)
    assert content_keys(log.events) == set() and not wants_content(executor.sink)


async def test_confirm_tools_withhold_results_unless_opted_in_and_never_twice() -> None:
    wanting, opted = Wanting(), Wanting()
    opted.wants_confirm_content = True
    executor = Executor(registry, sink=FanoutSink(wanting, opted), approval=AutoApprove())
    await loop(executor, "please refund me")
    for sink in (wanting, opted):
        assert "arguments" not in first(sink.events, "tool.finished").data
        second = [e for e in sink.events if e.type == "llm.finished"][1]
        tool_message = second.data["input_messages"][2]
        assert tool_message == {"role": "tool", "content": safe.WITHHELD, "name": "refund"}
    assert "result" not in first(wanting.events, "tool.finished").data
    assert first(opted.events, "tool.finished").data["result"] == "refunded card [CARD]"
    assert safe.tool_arguments(None, {"a": 1}) == {}


async def test_a_policy_or_rule_confirm_also_withholds() -> None:
    ruled = Registry()

    @ruled.tool(confirm=lambda *_: True)
    def payout(amount: int) -> str:
        """Pay out."""
        return f"paid {amount}"

    wanting = Wanting()
    executor = Executor(ruled, sink=wanting, approval=AutoApprove())
    await executor.run("payout", {"amount": 5})
    assert content_keys(wanting.events) == set()


async def test_logging_sink_never_logs_content(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    logging_sink = LoggingSink(logging.getLogger("events"), capture_content=True)
    executor = Executor(registry, sink=FanoutSink(logging_sink, Wanting()))
    await loop(executor)
    assert "Where is order 1042?" not in caplog.text and "1042" not in caplog.text


def test_a_failing_builder_or_sink_property_never_breaks_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wanting = Wanting()

    def broken() -> dict[str, Any]:
        raise ValueError("secret text")

    class Odd:
        @property
        def wants_content(self) -> bool:
            raise RuntimeError("odd")

        def emit(self, event: Event) -> None:
            pass

    with caplog.at_level(logging.WARNING):
        emit_with_content(wanting, Event("custom.x"), broken)
        emit_with_content(None, Event("custom.x"), broken)
        emit_with_content(FanoutSink(Odd(), wanting), Event("custom.y"), lambda: {"result": 1})
    assert [e.data for e in wanting.events] == [{}, {"result": 1}]
    assert "ValueError" in caplog.text and "secret text" not in caplog.text
    assert "RuntimeError" in caplog.text


class Order(BaseModel):
    question: str = ""
    password: str = ""
    answer: str = ""


async def test_flow_input_and_output_reach_only_content_sinks() -> None:
    flow = (
        Flow("answer", state=Order)
        .node("a", lambda state: {"answer": f"re: {state.question}"})
        .end("a")
    )
    plain, wanting = EventLog(), Wanting()
    executor = Executor(registry, sink=FanoutSink(plain, wanting))
    await FlowRunner(executor).run(flow, {"question": "hours?", "password": "p"})
    started = first(wanting.events, "flow.started")
    assert started.data["input"] == {"question": "hours?", "password": "***", "answer": ""}
    assert first(wanting.events, "flow.finished").data["output"]["answer"] == "re: hours?"
    assert content_keys(plain.events) == set()


async def test_cases_from_natively_recorded_runs_pick_the_current_question(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "runs"
    recorder = Recorder(folder, capture_content=True)
    executor = Executor(registry, sink=recorder)
    history = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello, how can I help?"),
    ]
    await loop(executor, history=tuple(history))
    flow = Flow("answer", state=Order).node("a", lambda s: {"answer": "open 9-5"}).end("a")
    await FlowRunner(executor).run(flow, {"question": "hours?"})
    assert recorder.flush()
    recorder.stop()
    found = cases_from_runs(folder)
    assert found.skipped == ()
    chat = next(c.case for c in found.cases if "tool:order_status" in c.case.tags)
    assert chat.input == "Where is order 1042?" and chat.expected == "Order 1042 shipped."
    flow_case = next(c.case for c in found.cases if "flow:answer" in c.case.tags)
    assert flow_case.input == {"question": "hours?", "password": "***", "answer": ""}
    assert flow_case.expected["answer"] == "open 9-5" and "masked" in flow_case.tags
    assert "masked" not in chat.tags


async def test_without_capture_the_recorder_asks_for_nothing(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "runs")
    assert recorder.wants_content is False and recorder.wants_confirm_content is False
    executor = Executor(registry, sink=recorder)
    await loop(executor)
    assert recorder.flush()
    recorder.stop()
    found = cases_from_runs(tmp_path / "runs")
    assert found.cases == () and "capture_content=True" in found.skipped[0].reason


def test_confirm_content_needs_content(tmp_path: Path) -> None:
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="capture_content=True"):
        Recorder(tmp_path / "runs", capture_confirm_content=True)
