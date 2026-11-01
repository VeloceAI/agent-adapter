"""node.finished `changed` (always) and `updates` (FlowRunner(trace_state=True) only)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from agent_adapter import Executor, Flow, FlowRunner
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.flow import Map
from agent_adapter.flow._trace import MAX_VALUE_CHARS, changed_fields, traced_updates
from agent_adapter.hooks.base import Hook
from agent_adapter.tracing import EventLog
from tests.flow.helpers import State, registry


def flow() -> Flow[State]:
    def plan(state: State) -> dict[str, Any]:
        return {"items": ["a", "b"], "note": "planned"}

    def quiet(state: State) -> None:
        return None

    return (
        Flow("traced", state=State)
        .node("plan", plan)
        .node("search", Map("items", tool="search", args_fn=lambda q: {"query": q}, into="hits"))
        .node("quiet", quiet)
        .edge("plan", "search")
        .edge("search", "quiet")
        .end("quiet")
    )


async def finished(runner_options: dict[str, Any]) -> tuple[EventLog, dict[str, dict[str, Any]]]:
    log = EventLog()
    runner = FlowRunner(Executor(registry, sink=log), **runner_options)
    result = await runner.run(flow(), {})
    assert result.status == "done"
    return log, {e.data["node"]: dict(e.data) for e in log.of_type("node.finished")}


async def test_node_finished_names_the_changed_fields_without_values_by_default() -> None:
    log, nodes = await finished({})
    assert nodes["plan"]["changed"] == ["items", "note"]
    assert nodes["search"]["changed"] == ["hits"]
    assert nodes["quiet"]["changed"] == []
    assert not any("updates" in data for data in nodes.values())
    (started,) = log.of_type("flow.started")
    assert started.data == {"flow": "traced"}


async def test_trace_state_adds_the_values_and_marks_the_flow() -> None:
    log, nodes = await finished({"trace_state": True})
    assert nodes["plan"]["updates"] == {"items": ["a", "b"], "note": "planned"}
    hits = nodes["search"]["updates"]["hits"]  # every item's value, in item order
    assert [[hit["source_id"] for hit in item] for item in hits] == [
        ["src-a", "shared"],
        ["src-b", "shared"],
    ]
    assert nodes["quiet"]["updates"] == {}
    (started,) = log.of_type("flow.started")
    assert started.data == {"flow": "traced", "trace_state": True}


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_trace_state_must_be_a_bool(bad: Any) -> None:
    with pytest.raises(ToolUsageError, match="trace_state"):
        FlowRunner(Executor(registry), trace_state=bad)


async def test_values_a_hook_replaced_are_the_ones_traced() -> None:
    class Rename(Hook):
        async def after_node(self, inv: Any, update: Any) -> Any:
            return {"note": "from hook"} if inv.node == "plan" else update

    log = EventLog()
    runner = FlowRunner(Executor(registry, sink=log), hooks=[Rename()], trace_state=True)
    await runner.run(flow(), {})
    plan = next(e for e in log.of_type("node.finished") if e.data["node"] == "plan")
    assert plan.data["changed"] == ["note"] and plan.data["updates"] == {"note": "from hook"}


def test_changed_fields_keep_first_seen_order_and_merge_repeats() -> None:
    assert changed_fields([{"b": 1, "a": 2}, {"a": 3, "c": 4}]) == ["b", "a", "c"]
    assert changed_fields([]) == []


class Point(BaseModel):
    x: int


def test_traced_updates_are_json_safe_and_only_for_changed_fields() -> None:
    view = {"p": Point(x=1), "n": 2, "raw": bytes(1), "hidden": 1}
    found = traced_updates(view, ["p", "n", "raw", "missing"])
    assert found == {"p": {"x": 1}, "n": 2, "raw": "AA=="}


def test_a_long_value_is_cut_to_the_cap() -> None:
    found = traced_updates({"text": "x" * 5000, "short": "ok"}, ["text", "short"])
    assert found["short"] == "ok"
    assert isinstance(found["text"], str) and found["text"].endswith(" chars]")
    assert len(found["text"]) < MAX_VALUE_CHARS + 40


def test_a_value_without_a_json_form_is_traced_as_its_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop: list[Any] = []
    loop.append(loop)  # a cycle: no JSON form
    assert traced_updates({"cycle": loop}, ["cycle"]) == {"cycle": "<list>"}
    assert "no JSON form" in caplog.text


def test_a_secret_inside_a_value_over_the_cap_is_masked_before_it_is_cut() -> None:
    config = {"api_key": "sk-LIVE-123456", "blob": "x" * 3000}
    found = traced_updates({"config": config}, ["config"])
    assert isinstance(found["config"], str) and found["config"].endswith(" chars]")
    assert "sk-LIVE-123456" not in found["config"] and '"api_key": "***"' in found["config"]


def test_a_secret_named_field_and_short_nested_secrets_are_masked() -> None:
    view = {"api_key": "sk-1", "settings": {"password": "pw", "n": 1}}
    found = traced_updates(view, ["api_key", "settings"])
    assert found == {"api_key": "***", "settings": {"password": "***", "n": 1}}


def slow_flow() -> Flow[State]:
    async def wait(state: State) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {"note": "late"}

    return Flow("slow", state=State).node("wait", wait).end("wait")


async def cancelled_run() -> EventLog:
    log = EventLog()
    executor = Executor(registry, sink=log)
    task = asyncio.create_task(FlowRunner(executor).run(slow_flow(), {}))
    while not log.of_type("node.started"):
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return log


async def test_a_cancelled_run_ends_its_flow_and_its_run() -> None:
    log = await cancelled_run()
    (finished,) = log.of_type("flow.finished")
    assert finished.data == {"flow": "slow", "status": "cancelled", "steps": 0}
    (ended,) = log.of_type("run.finished")
    assert ended.data == {"ok": False}
    assert not log.of_type("node.finished")


async def test_a_cancelled_flow_inside_a_caller_run_leaves_the_run_to_the_caller() -> None:
    log = EventLog()
    executor = Executor(registry, sink=log)
    async with executor.run_scope() as ctx:
        task = asyncio.create_task(FlowRunner(executor).run(slow_flow(), {}, ctx))
        while not log.of_type("node.started"):
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [e.data["status"] for e in log.of_type("flow.finished")] == ["cancelled"]
        assert not log.of_type("run.finished")  # the caller's scope ends its own run
