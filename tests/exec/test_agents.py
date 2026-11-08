import logging

import pytest

from agent_adapter import ApprovalDecision, Registry, ToolCall, ToolContext
from agent_adapter.core.errors import ConfigError, RunLimitExceeded, ToolError
from agent_adapter.core.types import ErrorInfo, ToolResult
from agent_adapter.exec import (
    AgentDepthExceeded,
    Executor,
    LoopAgent,
    RunLimits,
    agent_depth,
    agent_tool,
)
from agent_adapter.exec import agents as agents_module
from agent_adapter.viewer.lanes import build_spans
from tests.exec.agent_helpers import Scripted, shop
from tests.exec.helpers import DeferredFake
from tests.fakes.sinks import RecordingLedger, RecordingSink


def parent(*specs, **options):
    registry = Registry()
    for spec in specs:
        registry.add(spec)
    return registry, Executor(registry, **options)


async def test_task_agent_answers_through_the_executor():
    async def writer(task: str, ctx: ToolContext) -> str:
        return f"draft for {task} (depth {agent_depth()})"

    spec = agent_tool(writer, name="writer", description="Writes drafts.")
    assert spec.kind == "agent"
    assert spec.json_schema()["required"] == ["task"]
    assert "every detail" in spec.json_schema()["properties"]["task"]["description"]
    _, executor = parent(spec)
    assert await executor.run("writer", {"task": "a lamp ad"}) == "draft for a lamp ad (depth 1)"
    assert agent_depth() == 0


async def test_sync_task_agent_runs_in_a_thread():
    spec = agent_tool(lambda task, ctx: task.upper(), name="shout", description="Shouts.")
    _, executor = parent(spec)
    assert await executor.run("shout", {"task": "hi"}) == "HI"


async def test_sync_agent_returning_an_awaitable_is_awaited():
    async def later(task):
        return task + "!"

    spec = agent_tool(lambda task, ctx: later(task), name="later", description="Later.")
    _, executor = parent(spec)
    assert await executor.run("later", {"task": "hi"}) == "hi!"


async def test_a_non_text_answer_is_a_usage_error():
    spec = agent_tool(lambda task, ctx: 42, name="numbers", description="Numbers.")
    _, executor = parent(spec)
    result = await executor.run_call(ToolCall(name="numbers", arguments={"task": "x"}))
    assert result.error.kind == "tool-usage-error"


async def test_loop_agent_rolls_cost_events_and_limits_up_to_the_parent_run():
    ledger, sink = RecordingLedger(), RecordingSink()
    sub = LoopAgent(
        Scripted("stock", {"item": "lamp"}),
        shop().subset(names=["stock"]),
        instructions="You count stock.",
    )
    registry = Registry()
    executor = Executor(registry, ledger=ledger, sink=sink)
    registry.add(agent_tool(sub, name="clerk", description="Counts stock.", executor=executor))
    ctx = executor.context()
    result = await executor.run_call(
        ToolCall(name="clerk", arguments={"task": "lamps?"}, id="call_clerk"), ctx
    )
    assert result.value == "done: 3"
    assert sub.model.requests[0].messages[0].content == "You count stock."
    kinds = sorted(entry.kind for entry in ledger.entries(ctx.run_id))
    assert kinds == ["llm", "llm", "tool", "tool"]  # two sub-agent turns, its tool, the agent
    nested = [e for e in sink.events if e.type == "tool.started" and e.data["tool"] == "stock"]
    assert nested[0].parent_id == "call_clerk"
    assert {e.parent_id for e in sink.of_type("llm.started")} == {"call_clerk"}
    assert [e.type for e in sink.of_type("custom.agent")] == [
        "custom.agent.started",
        "custom.agent.finished",
    ]
    spans = build_spans([e.to_dict() for e in sink.events])
    by_name = {span["name"]: span for span in spans if span["family"] == "tool"}
    assert by_name["stock"]["parent_id"] == by_name["clerk"]["call_id"] == "call_clerk"


async def test_run_limits_count_the_sub_agents_calls():
    sub = LoopAgent(Scripted("stock", {"item": "lamp"}), shop().subset(names=["stock"]))
    registry = Registry()
    executor = Executor(registry, limits=RunLimits(max_calls=1))
    registry.add(agent_tool(sub, name="clerk", description="Counts.", executor=executor))
    result = await executor.run_call(ToolCall(name="clerk", arguments={"task": "lamps?"}))
    assert result.value.startswith("done: ")
    assert "run-limit-exceeded" in result.value


async def test_recursion_stops_at_max_depth():
    registry = Registry()
    executor = Executor(registry)

    async def again(task: str, ctx: ToolContext) -> str:
        nested = await executor.run_call(ToolCall(name="loop", arguments={"task": task}), ctx)
        return nested.value if nested.ok else nested.error.kind

    registry.add(agent_tool(again, name="loop", description="Calls itself.", max_depth=2))
    assert await executor.run("loop", {"task": "x"}) == "agent-depth-exceeded"


def test_depth_error_is_a_run_limit():
    error = AgentDepthExceeded("loop", 2, run_id="run_1")
    assert isinstance(error, RunLimitExceeded)
    assert (error.limit, error.max_depth, error.run_id) == ("agent_depth", 2, "run_1")
    assert ErrorInfo.from_exception(error).kind == "agent-depth-exceeded"


async def test_a_loop_that_runs_out_of_steps_fails_the_call():
    sub = LoopAgent(
        Scripted("stock", {"item": "lamp"}), shop().subset(names=["stock"]), max_steps=1
    )
    registry = Registry()
    executor = Executor(registry)
    registry.add(agent_tool(sub, name="clerk", description="Counts.", executor=executor))
    result = await executor.run_call(ToolCall(name="clerk", arguments={"task": "lamps?"}))
    assert result.error.kind == "tool-error"
    assert "max_steps" in result.error.message


async def test_a_pending_sub_call_pauses_the_parent_and_resumes_the_sub_agent():
    port = DeferredFake()
    model = Scripted("refund", {"order": "A1"})
    sub = LoopAgent(model, shop().subset(names=["refund"]))
    registry = Registry()
    executor = Executor(registry, approval=port)
    registry.add(agent_tool(sub, name="desk", description="Refunds.", executor=executor))
    ctx = executor.context()
    call = ToolCall(name="desk", arguments={"task": "refund A1"}, id="call_desk")
    first = await executor.run_call(call, ctx)
    assert first.status == "pending"
    assert first.error.details["approval_id"] == "apr_pending"
    port.decide(ctx.run_id, "c_refund", ApprovalDecision(approved=True, decided_by="lead"))
    second = await executor.run_call(call, ctx)
    assert second.value == "done: refunded A1"
    assert len(model.requests) == 2  # resumed from the parked history, not restarted


async def test_parked_histories_are_capped(monkeypatch, caplog):
    monkeypatch.setattr(agents_module, "MAX_PARKED", 1)
    port = DeferredFake()
    sub = LoopAgent(Scripted("refund", {"order": "A1"}), shop().subset(names=["refund"]))
    registry = Registry()
    executor = Executor(registry, approval=port)
    registry.add(agent_tool(sub, name="desk", description="Refunds.", executor=executor))
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            await executor.run_call(ToolCall(name="desk", arguments={"task": "refund A1"}))
    assert "dropped the paused history" in caplog.text


def test_pending_without_the_exception_is_rebuilt():
    result = ToolResult(
        "c1",
        "refund",
        "pending",
        error=ErrorInfo(kind="approval-pending", message="", details={"approval_id": "apr_9"}),
    )
    rebuilt = agents_module._pending([ToolResult("c0", "stock", "ok", value=1), result])
    assert (rebuilt.approval_id, rebuilt.tool, rebuilt.call_id) == ("apr_9", "refund", "c1")


@pytest.mark.parametrize(
    ("agent", "options", "message"),
    [
        (LoopAgent(Scripted(), Registry()), {}, "needs executor"),
        (LoopAgent(Scripted(), Registry(), max_steps=0), {"executor": Executor()}, "max_steps"),
        ("not an agent", {}, "LoopAgent or a callable"),
        (lambda t, c: t, {"max_depth": 0}, "max_depth"),
        (lambda t, c: t, {"max_depth": True}, "max_depth"),
    ],
)
def test_bad_configuration(agent, options, message):
    with pytest.raises(ConfigError, match=message):
        agent_tool(agent, name="a", description="An agent.", **options)


async def test_a_budget_on_the_parent_run_stops_the_sub_agent():
    from agent_adapter.cost import Ledger, Prices
    from agent_adapter.hooks import BudgetHook

    prices = Prices.load(
        {"version": "t", "models": {"scripted": {"provider": "host", "input": 1e5, "output": 0}}}
    )
    sub = LoopAgent(Scripted("stock", {"item": "lamp"}), shop().subset(names=["stock"]))
    registry = Registry()
    executor = Executor(registry, ledger=Ledger(prices=prices), hooks=[BudgetHook(0.5)])
    registry.add(agent_tool(sub, name="clerk", description="Counts.", executor=executor))
    result = await executor.run_call(ToolCall(name="clerk", arguments={"task": "lamps?"}))
    assert result.error.kind == "budget-exceeded"  # the sub-agent's second model call


async def test_the_parent_loop_resumes_a_paused_sub_agent():
    from agent_adapter import Message, run_tool_loop

    port = DeferredFake()
    sub_model = Scripted("refund", {"order": "A1"})
    registry = Registry()
    executor = Executor(registry, approval=port)
    registry.add(
        agent_tool(
            LoopAgent(sub_model, shop().subset(names=["refund"])),
            name="desk",
            description="Refunds.",
            executor=executor,
        )
    )
    boss = Scripted("desk", {"task": "refund A1"})
    ctx = executor.context()
    first = await run_tool_loop(
        boss, registry, [Message(role="user", content="refund A1")], executor=executor, ctx=ctx
    )
    assert first.finish == "pending"
    port.decide(ctx.run_id, "c_refund", ApprovalDecision(approved=True, decided_by="lead"))
    second = await run_tool_loop(boss, registry, first.messages, executor=executor, ctx=ctx)
    assert (second.finish, second.final.text) == ("stop", "done: done: refunded A1")
    assert len(sub_model.requests) == 2


# Review fix round 1: the reviewer's scenarios.


def desk_with(port, **options):
    model = Scripted("refund", {"order": "A1"})
    registry = Registry()
    executor = Executor(registry, approval=port, sink=RecordingSink())
    registry.add(
        agent_tool(
            LoopAgent(model, shop().subset(names=["refund"])),
            name="desk",
            description="Refunds.",
            executor=executor,
            **options,
        )
    )
    return model, executor


async def test_a_confirmed_agent_tool_is_not_asked_again_when_its_sub_agent_resumes():
    port = DeferredFake()
    _, executor = desk_with(port, confirm=True)
    ctx = executor.context()
    call = ToolCall(name="desk", arguments={"task": "refund A1"}, id="call_desk")
    assert (await executor.run_call(call, ctx)).status == "pending"  # the desk itself
    port.decide(ctx.run_id, "call_desk", ApprovalDecision(approved=True, decided_by="lead"))
    assert (await executor.run_call(call, ctx)).status == "pending"  # the refund inside
    port.decide(ctx.run_id, "c_refund", ApprovalDecision(approved=True, decided_by="lead"))
    assert (await executor.run_call(call, ctx)).value == "done: refunded A1"
    assert [r.tool for r in port.requests] == ["desk", "desk", "refund", "refund"]


async def test_a_resumed_call_with_another_task_asks_and_restarts():
    port = DeferredFake()
    _, executor = desk_with(port, confirm=True)
    ctx = executor.context()
    port.decide(ctx.run_id, "call_desk", ApprovalDecision(approved=True, decided_by="lead"))
    first = ToolCall(name="desk", arguments={"task": "refund A1"}, id="call_desk")
    assert (await executor.run_call(first, ctx)).status == "pending"
    other = ToolCall(name="desk", arguments={"task": "refund B2"}, id="call_desk")
    await executor.run_call(other, ctx)
    assert [r.tool for r in port.requests][-1] == "desk"  # asked again, not skipped


async def test_agent_finished_is_sent_for_every_outcome():
    sink = RecordingSink()
    registry = Registry()
    failing = Executor(registry, sink=sink)

    def broken(task, ctx):
        raise ToolError("no")

    registry.add(agent_tool(broken, name="broken", description="Fails."))
    registry.add(agent_tool(lambda t, c: t, name="tiny", description="Tiny.", max_depth=1))

    async def nested(task, ctx):
        result = await failing.run_call(ToolCall(name="tiny", arguments={"task": task}), ctx)
        return result.error.kind

    registry.add(agent_tool(nested, name="outer", description="Outer."))
    await failing.run_call(ToolCall(name="broken", arguments={"task": "x"}))
    assert await failing.run("outer", {"task": "x"}) == "agent-depth-exceeded"
    statuses = [
        (e.type, e.data["agent"], e.data.get("status")) for e in sink.of_type("custom.agent")
    ]
    assert statuses == [
        ("custom.agent.started", "broken", None),
        ("custom.agent.finished", "broken", "error"),
        ("custom.agent.started", "outer", None),
        ("custom.agent.refused", "tiny", None),
        ("custom.agent.finished", "outer", "ok"),
    ]


async def test_a_pending_sub_agent_reports_pending():
    sink = RecordingSink()
    port = DeferredFake()
    registry = Registry()
    executor = Executor(registry, approval=port, sink=sink)
    registry.add(
        agent_tool(
            LoopAgent(Scripted("refund", {"order": "A1"}), shop().subset(names=["refund"])),
            name="desk",
            description="Refunds.",
            executor=executor,
        )
    )
    await executor.run_call(ToolCall(name="desk", arguments={"task": "refund A1"}))
    assert sink.of_type("custom.agent.finished")[0].data["status"] == "pending"


async def test_paused_histories_expire(monkeypatch):
    port = DeferredFake()
    model, executor = desk_with(port)
    ctx = executor.context()
    call = ToolCall(name="desk", arguments={"task": "refund A1"}, id="call_desk")
    await executor.run_call(call, ctx)
    monkeypatch.setattr(agents_module, "PARK_TTL_S", -1.0)
    port.decide(ctx.run_id, "c_refund", ApprovalDecision(approved=True, decided_by="lead"))
    await executor.run_call(call, ctx)
    assert len(model.requests) == 3  # the expired history was dropped: the task restarted
    assert model.requests[1].messages == model.requests[0].messages
