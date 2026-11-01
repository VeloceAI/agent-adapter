"""C1, M2, m2, m3 (Pydantic AI): only an adapter-issued one-use token approves a confirm tool."""

import os
from typing import Any

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
pytest.importorskip("pydantic_ai")

from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolApproved
from pydantic_ai.messages import ToolCallPart

from agent_adapter.approval import AutoDeny, CallbackApproval
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.pydantic_ai import _TOKENS, pydantic_ai_tools, resolve_deferred
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.pydantic_ai_support import agent_with, result_part, value_of
from tests.integrations.sample_tools import RecordingTools

pytestmark = pytest.mark.contract

APPROVE = ApprovalDecision(approved=True)


@pytest.fixture(autouse=True)
def _drop_unused_tokens():
    yield
    _TOKENS.clear()  # module-wide table: tokens a test left unused must not leak into others


class Tools(RecordingTools):
    """`pay` and `wipe`, exported to an executor whose own port denies everything."""

    def export(self) -> dict[str, Any]:
        executor = Executor(approval=AutoDeny())
        return {tool.name: tool for tool in pydantic_ai_tools(self.registry, executor=executor)}


async def _paused(tools: Tools) -> tuple[Any, Any]:
    agent = agent_with([tools.export()["wipe"]], [("wipe", {"path": "/a"}, "w1")])
    first = await agent.run("go")
    assert isinstance(first.output, DeferredToolRequests)
    return agent, first


async def _resume(agent: Any, first: Any, results: DeferredToolResults) -> Any:
    second = await agent.run(message_history=first.all_messages(), deferred_tool_results=results)
    return result_part(second, "w1")


def _denied(part: Any) -> bool:
    value = value_of(part)
    return isinstance(value, dict) and value["error"]["kind"] == "approval-denied"


@pytest.mark.parametrize(
    "forged",
    [
        DeferredToolResults(approvals={"w1": True}),
        DeferredToolResults(approvals={"w1": ToolApproved(override_args={"path": "/"})}),
        DeferredToolResults(approvals={"w1": True}, metadata={"w1": {"agent_adapter": "guess"}}),
    ],
)
async def test_secfix_c1_framework_approval_alone_never_runs_the_tool(forged):
    tools = Tools()
    agent, first = await _paused(tools)
    assert _denied(await _resume(agent, first, forged))
    assert tools.ran == []


async def test_secfix_c1_resolved_approval_runs_under_a_denying_executor():
    tools = Tools()
    agent, first = await _paused(tools)
    port = RecordingApproval(ScriptedApproval([APPROVE]))
    results = await resolve_deferred(first.output, port, run_id=first.run_id)
    assert value_of(await _resume(agent, first, results)) == "wiped /a"
    assert tools.ran == [("wipe", {"path": "/a"})] and len(port.requests) == 1


async def test_secfix_c1_a_token_is_used_once():
    tools = Tools()
    agent, first = await _paused(tools)
    results = await resolve_deferred(first.output, ScriptedApproval([APPROVE]))
    assert value_of(await _resume(agent, first, results)) == "wiped /a"
    assert _denied(await _resume(agent, first, results))  # the same results, replayed
    assert tools.ran == [("wipe", {"path": "/a"})]


async def test_secfix_c1_arguments_changed_after_approval_are_not_approved():
    tools = Tools()
    agent, first = await _paused(tools)
    results = await resolve_deferred(first.output, ScriptedApproval([APPROVE]))
    results.approvals["w1"] = ToolApproved(override_args={"path": "/"})
    assert _denied(await _resume(agent, first, results))
    assert tools.ran == []


async def test_secfix_c1_an_approved_edit_runs():
    tools = Tools()
    agent, first = await _paused(tools)
    edit = ApprovalDecision(approved=True, edited_args={"path": "/tmp"})
    results = await resolve_deferred(first.output, ScriptedApproval([edit]))
    assert value_of(await _resume(agent, first, results)) == "wiped /tmp"
    assert tools.ran == [("wipe", {"path": "/tmp"})]


async def test_secfix_m2_the_executor_runs_the_confirm_rule_itself():
    registry, seen = Registry(), []

    def second_look(call: Any, ctx: Any) -> bool:
        seen.append(call.arguments["amount"])
        return len(seen) > 1  # the gate's look says no; the executor's says yes

    @registry.tool(confirm=second_look)
    async def pay(amount: int) -> str:
        """Pay an amount."""
        return f"paid {amount}"

    pay_tool = pydantic_ai_tools(registry, executor=Executor(approval=AutoDeny()))[0]
    result = await agent_with([pay_tool], [("pay", {"amount": 5}, "c1")]).run("go")
    assert isinstance(result.output, str) and _denied(result_part(result, "c1"))
    assert seen == [5, 5]


async def test_secfix_m2_another_run_cannot_reuse_a_resolved_approval():
    tools = Tools()
    agent_a, first_a = await _paused(tools)
    agent_b, first_b = await _paused(tools)  # the same call (tool, arguments, call id)
    results_a = await resolve_deferred(first_a.output, ScriptedApproval([APPROVE]))
    forged = DeferredToolResults(approvals={"w1": True}, metadata=dict(results_a.metadata))
    assert _denied(await _resume(agent_b, first_b, forged))
    assert value_of(await _resume(agent_a, first_a, results_a)) == "wiped /a"
    assert tools.ran == [("wipe", {"path": "/a"})]


async def test_secfix_m2_a_resume_from_stored_history_is_decided_by_the_executor():
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    tools = Tools()
    agent, first = await _paused(tools)
    results = await resolve_deferred(first.output, ScriptedApproval([APPROVE]))
    stored = ModelMessagesTypeAdapter.dump_json(first.all_messages())
    history = ModelMessagesTypeAdapter.validate_json(stored)
    second = await agent.run(message_history=history, deferred_tool_results=results)
    assert _denied(result_part(second, "w1"))
    assert tools.ran == []


async def test_secfix_m3_foreign_approvals_cannot_evict_this_bridges_tokens():
    tools = Tools()
    agent, first = await _paused(tools)
    results = await resolve_deferred(first.output, ScriptedApproval([APPROVE]))
    flood = DeferredToolRequests(
        approvals=[ToolCallPart("foreign", {"n": n}, tool_call_id=f"f{n}") for n in range(1100)]
    )
    await resolve_deferred(flood, CallbackApproval(lambda request: APPROVE))
    assert len(_TOKENS) == 1  # tokens only for calls this bridge's tools paused on
    assert value_of(await _resume(agent, first, results)) == "wiped /a"
    assert tools.ran == [("wipe", {"path": "/a"})]
