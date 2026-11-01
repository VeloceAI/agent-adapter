"""C1 + M2 (OpenAI Agents SDK): only an adapter-issued one-use token approves a confirm tool."""

from typing import Any

import pytest

pytest.importorskip("agents")

from agents import RunState

from agent_adapter.approval import AutoDeny
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.openai_agents import (
    _TOKENS,
    function_tools,
    resolve_interruptions,
)
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.openai_agents_support import agent_with, decoded, output_of, run
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
        return {tool.name: tool for tool in function_tools(self.registry, executor=executor)}


class Native:
    """A tool context whose SDK gate says "approved" and that may claim a run scope."""

    def __init__(self, call_id: str, scope: str | None = None) -> None:
        self.tool_call_id = call_id
        self._agent_tool_state_scope_id = scope

    def is_tool_approved(self, tool: str, call_id: str) -> bool:
        return True


def _denied(output: Any) -> bool:
    value = decoded(output)
    return isinstance(value, dict) and value["error"]["kind"] == "approval-denied"


async def _paused(tools: Tools, path: str = "/a", call_id: str = "w1") -> tuple[Any, Any, Any]:
    wipe = tools.export()["wipe"]
    agent = agent_with([wipe], [("wipe", {"path": path}, call_id)])
    first = await run(agent)
    assert [item.call_id for item in first.interruptions] == [call_id]
    return wipe, agent, first.to_state()


async def test_secfix_c1_sdk_approval_alone_never_runs_the_tool():
    tools = Tools()
    _, agent, state = await _paused(tools)
    state.approve(state.get_interruptions()[0], always_approve=True)  # host, not the port
    assert _denied(output_of(await run(agent, state), "w1"))
    assert tools.ran == []


async def test_secfix_c1_always_approve_does_not_cover_another_call():
    tools = Tools()
    wipe, _, state = await _paused(tools)
    state.approve(state.get_interruptions()[0], always_approve=True)
    assert _denied(await wipe.on_invoke_tool(Native("w2"), '{"path": "/"}'))
    assert tools.ran == []


async def test_secfix_c1_resolved_approval_runs_once_under_a_denying_executor():
    tools = Tools()
    wipe, agent, state = await _paused(tools)
    port = RecordingApproval(ScriptedApproval([APPROVE]))
    await resolve_interruptions(state, port)
    scope = state._agent_tool_state_scope_id
    other_args = await wipe.on_invoke_tool(Native("w1", scope), '{"path": "/"}')
    assert _denied(other_args)  # the token is bound to the approved arguments
    assert output_of(await run(agent, state), "w1") == "wiped /a"
    assert _denied(await wipe.on_invoke_tool(Native("w1", scope), '{"path": "/a"}'))  # used up
    assert tools.ran == [("wipe", {"path": "/a"})] and len(port.requests) == 1


async def test_secfix_c1_a_token_is_bound_to_its_run_state():
    tools = Tools()
    wipe = tools.export()["wipe"]
    agent = agent_with([wipe], [("wipe", {"path": "/a"}, "w1")])
    first = await run(agent)
    resolved, other = first.to_state(), first.to_state()
    await resolve_interruptions(resolved, ScriptedApproval([APPROVE]))
    other.approve(other.get_interruptions()[0])  # the SDK flag on another copy of the run
    assert _denied(output_of(await run(agent, other), "w1"))
    assert tools.ran == []


async def test_secfix_c1_a_round_tripped_state_is_decided_by_the_executor():
    tools = Tools()
    _, agent, state = await _paused(tools)
    await resolve_interruptions(state, ScriptedApproval([APPROVE]))
    restored = await RunState.from_json(agent, state.to_json())  # e.g. another process
    assert _denied(output_of(await run(agent, restored), "w1"))
    assert tools.ran == []


async def test_secfix_c1_an_edit_travels_in_the_token():
    tools = Tools()
    _, agent, state = await _paused(tools)
    edit = ApprovalDecision(approved=True, edited_args={"path": "/tmp"})
    await resolve_interruptions(state, ScriptedApproval([edit]))
    assert output_of(await run(agent, state), "w1") == "wiped /tmp"
    assert tools.ran == [("wipe", {"path": "/tmp"})]


async def test_secfix_c1_a_denial_is_rejected_on_the_state():
    tools = Tools()
    _, agent, state = await _paused(tools)
    await resolve_interruptions(state, ScriptedApproval([ApprovalDecision(approved=False)]))
    assert _denied(output_of(await run(agent, state), "w1"))
    assert tools.ran == []


async def test_secfix_m2_the_executor_runs_the_confirm_rule_itself():
    registry, seen = Registry(), []

    def second_look(call: Any, ctx: Any) -> bool:
        seen.append(call.arguments["amount"])
        return len(seen) > 1  # the gate's look says no; the executor's says yes

    @registry.tool(confirm=second_look)
    async def pay(amount: int) -> str:
        """Pay an amount."""
        return f"paid {amount}"

    pay_tool = function_tools(registry, executor=Executor(approval=AutoDeny()))[0]
    result = await run(agent_with([pay_tool], [("pay", {"amount": 5}, "c1")]))
    assert not result.interruptions and _denied(output_of(result, "c1"))
    assert seen == [5, 5]
