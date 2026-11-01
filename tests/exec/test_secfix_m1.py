"""M1: one pre-approval (`ctx.approval`) never covers more than the one call it was made for."""

import pytest

from agent_adapter import ApprovalDecision, ApprovalDenied, Registry, Step, ToolCall, ToolContext
from agent_adapter.approval import AutoDeny
from agent_adapter.core.messages import Message
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.runtime.current import current_context
from tests.fakes.models import ScriptedModel, calls, text

YES = ApprovalDecision(approved=True)

registry = Registry()
ran: list[str] = []


@registry.tool(confirm=True)
def wire(amount: int) -> str:
    """Wire money."""
    ran.append("wire")
    return "sent"


@registry.tool(confirm=True)
def drop_db() -> str:
    """Drop the database."""
    ran.append("drop_db")
    return "dropped"


WIRE = ToolCall(name="wire", arguments={"amount": 1}, id="c1")
DROP = ToolCall(name="drop_db", arguments={}, id="c2")


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


def _executor() -> Executor:
    return Executor(registry, approval=AutoDeny())


async def test_secfix_m1_unbound_pre_approval_covers_no_call_of_a_batch() -> None:
    results = await _executor().run_many([WIRE, DROP], ToolContext(run_id="r1", approval=YES))
    assert [r.error.kind for r in results if r.error] == ["approval-denied", "approval-denied"]
    assert ran == []


async def test_secfix_m1_bound_pre_approval_covers_no_call_of_a_batch() -> None:
    ctx = ToolContext(run_id="r1", call_id="c1", approval=YES)
    results = await _executor().run_many([WIRE, DROP], ctx)
    assert [r.ok for r in results] == [False, False] and ran == []


async def test_secfix_m1_as_completed_does_not_spread_a_pre_approval() -> None:
    ctx = ToolContext(run_id="r1", approval=YES)
    results = [r async for r in _executor().as_completed([WIRE, DROP], ctx)]
    assert all(r.status == "error" for r in results) and ran == []


async def test_secfix_m1_run_plan_does_not_spread_a_pre_approval() -> None:
    steps = [Step(id="a", tool="wire", args={"amount": 1}), Step(id="b", tool="drop_db")]
    results = await _executor().run_plan(steps, ToolContext(run_id="r1", approval=YES))
    assert not results["a"].ok and not results["b"].ok and ran == []


async def test_secfix_m1_run_tool_loop_does_not_spread_a_pre_approval() -> None:
    model = ScriptedModel([calls(("wire", {"amount": 1}), ("drop_db", {})), text("done")])
    ctx = ToolContext(run_id="r1", approval=YES)
    await run_tool_loop(
        model, registry, (Message(role="user", content="go"),), executor=_executor(), ctx=ctx
    )
    assert ran == []


async def test_secfix_m1_single_call_still_honours_an_unbound_pre_approval() -> None:
    assert await _executor().run("wire", {"amount": 1}, ToolContext(run_id="r1", approval=YES))
    assert ran == ["wire"]


async def test_secfix_m1_pre_approval_for_another_call_id_is_ignored() -> None:
    ctx = ToolContext(run_id="r1", call_id="c1", approval=YES)
    with pytest.raises(ApprovalDenied):
        await _executor().run("drop_db", {}, ctx, call_id="c2")
    assert ran == []


async def test_secfix_m1_tool_loop_call_reusing_the_bound_call_id_asks_the_port() -> None:
    model = ScriptedModel([calls(ToolCall(name="drop_db", arguments={}, id="c1")), text("done")])
    ctx = ToolContext(run_id="r1", call_id="c1", approval=YES)
    await run_tool_loop(
        model, registry, (Message(role="user", content="go"),), executor=_executor(), ctx=ctx
    )
    assert ran == []


async def test_secfix_m1_tool_body_context_carries_no_pre_approval() -> None:
    reg = Registry()
    ex = Executor(reg, approval=AutoDeny())
    reg.add(registry.get("drop_db"))
    seen: list[object] = []

    @reg.tool(confirm=True, parallel_safe=True)
    async def outer(ctx: ToolContext) -> str:
        """Runs drop_db with the body's own context and call id."""
        here = current_context()
        assert here is not None
        seen.extend([ctx.approval, here.approval])
        return str(await ex.run("drop_db", {}, here, call_id=here.call_id))

    ctx = ToolContext(run_id="r1", call_id="o1", approval=YES)
    result = await ex.run_call(ToolCall(name="outer", arguments={}, id="o1"), ctx)
    assert result.error is not None and result.error.kind == "approval-denied"
    assert seen == [None, None] and ran == []
