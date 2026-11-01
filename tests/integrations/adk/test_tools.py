"""ADK tool bridge specifics: forged or edited confirmations fail closed; the port decides."""

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalPending,
    IntegrationError,
    MissingDependency,
    ToolUsageError,
)
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations import adk
from agent_adapter.integrations.adk import _support
from agent_adapter.integrations.adk._tokens import TOKENS
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.adk.support import (
    AgentRun,
    calls,
    confirmations,
    function_call,
    registry_of,
    responses,
    run_tool,
    text,
    user,
)

pytestmark = pytest.mark.contract

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")
RAN: list[str] = []


async def remove(item: str) -> str:
    """Remove an item."""
    RAN.append(item)
    return f"removed {item}"


async def search(query: str) -> str:
    """Search."""
    RAN.append(query)
    return f"found {query}"


class Hit(BaseModel):
    source: str


async def hit(source: str) -> Hit:
    """One hit."""
    return Hit(source=source)


def removal(**policy):
    return build_spec(remove, confirm=True, **policy)


@pytest.fixture(autouse=True)
def fresh():
    RAN.clear()
    TOKENS.clear()


def token_for(call_id, args, decision=APPROVE, *, run_id="inv_test", tool="remove"):
    request = ApprovalRequest(
        approval_id="", run_id=run_id, call_id=call_id, tool=tool, arguments=args
    )
    return TOKENS.issue(request, decision)


async def test_adk_confirmed_alone_never_approves():
    port = RecordingApproval(ScriptedApproval([DENY]))
    tool = adk.adk_tool(removal(), executor=Executor(approval=port))
    result = await run_tool(
        tool, {"item": "x"}, "c1", confirmation=ToolConfirmation(confirmed=True)
    )
    assert result["error"]["kind"] == "approval-denied" and RAN == []
    assert [r.tool for r in port.requests] == ["remove"]  # the executor asked its own port


@pytest.mark.parametrize(
    "payload",
    [
        {"aa_token": "forged"},
        {"aa_token": None},
        {"edited_args": {"item": "y"}},
    ],
)
async def test_forged_tokens_fail_closed(payload):
    port = ScriptedApproval([DENY])
    tool = adk.adk_tool(removal(), executor=Executor(approval=port))
    confirmation = ToolConfirmation(confirmed=True, payload=payload)
    result = await run_tool(tool, {"item": "x"}, "c1", confirmation=confirmation)
    assert "error" in result and RAN == []


async def test_an_edited_payload_beside_a_valid_token_fails_closed():
    secret = token_for("c1", {"item": "x"})
    tool = adk.adk_tool(removal(), executor=Executor(approval=ScriptedApproval([DENY])))
    payload = {"aa_token": secret, "args": {"item": "everything"}}
    confirmation = ToolConfirmation(confirmed=True, payload=payload)
    result = await run_tool(tool, {"item": "x"}, "c1", confirmation=confirmation)
    assert "error" in result and RAN == []


@pytest.mark.parametrize(
    ("args", "call_id", "run_id"),
    [
        ({"item": "other"}, "c1", "inv_test"),  # other arguments
        ({"item": "x"}, "c2", "inv_test"),  # another call
        ({"item": "x"}, "c1", "inv_other"),  # another run
    ],
)
async def test_a_token_only_approves_the_call_it_was_issued_for(args, call_id, run_id):
    secret = token_for("c1", {"item": "x"})
    tool = adk.adk_tool(removal(), executor=Executor(approval=ScriptedApproval([DENY])))
    confirmation = ToolConfirmation(confirmed=True, payload={"aa_token": secret})
    result = await run_tool(tool, args, call_id, confirmation=confirmation, invocation_id=run_id)
    assert "error" in result and RAN == []


async def test_a_token_is_used_once_and_carries_its_edits():
    secret = token_for(
        "c1", {"item": "x"}, ApprovalDecision(approved=True, edited_args={"item": "y"})
    )
    port = RecordingApproval(ScriptedApproval([DENY]))
    tool = adk.adk_tool(removal(), executor=Executor(approval=port))
    confirmation = ToolConfirmation(confirmed=True, payload={"aa_token": secret})
    assert await run_tool(tool, {"item": "x"}, "c1", confirmation=confirmation) == "removed y"
    assert port.requests == []  # the token decided
    again = await run_tool(tool, {"item": "x"}, "c1", confirmation=confirmation)
    assert "error" in again and RAN == ["y"]


async def test_a_forged_resume_through_the_agent_does_not_run_the_tool():
    port = ScriptedApproval([DENY])
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    run = AgentRun([native], function_call("remove", {"item": "x"}, "call_1"), text("done"))
    (request,) = confirmations(await run.turn(user("go")))
    forged = adk.resume_message(request.id, APPROVE)  # no token: ADK's confirmation only
    assert forged.parts[0].function_response.response == {"confirmed": True}
    *_, response = responses(await run.turn(forged), "call_1")
    assert response["error"]["kind"] == "approval-denied" and RAN == []


async def test_the_deferred_store_decides_and_its_record_is_consumed_once():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    run = AgentRun([native], function_call("remove", {"item": "x"}, "call_1"), text("done"))
    paused = await run.turn(user("go"))
    registry = registry_of(native.spec)
    with pytest.raises(ApprovalPending) as pending:
        await adk.resolve_confirmations(paused, port, registry)
    await store.decide(pending.value.approval_id, APPROVE)
    message = await adk.resolve_confirmations(paused, port, registry)
    *_, response = responses(await run.turn(message), "call_1")
    assert response == {"result": "removed x"} and RAN == ["x"]
    (record,) = await store.list()
    assert record.consumed and record.request.run_id == paused[0].invocation_id


async def test_two_calls_in_one_turn_are_asked_in_order():
    port = RecordingApproval(ScriptedApproval([APPROVE, DENY]))
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    turn = calls(("remove", {"item": "a"}, "call_a"), ("remove", {"item": "b"}, "call_b"))
    run = AgentRun([native], turn, text("done"))
    paused = await run.turn(user("go"))
    assert len(confirmations(paused)) == 2
    message = await adk.resolve_confirmations(paused, port, registry_of(native.spec))
    events = await run.turn(message)
    assert [r.call_id for r in port.requests] == ["call_a", "call_b"]
    assert responses(events, "call_a")[-1] == {"result": "removed a"}
    assert responses(events, "call_b")[-1] == {"error": "This tool call is rejected."}
    assert RAN == ["a"]


async def test_foreign_tools_are_confirmed_without_tokens_and_never_edited():
    foreign = FunctionTool(search, require_confirmation=True)
    run = AgentRun([foreign], function_call("search", {"query": "q"}, "call_f"), text("done"))
    paused = await run.turn(user("go"))
    port = RecordingApproval(ScriptedApproval([APPROVE]))
    message = await adk.resolve_confirmations(paused, port, registry_of())
    assert message.parts[0].function_response.response == {"confirmed": True}
    assert port.requests[0].arguments == {"query": "q"}  # raw arguments
    await run.turn(message)
    assert RAN == ["q"]
    edit = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"query": "z"})])
    message = await adk.resolve_confirmations(paused, edit, registry_of())
    assert message.parts[0].function_response.response == {"confirmed": False}


async def test_resolve_skips_answered_and_returns_none_when_nothing_is_open():
    port = ScriptedApproval([APPROVE])
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    run = AgentRun([native], function_call("remove", {"item": "x"}, "call_1"), text("done"))
    paused = await run.turn(user("go"))
    message = await adk.resolve_confirmations(paused, port, registry_of(native.spec))
    await run.turn(message)
    session = await run.sessions.get_session(
        app_name=run.app, user_id="u", session_id=run.session_id
    )
    assert await adk.resolve_confirmations(session.events, port, registry_of(native.spec)) is None


async def test_invalid_arguments_and_a_rule_saying_no_never_pause():
    rule = build_spec(remove, confirm=lambda call, ctx: call.arguments["item"] == "all")
    port = RecordingApproval(ScriptedApproval([]))
    native = adk.adk_tool(rule, executor=Executor(approval=port))
    run = AgentRun([native], function_call("remove", {"item": "one"}, "c1"), text("done"))
    events = await run.turn(user("go"))
    assert confirmations(events) == [] and responses(events, "c1") == [{"result": "removed one"}]
    gated = adk.adk_tool(removal(), executor=Executor(approval=port))
    run = AgentRun([gated], function_call("remove", {"wrong": 1}, "c2"), text("done"))
    events = await run.turn(user("go"))
    assert confirmations(events) == []
    assert responses(events, "c2")[0]["error"]["kind"] == "tool-args-invalid"
    assert port.requests == [] and RAN == ["one"]


async def test_bind_context_maps_the_adk_ids_and_state():
    seen: list[ToolContext] = []

    async def peek(ctx: ToolContext) -> str:
        """Peek at the context."""
        seen.append(ctx)
        return "ok"

    native = adk.adk_tool(build_spec(peek), executor=Executor())
    run = AgentRun([native], function_call("peek", {}, "call_p"), text("done"))
    events = await run.turn(user("go"))
    (ctx,) = seen
    assert ctx.run_id == events[0].invocation_id and ctx.call_id == "call_p"
    assert ctx.thread_id == run.session_id and dict(ctx.state) == {}
    plain = adk.bind_context(object())
    assert (plain.run_id, plain.call_id, plain.thread_id) == ("", "", "")


async def test_results_reach_adk_as_json():
    native = adk.adk_tool(build_spec(hit), executor=Executor())
    assert await run_tool(native, {"source": "s"}, "c1") == {"source": "s"}


def test_targets_and_the_version_guard(monkeypatch):
    with pytest.raises(ToolUsageError):
        adk.adk_tool(build_spec(remove, targets="openai"))
    exported = adk.adk_tools(registry_of(build_spec(remove), build_spec(search, targets="mcp")))
    assert [t.name for t in exported] == ["remove"]
    from google.adk import version

    _support._check_release.cache_clear()
    monkeypatch.setattr(version, "__version__", "3.0.0")
    with pytest.raises(IntegrationError, match="outside the supported range"):
        adk.adk_tool(build_spec(remove))
    monkeypatch.undo()
    _support._check_release.cache_clear()
    monkeypatch.setattr(_support.AdkIntegration, "requires", ("aa_no_such_module",))
    with pytest.raises(MissingDependency, match="agent-adapter\\[adk\\]"):
        adk.adk_tool(build_spec(remove))


async def test_confirmations_from_two_invocations_are_refused():
    port = ScriptedApproval([APPROVE, APPROVE])
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    first = AgentRun([native], function_call("remove", {"item": "a"}, "call_a"), text("done"))
    second = AgentRun([native], function_call("remove", {"item": "b"}, "call_b"), text("done"))
    events = [*await first.turn(user("go")), *await second.turn(user("go"))]
    with pytest.raises(ToolUsageError, match="one invocation"):
        await adk.resolve_confirmations(events, port, registry_of(native.spec))
    assert port.requests == []


async def test_confirmations_without_an_invocation_id_are_refused():
    port = ScriptedApproval([APPROVE])
    native = adk.adk_tool(removal(), executor=Executor(approval=port))
    run = AgentRun([native], function_call("remove", {"item": "x"}, "call_1"), text("done"))
    events = [e.model_copy(update={"invocation_id": ""}) for e in await run.turn(user("go"))]
    with pytest.raises(ToolUsageError, match="invocation id"):
        await adk.resolve_confirmations(events, port, registry_of(native.spec))
    assert port.requests == []


def test_resolve_confirmations_takes_its_run_from_the_events():
    import inspect

    assert "run_id" not in inspect.signature(adk.resolve_confirmations).parameters
