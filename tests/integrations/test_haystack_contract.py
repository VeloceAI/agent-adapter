"""Contract kit C1-C8 for Haystack, plus the confirmation hook's own edge cases."""

import json
from typing import Any

import pytest

pytest.importorskip("haystack")

from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage, ToolCall

from agent_adapter.core.errors import ApprovalPending, IntegrationError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations import haystack
from agent_adapter.integrations.base import ToolRunner
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


@component
class ScriptedGenerator:
    """A chat generator replying from a script, offline."""

    def __init__(self, replies: list[ChatMessage]) -> None:
        self.replies = list(replies)

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {"replies": [self.replies.pop(0)]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.run(messages, tools, **kwargs)


def calls(*items: tuple[str, str, dict[str, Any]]) -> ChatMessage:
    tool_calls = [ToolCall(tool_name=name, arguments=args, id=cid) for cid, name, args in items]
    return ChatMessage.from_assistant(tool_calls=tool_calls)


def agent(native: Any, replies: list[ChatMessage], hook: Any = None) -> Agent:
    hooks = {"before_tool": [hook]} if hook is not None else None
    return Agent(chat_generator=ScriptedGenerator(replies), tools=[native], hooks=hooks)


def results(output: dict[str, Any]) -> dict[str, Any]:
    """Tool call results by call id."""
    found = {}
    for message in output["messages"]:
        for result in message.tool_call_results:
            found[result.origin.id] = result
    return found


async def run_agent(native: Any, call_id: str, args: dict[str, Any], hook: Any = None) -> Any:
    replies = [calls((call_id, native.name, args)), ChatMessage.from_assistant("done")]
    output = await agent(native, replies, hook).run_async(messages=[ChatMessage.from_user("go")])
    return results(output)[call_id]


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return self.decision


class HaystackHarness:
    """Drives `Agent.run_async` with a scripted generator; C6 goes through the hook."""

    name = "haystack"
    edits_supported = False
    approval = "native"
    native_uses_port = True
    port_asks = 1  # the hook asks; the executor must not ask again

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> ApprovalPort:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return haystack.haystack_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return haystack.haystack_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await run_agent(native, call_id, args)

    def value(self, raw: Any) -> Any:
        try:
            return json.loads(raw.result)
        except (TypeError, ValueError):
            return raw.result

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception) or raw.error:
            return True
        value = self.value(raw)
        return isinstance(value, dict) and set(value) == {"error"}

    async def approve_flow(
        self,
        native: Any,
        args: dict[str, Any],
        decision: ApprovalDecision,
        *,
        port: RecordingApproval | None = None,
    ) -> Any:
        assert port is not None
        self.gate.decision = decision
        registry = Registry()
        registry.add(getattr(native, haystack._SPEC_ATTR))
        hook = haystack.confirmation_hook(port, registry)
        return await run_agent(native, "call_haystack_ok", args, hook)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.parameters

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # Haystack takes the neutral JSON schema as given


globals().update((item.__name__, item) for item in contract_tests(HaystackHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(HaystackHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


async def remove(item: str) -> str:
    """Remove an item."""
    return f"removed {item}"


def registry_with(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


async def test_without_the_hook_a_confirm_tool_asks_the_executor_port():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="nope")])
    runner = FakeRunner(approval=port)
    result = await run_agent(
        haystack.haystack_tool(build_spec(remove, confirm=True), executor=runner),
        "c1",
        {"item": "a"},
    )
    assert json.loads(result.result)["error"]["kind"] == "approval-denied"
    assert [r.call_id for r in port.requests] == ["c1"]


async def test_a_pending_approval_rejects_the_call_with_its_id():
    spec = build_spec(remove, confirm=True)
    hook = haystack.confirmation_hook(
        ScriptedApproval([ApprovalPending("apr_1")]), registry_with(spec)
    )
    result = await run_agent(haystack.haystack_tool(spec), "c2", {"item": "a"}, hook)
    assert result.result == "approval required: apr_1"


async def test_the_hook_passes_unknown_and_unconfirmed_tools_through():
    port = ScriptedApproval([])
    plain = build_spec(remove)
    hook = haystack.confirmation_hook(port, registry_with(plain))
    result = await run_agent(haystack.haystack_tool(plain), "c3", {"item": "b"}, hook)
    assert result.result == "removed b"
    other = build_spec(remove, name="other", confirm=True)
    runner = FakeRunner(approval=ScriptedApproval([ApprovalDecision(approved=True)]))
    hook = haystack.confirmation_hook(port, Registry())
    result = await run_agent(
        haystack.haystack_tool(other, executor=runner), "c4", {"item": "a"}, hook
    )
    assert result.result == "removed a" and port.requests == []


async def test_bind_approval_reads_specs_from_the_agent_tools():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    runner = FakeRunner(approval=ScriptedApproval([]))
    native = haystack.haystack_tool(spec, executor=runner)
    result = await run_agent(native, "c5", {"item": "a"}, haystack.INTEGRATION.bind_approval(port))
    assert result.result == "removed a" and [r.call_id for r in port.requests] == ["c5"]
    assert runner.calls[0].ctx.approval is not None and runner.calls[0].ctx.call_id == "c5"


async def test_identical_parallel_calls_get_their_own_call_ids_and_approvals():
    spec = build_spec(remove, confirm=True)
    approve = ApprovalDecision(approved=True)
    port = RecordingApproval(ScriptedApproval([approve, approve]))
    runner = FakeRunner(approval=ScriptedApproval([]))
    native = haystack.haystack_tool(spec, executor=runner)
    replies = [
        calls(("p1", "remove", {"item": "a"}), ("p2", "remove", {"item": "a"})),
        ChatMessage.from_assistant("done"),
    ]
    hook = haystack.confirmation_hook(port, registry_with(spec))
    output = await agent(native, replies, hook).run_async(messages=[ChatMessage.from_user("go")])
    assert set(results(output)) == {"p1", "p2"}
    assert sorted(inv.ctx.call_id for inv in runner.calls) == ["p1", "p2"]
    assert all(inv.ctx.approval is not None for inv in runner.calls)


def test_the_sync_agent_path_runs_the_hook_and_the_tool():
    spec = build_spec(remove, confirm=True)
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="sync no")])
    native = haystack.haystack_tool(spec, executor=FakeRunner())
    replies = [calls(("s1", "remove", {"item": "a"})), ChatMessage.from_assistant("done")]
    hook = haystack.confirmation_hook(port, registry_with(spec))
    output = agent(native, replies, hook).run(messages=[ChatMessage.from_user("go")])
    assert json.loads(results(output)["s1"].result)["error"]["kind"] == "approval-denied"
    plain = haystack.haystack_tool(build_spec(remove), executor=FakeRunner())
    assert plain.invoke(item="b") == "removed b"


async def test_a_direct_invoke_has_no_state_and_no_call_id():
    runner = FakeRunner()
    tool = haystack.haystack_tool(build_spec(remove), executor=runner)
    assert await tool.invoke_async(item="z") == "removed z"
    assert runner.calls[0].ctx.call_id.startswith("call_")


def test_bind_context_claims_each_call_id_once_and_keeps_claims_bounded():
    spec = build_spec(remove)
    state = State(schema={}, data={"messages": [calls(("x1", "remove", {"item": "a"}))]})
    first = haystack.INTEGRATION.bind_context((state, spec, {"item": "a"}))
    second = haystack.INTEGRATION.bind_context((state, spec, {"item": "a"}))
    assert (first.call_id, second.call_id) == ("x1", "")
    assert first.run_id == second.run_id and first.run_id.startswith("run_")
    state.data["messages"] = [calls(("x2", "remove", {"item": "b"}))]
    haystack.INTEGRATION.bind_context((state, spec, {"item": "b"}))
    assert haystack._SCOPES[state].claimed == {"x2"}
    empty = State(schema={}, data={})
    assert haystack.INTEGRATION.bind_context((empty, spec, {})).call_id == ""
    assert haystack.INTEGRATION.bind_context((None, spec, {})).run_id == ""


async def test_approvals_stay_out_of_the_agent_state():
    """Regression: grants lived in State.data and leaked into outputs and snapshots."""
    spec = build_spec(remove, confirm=True)
    hook = haystack.confirmation_hook(
        ScriptedApproval([ApprovalDecision(approved=True)]), registry_with(spec)
    )
    replies = [calls(("s1", "remove", {"item": "a"})), ChatMessage.from_assistant("done")]
    output = await agent(haystack.haystack_tool(spec), replies, hook).run_async(
        messages=[ChatMessage.from_user("go")]
    )
    assert results(output)["s1"].result == "removed a"
    assert not [key for key in output if "agent_adapter" in key]


def test_a_token_only_approves_a_pending_call_of_its_step():
    spec = build_spec(remove, confirm=True)
    state = State(schema={}, data={"messages": [calls(("p1", "remove", {"item": "a"}))]})
    scope = haystack._scope(state)
    approve = ApprovalDecision(approved=True)
    tokens = scope.tokens_of(object())
    tokens.issue("remove", {"item": "a"}, approve, call_id="gone", run_id=scope.run_id)
    ctx = haystack.INTEGRATION.bind_context((state, spec, {"item": "a"}))
    assert (ctx.call_id, ctx.approval) == ("p1", None)  # the token was for another call
    tokens.issue("remove", {"item": "a"}, approve, call_id="p2", run_id=scope.run_id)
    state.data["messages"] = [calls(("p2", "remove", {"item": "a"}))]
    assert haystack.INTEGRATION.bind_context((state, spec, {"item": "a"})).approval == approve


async def test_tokens_not_taken_in_a_step_are_dropped_at_the_next():
    spec = build_spec(remove, confirm=True)
    state = State(schema={}, data={"messages": [ChatMessage.from_user("go")], "tools": []})
    scope = haystack._scope(state)
    approve = ApprovalDecision(approved=True)
    hook = haystack.confirmation_hook(ScriptedApproval([]), registry_with(spec))
    other = haystack.confirmation_hook(ScriptedApproval([]), registry_with(spec))
    for owner in (hook, other):
        scope.tokens_of(owner).issue(
            "remove", {"item": "a"}, approve, call_id="old", run_id=scope.run_id
        )
    await hook.run_async(state)
    assert len(scope.tokens_of(hook)) == 0 and len(scope.tokens_of(other)) == 1


def test_states_that_cannot_be_tracked_get_no_scope():
    assert haystack._scope({}) is None
    ctx = haystack.INTEGRATION.bind_context(({}, build_spec(remove), {}))
    assert (ctx.run_id, ctx.call_id) == ("", "")


def test_the_hook_and_strategy_refuse_serialisation():
    hook = haystack.confirmation_hook(ScriptedApproval([]), Registry())
    with pytest.raises(IntegrationError, match="live approval port"):
        hook.to_dict()
    with pytest.raises(IntegrationError):
        type(hook).from_dict({})
    strategy = haystack._PortStrategy(ScriptedApproval([]), {}, "run_x", haystack.TokenTable())
    with pytest.raises(IntegrationError):
        strategy.to_dict()
    with pytest.raises(IntegrationError):
        haystack._PortStrategy.from_dict({})
    decision = strategy.run(tool_name="x", tool_description="", tool_params={})
    assert decision.execute is True


def test_the_example_runs(capsys):
    rate, sent, denied = _run("haystack_tools.py", capsys).splitlines()
    assert (rate, sent) == ("5.0", "invoiced cus_1024 for 49.50 USD")
    assert json.loads(denied)["error"]["kind"] == "approval-denied"


async def test_two_confirmation_hooks_keep_their_own_tokens():
    """Regression: the second hook's step-start clear wiped the first hook's tokens."""
    ran: list[str] = []

    async def wipe(item: str) -> str:
        """Wipe an item."""
        ran.append(item)
        return f"wiped {item}"

    spec, other = build_spec(wipe, confirm=True), build_spec(remove, name="other", confirm=True)
    first = haystack.confirmation_hook(
        ScriptedApproval([ApprovalDecision(approved=True)]), registry_with(spec)
    )
    second = haystack.confirmation_hook(ScriptedApproval([]), registry_with(other))
    replies = [calls(("w1", "wipe", {"item": "a"})), ChatMessage.from_assistant("done")]
    native = haystack.haystack_tool(spec, executor=FakeRunner())  # no port: needs the token
    output = await Agent(
        chat_generator=ScriptedGenerator(replies),
        tools=[native],
        hooks={"before_tool": [first, second]},
    ).run_async(messages=[ChatMessage.from_user("go")])
    assert results(output)["w1"].result == "wiped a" and ran == ["a"]
