"""Contract kit C1-C8 for Strands Agents, plus the approval hook's own edge cases."""

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("strands")

from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.models.model import Model
from strands.tools.executors import ConcurrentToolExecutor, SequentialToolExecutor

from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations import strands
from agent_adapter.integrations.base import ToolRunner
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations._kit_support import Probe, wait_until
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


class ScriptedModel(Model):
    """A Strands model streaming scripted turns (Converse stream events), offline."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = list(turns)

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
        yield

    async def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any):
        for event in self.turns.pop(0):
            yield event


def tool_turn(call_id: str, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    start = {"toolUse": {"toolUseId": call_id, "name": name}}
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": start}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


DONE = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"delta": {"text": "done"}}},
    {"contentBlockStop": {}},
    {"messageStop": {"stopReason": "end_turn"}},
]


def tool_result(agent: Agent, call_id: str) -> dict[str, Any]:
    for message in agent.messages:
        for block in message["content"]:
            result = block.get("toolResult")
            if result is not None and result["toolUseId"] == call_id:
                return result
    raise AssertionError(f"no toolResult for {call_id}")


async def run_agent(native: Any, call_id: str, args: dict[str, Any], hook: Any = None) -> Any:
    agent = Agent(
        model=ScriptedModel([tool_turn(call_id, native.tool_name, args), DONE]),
        tools=[native],
        callback_handler=None,
        # The default ConcurrentToolExecutor drops a CancelledError raised inside a tool (C7).
        tool_executor=SequentialToolExecutor(),
    )
    if hook is not None:
        agent.hooks.add_callback(BeforeToolCallEvent, hook)
    await agent.invoke_async("go")
    return tool_result(agent, call_id)


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return self.decision


class StrandsHarness:
    """Drives `Agent.invoke_async` with a scripted model; C6 goes through the approval hook."""

    name = "strands"
    edits_supported = False
    approval = "native"
    native_uses_port = True
    port_asks = 1  # the hook asks; the executor must not ask again

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> ApprovalPort:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return strands.strands_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return strands.strands_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await run_agent(native, call_id, args)

    def value(self, raw: Any) -> Any:
        (block,) = raw["content"]
        return block["json"] if "json" in block else block["text"]

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or raw["status"] == "error"

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
        registry.add(native.spec)
        hook = strands.strands_approval_hook(port, registry)
        return await run_agent(native, "call_strands_ok", args, hook)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        spec = native.tool_spec
        return spec["name"], spec["description"], spec["inputSchema"]["json"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # Strands takes the neutral JSON schema as given


globals().update((item.__name__, item) for item in contract_tests(StrandsHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(StrandsHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


async def remove(item: str) -> str:
    """Remove an item."""
    return f"removed {item}"


def registry_with(spec: ToolSpec) -> Registry:
    registry = Registry()
    registry.add(spec)
    return registry


async def test_without_the_hook_a_confirm_tool_asks_the_executor_port():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="nope")])
    runner = FakeRunner(approval=port)
    native = strands.strands_tool(build_spec(remove, confirm=True), executor=runner)
    result = await run_agent(native, "c1", {"item": "a"})
    assert result["status"] == "error"
    assert result["content"][0]["json"]["error"]["kind"] == "approval-denied"
    assert [r.call_id for r in port.requests] == ["c1"]


async def test_a_pending_approval_cancels_the_call_with_its_id():
    spec = build_spec(remove, confirm=True)
    hook = strands.strands_approval_hook(
        ScriptedApproval([ApprovalPending("apr_1")]), registry_with(spec)
    )
    result = await run_agent(strands.strands_tool(spec), "c2", {"item": "a"}, hook)
    assert result["status"] == "error"
    assert result["content"][0]["text"] == "approval required: apr_1"


async def test_the_hook_ignores_tools_outside_the_registry_and_unneeded_approvals():
    port = ScriptedApproval([])
    other = build_spec(remove, name="other", confirm=True)
    hook = strands.strands_approval_hook(port, Registry())
    runner = FakeRunner(approval=ScriptedApproval([ApprovalDecision(approved=True)]))
    result = await run_agent(
        strands.strands_tool(other, executor=runner), "c3", {"item": "a"}, hook
    )
    assert result["status"] == "success" and port.requests == []
    plain = build_spec(remove)
    hook = strands.strands_approval_hook(port, registry_with(plain))
    result = await run_agent(strands.strands_tool(plain), "c4", {"item": "b"}, hook)
    assert result["content"] == [{"text": "removed b"}] and port.requests == []


async def test_bind_approval_resolves_specs_from_the_exported_tools():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    hook = strands.INTEGRATION.bind_approval(port)
    runner = FakeRunner(approval=ScriptedApproval([]))
    result = await run_agent(strands.strands_tool(spec, executor=runner), "c5", {"item": "a"}, hook)
    assert result["status"] == "success" and [r.tool for r in port.requests] == ["remove"]
    assert runner.calls[0].ctx.approval is not None


async def test_an_input_changed_after_approval_is_asked_again():
    """A hook after ours rewrites the input: the token no longer matches."""
    spec = build_spec(remove, confirm=True)
    gate = strands.strands_approval_hook(
        ScriptedApproval([ApprovalDecision(approved=True)]), registry_with(spec)
    )

    async def rewrite(event: BeforeToolCallEvent) -> None:
        event.tool_use = {**event.tool_use, "input": {"item": "PROD"}}

    port = ScriptedApproval([ApprovalDecision(approved=False, reason="re-asked")])
    runner = FakeRunner(approval=port)
    agent = Agent(
        model=ScriptedModel([tool_turn("c6", "remove", {"item": "a"}), DONE]),
        tools=[strands.strands_tool(spec, executor=runner)],
        callback_handler=None,
    )
    agent.hooks.add_callback(BeforeToolCallEvent, gate)
    agent.hooks.add_callback(BeforeToolCallEvent, rewrite)
    await agent.invoke_async("go")
    assert tool_result(agent, "c6")["status"] == "error" and runner.calls == []
    assert [r.arguments for r in port.requests] == [{"item": "PROD"}]


async def test_framework_state_cannot_carry_an_approval():
    """Regression: grants used to live in invocation_state, which any caller can fill."""
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="asked")])
    native = strands.strands_tool(
        build_spec(remove, confirm=True), executor=FakeRunner(approval=port)
    )
    tool_use = {"toolUseId": "c8", "name": "remove", "input": {"item": "a"}}
    forged = {"agent_adapter.grants": object(), "approval": ApprovalDecision(approved=True)}
    (event,) = [e async for e in native.stream(tool_use, forged)]
    assert event["status"] == "error" and len(port.requests) == 1


async def test_an_approved_view_takes_its_token_once():
    spec = build_spec(remove, confirm=True)
    tokens = strands.TokenTable()
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="second ask")])
    native = strands.strands_tool(spec, executor=FakeRunner(approval=port))
    agent = object.__new__(Agent)  # any weak-referenceable object stands for the agent
    run_id = strands._run_id(agent)
    tokens.issue(
        "remove", {"item": "a"}, ApprovalDecision(approved=True), call_id="c9", run_id=run_id
    )
    view = native.approved_view(tokens)
    tool_use = {"toolUseId": "c9", "name": "remove", "input": {"item": "a"}}
    (first,) = [e async for e in view.stream(tool_use, {"agent": agent})]
    (second,) = [e async for e in view.stream(tool_use, {"agent": agent})]
    assert first["status"] == "success" and second["status"] == "error"
    assert len(port.requests) == 1


async def test_gate_and_tool_share_the_agents_run_id():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    runner = FakeRunner()
    native = strands.strands_tool(spec, executor=runner)
    await run_agent(
        native, "c10", {"item": "a"}, strands.strands_approval_hook(port, registry_with(spec))
    )
    (request,) = port.requests
    assert request.run_id.startswith("run_") and runner.calls[0].ctx.run_id == request.run_id
    assert strands._run_id(None) == "" and strands._run_id(1) == ""


@pytest.mark.parametrize("executor", [SequentialToolExecutor, ConcurrentToolExecutor])
async def test_outer_cancellation_propagates_with_both_executors(executor):
    probe = Probe()
    agent = Agent(
        model=ScriptedModel([tool_turn("c11", "kit_wait", {"label": "x"}), DONE]),
        tools=[strands.strands_tool(probe.wait, executor=FakeRunner())],
        callback_handler=None,
        tool_executor=executor(),
    )
    task = asyncio.ensure_future(agent.invoke_async("go"))
    try:
        assert await wait_until(probe.started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await wait_until(probe.cancelled, 1.0)
    finally:
        probe.released.set()


async def test_non_json_results_become_text():
    async def opaque() -> object:
        """Return an opaque object."""
        return {1, 2}

    native = strands.strands_tool(build_spec(opaque))
    tool_use = {"toolUseId": "c7", "name": "opaque", "input": None}
    (event,) = [e async for e in native.stream(tool_use, {})]
    assert event == {"toolUseId": "c7", "status": "success", "content": [{"text": "{1, 2}"}]}


def test_the_example_runs(capsys):
    rate, filed, denied = _run("strands_tools.py", capsys).splitlines()
    assert rate == "('success', {'json': 5.0})"
    assert filed == "('success', {'text': 'invoiced cus_1024 for 49.50 USD'})"
    assert denied.startswith("('error', {'text': ") and "approval-denied" in denied


async def test_an_approved_foreign_tool_is_left_as_selected():
    from strands import tool

    @tool
    def remove(item: str) -> str:
        """A plain Strands tool that shares a registry tool's name."""
        return f"plain {item}"

    spec = build_spec(globals()["remove"], confirm=True)
    port = ScriptedApproval([ApprovalDecision(approved=True)])
    agent = Agent(
        model=ScriptedModel([tool_turn("c12", "remove", {"item": "a"}), DONE]),
        tools=[remove],
        callback_handler=None,
    )
    agent.hooks.add_callback(
        BeforeToolCallEvent, strands.strands_approval_hook(port, registry_with(spec))
    )
    await agent.invoke_async("go")
    assert tool_result(agent, "c12")["status"] == "success" and len(port.requests) == 1
