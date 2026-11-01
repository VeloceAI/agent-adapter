"""Contract kit C1-C8 for the Claude Agent SDK, plus its bridge, gates and SDK MCP server."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

import mcp
from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claude_agent_sdk.types import CanUseToolShadowedWarning

from agent_adapter.core.errors import ApprovalPending, IntegrationError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations import claude_agent
from agent_adapter.integrations.base import ToolRunner
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations._claude_cli import (
    TOOL,
    MessagesApi,
    Reviewer,
    call_server,
    cli_setup,
    dispatch,
    needs_cli,
    run_cli,
    text_of,
)
from tests.integrations._claude_cli import api as api  # the scripted Messages API fixture
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return self.decision


def permission_context(call_id: str) -> ToolPermissionContext:
    return ToolPermissionContext(tool_use_id=call_id)


def registry_of(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


class ClaudeAgentHarness:
    """Calls a bridge's MCP `tools/call` handler with Claude Code's tool-use id; C6 via its gate."""

    name = "claude_agent"
    edits_supported = True  # PermissionResultAllow(updated_input=…)
    approval = "native"
    native_uses_port = True
    port_asks = 1  # the bridge's can_use_tool asks; its token spares the executor's ask

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> ApprovalPort:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        bridge = claude_agent.claude_sdk_bridge(registry_of(spec), self.gate, executor=executor)
        tool = claude_agent.claude_sdk_tool(spec, executor=executor)
        return SimpleNamespace(spec=spec, executor=executor, bridge=bridge, tool=tool)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return claude_agent.claude_sdk_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await dispatch(native.bridge, native.spec.name, args, call_id)

    def value(self, raw: Any) -> Any:
        return text_of(raw)

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or bool(raw.get("is_error"))

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
        spec, runner = native.spec, native.executor
        bridge = claude_agent.claude_sdk_bridge(registry_of(spec), port, executor=runner)
        answer = await bridge.can_use_tool(
            f"mcp__agent_adapter__{spec.name}", args, permission_context("tu_1")
        )
        if isinstance(answer, PermissionResultDeny):
            return {"content": [{"type": "text", "text": answer.message}], "is_error": True}
        return await call_server(bridge.server, spec.name, answer.updated_input or args, "tu_1")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        tool = getattr(native, "tool", native)  # a bridge export, or an `SdkMcpTool`
        return tool.name, tool.description, tool.input_schema

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # the SDK takes the neutral JSON schema as given


globals().update((item.__name__, item) for item in contract_tests(ClaudeAgentHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(ClaudeAgentHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


async def remove(item: str) -> str:
    """Remove an item."""
    return f"removed {item}"


async def lookup(term: str) -> dict[str, Any]:
    """Look a term up."""
    return {"term": term}


def deny() -> ApprovalDecision:
    return ApprovalDecision(approved=False, reason="executor")


def deny_all(count: int = 3) -> ScriptedApproval:
    return ScriptedApproval([ApprovalDecision(approved=False, reason="executor")] * count)


async def test_the_server_serves_the_registry_over_mcp():
    registry = registry_of(build_spec(lookup), build_spec(remove, targets="elsewhere"))
    config = claude_agent.claude_sdk_server(registry, name="tools")
    assert (config["type"], config["name"]) == ("sdk", "tools")
    async with mcp.Client(config["instance"]) as client:
        listed = await client.list_tools()
        assert [t.name for t in listed.tools] == ["lookup"]
        ok = await client.call_tool("lookup", {"term": "gst"})
        assert not ok.is_error and json.loads(ok.content[0].text) == {"term": "gst"}
        assert (await client.call_tool("lookup", {})).is_error


async def test_a_bridge_approval_with_edits_runs_once_through_its_server():
    spec = build_spec(remove, confirm=True)
    edit = ApprovalDecision(approved=True, edited_args={"item": "beta"})
    port = RecordingApproval(ScriptedApproval([edit]))
    runner = FakeRunner(approval=ScriptedApproval([]))  # asking again would fail
    bridge = claude_agent.claude_sdk_bridge(registry_of(spec), port, executor=runner, name="tools")
    answer = await bridge.can_use_tool(
        "mcp__tools__remove", {"item": "alpha"}, permission_context("tu_9")
    )
    assert isinstance(answer, PermissionResultAllow) and answer.updated_input == {"item": "beta"}
    (request,) = port.requests
    assert (request.tool, request.arguments, request.call_id) == (
        "remove",
        {"item": "alpha"},
        "tu_9",
    )
    assert request.run_id == bridge.run_id
    assert text_of(await call_server(bridge.server, "remove", {"item": "beta"}, "tu_9")) == (
        "removed beta"
    )
    (inv,) = runner.calls
    assert (inv.ctx.call_id, inv.ctx.run_id) == ("tu_9", bridge.run_id)
    assert inv.ctx.approval is not None and inv.ctx.approval.edited_args is None
    replay = await call_server(bridge.server, "remove", {"item": "beta"}, "tu_9")
    assert replay["is_error"]  # the token was used: the executor's (empty) port is asked


async def test_a_token_only_covers_its_input_and_its_bridge():
    """Regression: grants were module-global, keyed by bare tool name and arguments."""
    spec = build_spec(remove, confirm=True)
    approve = ScriptedApproval([ApprovalDecision(approved=True)])
    first = claude_agent.claude_sdk_bridge(
        registry_of(spec), approve, executor=FakeRunner(approval=deny_all())
    )
    other_runner = FakeRunner(approval=deny_all())
    second = claude_agent.claude_sdk_bridge(registry_of(spec), deny_all(), executor=other_runner)
    await first.can_use_tool("mcp__agent_adapter__remove", {"item": "a"}, permission_context("t1"))
    assert (await call_server(second.server, "remove", {"item": "a"}, "t1"))["is_error"]
    assert other_runner.calls == []
    assert (await call_server(first.server, "remove", {"item": "other"}, "t1"))["is_error"]
    assert (await call_server(first.server, "remove", {"item": "a"}, "t2"))["is_error"]
    assert (await call_server(first.server, "remove", {"item": "a"}))["is_error"]  # no id
    assert text_of(await call_server(first.server, "remove", {"item": "a"}, "t1")) == "removed a"


async def test_other_servers_and_tools_never_grant_our_tools():
    """Regression: `mcp__evil__remove` mapped onto the registry and was allowed unasked."""
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(
        ScriptedApproval(
            [
                ApprovalDecision(approved=True),
                ApprovalDecision(approved=True, edited_args={"command": "ls -la"}),
                ApprovalPending("apr_2"),
                ApprovalDecision(approved=False, reason="no"),
            ]
        )
    )
    runner = FakeRunner(approval=deny_all())
    bridge = claude_agent.claude_sdk_bridge(registry_of(spec), port, executor=runner)
    evil = await bridge.can_use_tool("mcp__evil__remove", {"item": "a"}, permission_context("e1"))
    assert isinstance(evil, PermissionResultAllow)  # the port said yes, under its real name
    bash = await bridge.can_use_tool("Bash", {"command": "ls"}, permission_context("b1"))
    assert bash.updated_input == {"command": "ls -la"}
    pending = await bridge.can_use_tool("mcp__github__drop", {}, permission_context("b2"))
    assert pending.message == "approval required: apr_2"
    denied = await bridge.can_use_tool("Read", {}, permission_context("b3"))
    assert json.loads(denied.message)["error"]["kind"] == "approval-denied"
    assert [r.tool for r in port.requests] == [
        "mcp__evil__remove",
        "Bash",
        "mcp__github__drop",
        "Read",
    ]
    # a stale approval of another server's tool does not run ours
    assert (await call_server(bridge.server, "remove", {"item": "a"}))["is_error"]
    assert runner.calls == []


async def test_rules_that_need_no_approval_allow_without_asking():
    rule = build_spec(remove, confirm=lambda call, ctx: call.arguments["item"] == "danger")
    port = ScriptedApproval([])
    bridge = claude_agent.claude_sdk_bridge(registry_of(rule), port)
    answer = await bridge.can_use_tool(
        "mcp__agent_adapter__remove", {"item": "safe"}, permission_context("t")
    )
    assert isinstance(answer, PermissionResultAllow) and answer.updated_input is None
    assert port.requests == []


async def test_pending_denied_and_id_less_answers():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(
        ScriptedApproval([ApprovalPending("apr_7"), ApprovalDecision(approved=False, reason="no")])
    )
    bridge = claude_agent.claude_sdk_bridge(registry_of(spec), port)
    name = "mcp__agent_adapter__remove"
    pending = await bridge.can_use_tool(name, {"item": "a"}, permission_context("t1"))
    assert isinstance(pending, PermissionResultDeny)
    assert pending.message == "approval required: apr_7"
    denied = await bridge.can_use_tool(name, {"item": "a"}, permission_context("t2"))
    assert json.loads(denied.message)["error"]["kind"] == "approval-denied"
    no_id = await bridge.can_use_tool(name, {"item": "a"}, ToolPermissionContext())
    assert "no id" in json.loads(no_id.message)["error"]["message"] and len(port.requests) == 2


async def test_an_edit_that_fails_validation_gets_no_token():
    spec = build_spec(remove, confirm=True)
    edit = ApprovalDecision(approved=True, edited_args={"wrong": 1})
    runner = FakeRunner(approval=deny_all())
    bridge = claude_agent.claude_sdk_bridge(
        registry_of(spec), ScriptedApproval([edit]), executor=runner
    )
    answer = await bridge.can_use_tool(
        "mcp__agent_adapter__remove", {"item": "a"}, permission_context("t")
    )
    assert answer.updated_input == {"wrong": 1}
    assert (await call_server(bridge.server, "remove", {"wrong": 1}))["is_error"]


async def test_permission_only_gates_never_approve_the_tool():
    spec = build_spec(remove, confirm=True)
    executor_port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    native = claude_agent.claude_sdk_tool(spec, executor=FakeRunner(approval=executor_port))
    for gate in (
        claude_agent.can_use_tool(
            ScriptedApproval([ApprovalDecision(approved=True)]), registry_of(spec)
        ),
        claude_agent.INTEGRATION.bind_approval(ScriptedApproval([ApprovalDecision(approved=True)])),
    ):
        answer = await gate("mcp__agent_adapter__remove", {"item": "a"}, permission_context("t"))
        assert isinstance(answer, PermissionResultAllow)
    assert text_of(await native.handler({"item": "a"})) == "removed a"
    assert len(executor_port.requests) == 1  # the tool still asked its own port


def test_registry_name_maps_only_its_own_server():
    registry = registry_of(build_spec(remove), build_spec(lookup, name="a__b"))
    assert (
        claude_agent._registry_name("mcp__agent_adapter__remove", registry, "agent_adapter")
        == "remove"
    )
    assert claude_agent._registry_name("mcp__srv__a__b", registry, "srv") == "a__b"
    assert claude_agent._registry_name("mcp__evil__remove", registry, "agent_adapter") == ""
    assert claude_agent._registry_name("remove", registry, "agent_adapter") == ""
    assert (
        claude_agent._registry_name("mcp__agent_adapter__missing", registry, "agent_adapter") == ""
    )


async def test_non_json_results_are_sent_as_text():
    async def opaque() -> object:
        """Return an opaque object."""
        return {1, 2}

    result = await claude_agent.claude_sdk_tool(build_spec(opaque)).handler({})
    assert result == {"content": [{"type": "text", "text": "{1, 2}"}]}


def test_the_example_runs(capsys):
    assert _run("claude_agent_tools.py", capsys).splitlines() == [
        "PermissionResultAllow {'customer_id': 'cus_1024', 'amount': 49.5}",
        "18.0 invoiced cus_1024 for 49.50 USD",
    ]


# --- the full Claude Code loop, offline: the bundled CLI against a scripted Messages API ----


@needs_cli
async def test_cli_loop_the_bridge_gate_approves_once_and_the_tool_runs(api, tmp_path):
    reviewer = Reviewer()
    bridge, ran, runner = cli_setup(reviewer)
    await run_cli(api, tmp_path, bridge)
    assert ran == ["a"]
    assert [(r.tool, r.call_id) for r in reviewer.requests] == [("remove", "toolu_01")]
    assert runner.calls[0].ctx.call_id == "toolu_01"


@needs_cli
async def test_cli_loop_a_stale_other_server_approval_does_not_run_our_tool(api, tmp_path):
    reviewer = Reviewer()
    bridge, ran, _ = cli_setup(reviewer)
    stale = await bridge.can_use_tool(
        "mcp__other__remove", {"item": "a"}, ToolPermissionContext(tool_use_id="old")
    )
    assert stale.behavior == "allow"
    # allowed_tools skips can_use_tool: only a token of this bridge could approve the call
    with pytest.warns(CanUseToolShadowedWarning):
        await run_cli(api, tmp_path, bridge, allowed_tools=[TOOL])
    assert ran == []
    assert [r.tool for r in reviewer.requests] == ["mcp__other__remove"]


@needs_cli
async def test_cli_loop_an_unexecuted_approval_does_not_survive_the_turn(api, tmp_path):
    """Regression: an approved call that never ran left a token for a later identical call."""
    reviewer = Reviewer()
    bridge, ran, _ = cli_setup(reviewer)
    unused = await bridge.can_use_tool(TOOL, {"item": "a"}, ToolPermissionContext(tool_use_id="t0"))
    assert unused.behavior == "allow"
    # turn 1 says hello (the Stop hook ends it); turn 2 calls the tool via allowed_tools
    with pytest.warns(CanUseToolShadowedWarning):
        await run_cli(api, tmp_path, bridge, prompts=("hello", "remove a"), allowed_tools=[TOOL])
    assert ran == []


async def test_the_bridge_hooks_drop_its_tokens_at_turn_boundaries():
    spec = build_spec(remove, confirm=True)
    for event in ("Stop", "UserPromptSubmit"):
        approve = ScriptedApproval([ApprovalDecision(approved=True)])
        runner = FakeRunner(approval=deny_all())
        bridge = claude_agent.claude_sdk_bridge(registry_of(spec), approve, executor=runner)
        await bridge.can_use_tool(TOOL, {"item": "a"}, permission_context("t1"))
        (matcher,) = bridge.hooks[event]
        (hook,) = matcher.hooks
        assert await hook({"hook_event_name": event}, None, {"signal": None}) == {}
        assert (await call_server(bridge.server, "remove", {"item": "a"}, "t1"))["is_error"]
        assert runner.calls == []


async def test_each_bridge_is_its_own_run():
    """A bridge per session keeps run guards per session (one shared bridge spanned them)."""
    executor = Executor()
    spec = build_spec(lookup, once_per_run=True)
    first, second = (
        claude_agent.claude_sdk_bridge(registry_of(spec), deny_all(), executor=executor)
        for _ in range(2)
    )
    assert first.run_id != second.run_id
    for bridge in (first, second):
        assert text_of(await call_server(bridge.server, "lookup", {"term": "x"})) == {"term": "x"}
    again = await call_server(first.server, "lookup", {"term": "x"})
    assert again["is_error"] or "run-limit" in json.dumps(again)


@needs_cli
async def test_cli_loop_two_identical_calls_in_one_turn_keep_their_own_approval(
    api, tmp_path, monkeypatch
):
    """Regression: toolu_B (let through by a host PreToolUse hook) took toolu_A's token."""
    from claude_agent_sdk import HookMatcher

    monkeypatch.setattr(
        MessagesApi, "calls", [("toolu_A", {"item": "a"}), ("toolu_B", {"item": "a"})]
    )
    reviewer = Reviewer()
    bridge, ran, runner = cli_setup(reviewer, executor_port=ScriptedApproval([deny()]))

    async def host_allows_b(event: Any, tool_use_id: str | None, context: Any) -> Any:
        if event.get("tool_use_id") != "toolu_B":
            return {}
        output = {"hookEventName": "PreToolUse", "permissionDecision": "allow"}
        return {"hookSpecificOutput": output}

    hooks = {
        event: bridge.hooks.get(event, []) + mine
        for event, mine in {"PreToolUse": [HookMatcher(hooks=[host_allows_b])]}.items()
    } | {event: matchers for event, matchers in bridge.hooks.items() if event != "PreToolUse"}
    await run_cli(api, tmp_path, bridge, hooks=hooks)
    assert [r.call_id for r in reviewer.requests] == ["toolu_A"]  # the gate saw A only
    assert ran == ["a"]  # A ran with its token
    assert [r.call_id for r in runner.approval.requests] == ["toolu_B"]  # B was asked


@needs_cli
async def test_cli_loop_a_stale_token_without_hooks_is_bound_to_its_call(api, tmp_path):
    """Regression: without the bridge hooks, an earlier approval served a later identical call."""
    reviewer = Reviewer()
    bridge, ran, _ = cli_setup(reviewer)
    earlier = await bridge.can_use_tool(
        TOOL, {"item": "a"}, ToolPermissionContext(tool_use_id="t0")
    )
    assert earlier.behavior == "allow"
    with pytest.warns(CanUseToolShadowedWarning):
        await run_cli(api, tmp_path, bridge, hooks={}, allowed_tools=[TOOL])
    assert ran == []


async def test_the_bridge_lists_what_the_sdk_tool_declares():
    spec = build_spec(remove, confirm=True)
    bridge = claude_agent.claude_sdk_bridge(registry_of(spec), deny_all())
    entry = bridge.server["instance"].get_request_handler("tools/list")
    (tool,) = (await entry.handler(None, None)).tools
    sdk_tool = claude_agent.claude_sdk_tool(spec)
    assert (tool.name, tool.description, tool.input_schema) == (
        sdk_tool.name,
        sdk_tool.description,
        sdk_tool.input_schema,
    )
    assert (bridge.server["type"], bridge.server["name"]) == ("sdk", "agent_adapter")
    unknown = await dispatch(bridge, "missing", {}, "t1")
    assert unknown["is_error"] and "missing" in unknown["content"][0]["text"]


def test_an_mcp_before_2_is_refused(monkeypatch):
    import mcp.types

    monkeypatch.delattr(mcp.types, "jsonrpc_message_adapter")
    claude_agent._check_frameworks.cache_clear()
    try:
        with pytest.raises(IntegrationError, match="mcp"):
            claude_agent.claude_sdk_bridge(registry_of(build_spec(remove)), deny_all())
    finally:
        monkeypatch.undo()
        claude_agent._check_frameworks.cache_clear()
