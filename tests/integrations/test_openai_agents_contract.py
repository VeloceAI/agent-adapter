"""Contract kit C1-C8 for the OpenAI Agents SDK, through `Runner.run` with a scripted model."""

from typing import Any

import pytest

pytest.importorskip("agents")

from agent_adapter.approval import CallbackApproval
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, MissingDependency, ToolUsageError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.openai_agents import (
    _TOKENS,
    function_tool,
    function_tools,
    resolve_interruptions,
)
from agent_adapter.schema.transforms import to_openai_parameters
from tests.fakes.approval import ScriptedApproval
from tests.integrations import openai_agents_cases
from tests.integrations.kit import contract_tests
from tests.integrations.openai_agents_support import (
    agent_with,
    decoded,
    is_error_output,
    output_of,
    run,
)

pytestmark = pytest.mark.contract


class AgentsHarness:
    """Drives `FunctionTool`s through the Agents SDK run loop and its approval gate."""

    name = "openai_agents"
    edits_supported = True  # edits travel in the approval token
    approval = "native"
    native_uses_port = True  # resolve_interruptions asks the port
    port_asks = 1

    def __init__(self) -> None:
        self.decision: ApprovalDecision | None = None

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return function_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return function_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        result = await run(agent_with([native], [(native.name, args, call_id)]))
        assert not result.interruptions, "an unexpected approval interruption"
        return output_of(result, call_id)

    def value(self, raw: Any) -> Any:
        return decoded(raw)

    def is_error(self, raw: Any) -> bool:
        return is_error_output(raw)

    def approval_port(self) -> Any:
        return CallbackApproval(lambda request: self.decision)

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision, *, port: Any = None
    ) -> Any:
        self.decision = decision
        agent = agent_with([native], [(native.name, args, "call_kit_c6")])
        first = await run(agent)
        assert [item.tool_name for item in first.interruptions] == [native.name]
        state = await resolve_interruptions(first, port)
        return output_of(await run(agent, state), "call_kit_c6")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.params_json_schema

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_openai_parameters(spec.json_schema(), strict=True)


globals().update((item.__name__, item) for item in contract_tests(AgentsHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AgentsHarness, runner_factory=Executor)
)


# --- bridge behaviour beyond the kit ---------------------------------------------------------


class Tools:
    """A registry whose tools record what actually ran."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []
        registry = Registry()

        @registry.tool(confirm=lambda call, ctx: call.arguments["amount"] > 100)
        async def pay(amount: int, to: str = "acme") -> str:
            """Pay an amount (large payments need approval)."""
            self.ran.append(("pay", {"amount": amount, "to": to}))
            return f"paid {amount} to {to}"

        @registry.tool(confirm=True)
        async def wipe(path: str) -> dict[str, Any]:
            """Delete a path."""
            self.ran.append(("wipe", {"path": path}))
            return {"wiped": path}

        @registry.tool
        async def echo(text: str) -> str:
            """Echo the text."""
            self.ran.append(("echo", {"text": text}))
            return text

        self.registry = registry

    def export(self, **kwargs: Any) -> dict[str, Any]:
        return {tool.name: tool for tool in function_tools(self.registry, **kwargs)}


async def test_callable_confirm_asks_only_when_the_rule_says_so():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    small = await run(agent_with([pay], [("pay", {"amount": 5, "to": None}, "c1")]))
    assert not small.interruptions and output_of(small, "c1") == "paid 5 to acme"
    big = await run(agent_with([pay], [("pay", {"amount": 500, "to": "bob"}, "c2")]))
    assert [item.call_id for item in big.interruptions] == ["c2"]
    assert tools.ran == [("pay", {"amount": 5, "to": "acme"})]


async def test_callable_confirm_skips_approval_for_invalid_args():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    result = await run(agent_with([pay], [("pay", {"amount": "lots"}, "c1")]))
    assert not result.interruptions
    assert decoded(output_of(result, "c1"))["error"]["kind"] == "tool-args-invalid"
    assert tools.ran == []


async def test_approve_reject_and_edit_several_calls_in_one_pass():
    tools = Tools()
    exported = tools.export(executor=Executor())
    calls = [
        ("wipe", {"path": "/a"}, "w1"),
        ("wipe", {"path": "/b"}, "w2"),
        ("pay", {"amount": 900, "to": "eve"}, "p1"),
    ]
    agent = agent_with(list(exported.values()), calls)
    first = await run(agent)
    _TOKENS.clear()  # the table is module-wide: start from empty
    port = ScriptedApproval(
        [
            ApprovalDecision(approved=True),
            ApprovalDecision(approved=False, reason="not /b"),
            ApprovalDecision(approved=True, edited_args={"amount": 90, "to": "eve"}),
        ]
    )
    state = await resolve_interruptions(first, port)
    assert [(r.tool, r.call_id, r.arguments) for r in port.requests] == [
        ("wipe", "w1", {"path": "/a"}),
        ("wipe", "w2", {"path": "/b"}),
        ("pay", "p1", {"amount": 900, "to": "eve"}),
    ]
    assert all(r.metadata == {"integration": "openai_agents"} for r in port.requests)
    second = await run(agent, state)
    assert sorted(tools.ran, key=str) == [
        ("pay", {"amount": 90, "to": "eve"}),
        ("wipe", {"path": "/a"}),
    ]
    assert decoded(output_of(second, "w1")) == {"wiped": "/a"}
    denied = decoded(output_of(second, "w2"))["error"]
    assert denied["kind"] == "approval-denied" and "not /b" in denied["message"]
    assert len(_TOKENS) == 0  # both tokens were used up


async def test_an_edit_reaches_the_tool_without_any_side_table():
    tools = Tools()
    wipe = tools.export(executor=Executor())["wipe"]  # no port: only the token approves
    agent = agent_with([wipe], [("wipe", {"path": "/a"}, "w1")])
    first = await run(agent)
    port = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"path": "/tmp"})])
    state = await resolve_interruptions(first, port)
    assert decoded(output_of(await run(agent, state), "w1")) == {"wiped": "/tmp"}
    assert tools.ran == [("wipe", {"path": "/tmp"})]


async def test_pending_decisions_are_raised_after_asking_every_call():
    tools = Tools()
    wipe = tools.export(executor=Executor())["wipe"]
    calls = [("wipe", {"path": "/a"}, "w1"), ("wipe", {"path": "/b"}, "w2")]
    agent = agent_with([wipe], calls)
    state = (await run(agent)).to_state()
    port = ScriptedApproval(
        [ApprovalPending("apr_1", tool="wipe"), ApprovalDecision(approved=True)]
    )
    with pytest.raises(ApprovalPending) as caught:
        await resolve_interruptions(state, port)
    assert caught.value.approval_id == "apr_1" and len(port.requests) == 2
    again = ScriptedApproval([ApprovalDecision(approved=True), ApprovalDecision(approved=True)])
    assert await resolve_interruptions(state, again) is state  # a RunState is updated in place
    await run(agent, state)
    assert sorted(tools.ran, key=str) == [("wipe", {"path": "/a"}), ("wipe", {"path": "/b"})]


async def test_invalid_json_and_non_object_arguments_are_model_visible_errors():
    tools = Tools()
    echo = tools.export(executor=Executor())["echo"]
    for raw in ("{not json", "[1, 2]"):
        result = await run(agent_with([echo], [("echo", raw, "c1")]))
        assert decoded(output_of(result, "c1"))["error"]["kind"] == "tool-args-invalid"
    assert tools.ran == []


def test_approval_request_arguments_fall_back_to_empty_for_bad_json():
    from agent_adapter.integrations.openai_agents import _parse_arguments

    assert _parse_arguments(None) == {}
    assert _parse_arguments("{bad") == {}
    assert _parse_arguments("[1]") == {}
    assert _parse_arguments('{"a": 1}') == {"a": 1}


def test_strict_flag_and_the_downgrade_rule():
    plain = function_tool(Tools().registry.get("pay"), strict=False)
    assert plain.strict_json_schema is False
    assert plain.params_json_schema["required"] == ["amount"]
    imported = spec_from_schema(
        "loose",
        "A tool whose schema is not portable.",
        {"type": "object", "properties": {"q": {"type": "string", "format": "uri"}}},
        lambda q: q,
        portable=False,
    )
    tool = function_tool(imported)
    assert tool.strict_json_schema is False
    assert "additionalProperties" not in tool.params_json_schema


def test_targets_and_missing_dependency(monkeypatch):
    from agent_adapter.integrations import openai_agents

    elsewhere = build_spec(lambda: 1, name="elsewhere", description="x", targets="mcp")
    with pytest.raises(ToolUsageError, match="not exported"):
        function_tool(elsewhere)
    monkeypatch.setattr(openai_agents.OpenAIAgentsIntegration, "requires", ("aa_missing_pkg",))
    with pytest.raises(MissingDependency, match="openai-agents"):
        function_tool(Tools().registry.get("pay"))


async def test_default_executor_runs_the_tool_and_the_run_id_follows_the_trace(monkeypatch):
    import agents.tracing

    from agent_adapter.integrations import openai_agents

    seen: list[str] = []
    registry = Registry()

    @registry.tool
    async def whoami(ctx: ToolContext) -> str:
        """The run id the tool sees."""
        seen.append(ctx.run_id)
        return "ok"

    tool = function_tools(registry)[0]  # executor=None: the default executor
    result = await run(agent_with([tool], [("whoami", {}, "c1")]))
    assert output_of(result, "c1") == "ok" and seen[0].startswith("run_")

    class Trace:
        trace_id = "trace_abc"

    monkeypatch.setattr(agents.tracing, "get_current_trace", lambda: Trace())
    assert openai_agents.INTEGRATION.bind_context(object()).run_id == "trace_abc"
    monkeypatch.setattr(agents.tracing, "get_current_trace", lambda: None)
    assert openai_agents.INTEGRATION.bind_context(object()).run_id == ""


# Bridge cases kept in their own module (file size); collected here so the session measures them.
globals().update((n, f) for n, f in vars(openai_agents_cases).items() if n.startswith("test_"))
