"""Contract kit C1-C8 for Pydantic AI, through `Agent.run` with a `FunctionModel`."""

import os
from typing import Any

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
pytest.importorskip("pydantic_ai")

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart

from agent_adapter.approval import CallbackApproval
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, MissingDependency, ToolError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.pydantic_ai import (
    pydantic_ai_tool,
    pydantic_ai_tools,
    resolve_deferred,
)
from tests.fakes.approval import ScriptedApproval
from tests.integrations import pydantic_ai_cases
from tests.integrations.kit import contract_tests
from tests.integrations.pydantic_ai_support import (
    agent_with,
    is_error_part,
    result_part,
    value_of,
)

pytestmark = pytest.mark.contract


class PydanticAIHarness:
    """Drives `Tool`s through Pydantic AI's run loop and its deferred approvals."""

    name = "pydantic_ai"
    edits_supported = True  # ToolApproved(override_args=...)
    approval = "native"
    native_uses_port = True  # resolve_deferred asks the port
    port_asks = 1

    def __init__(self) -> None:
        self.decision: ApprovalDecision | None = None

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return pydantic_ai_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return pydantic_ai_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        result = await agent_with([native], [(native.name, args, call_id)]).run("go")
        assert isinstance(result.output, str), f"unexpected output {result.output!r}"
        return result_part(result, call_id)

    def value(self, raw: Any) -> Any:
        return value_of(raw)

    def is_error(self, raw: Any) -> bool:
        return is_error_part(raw)

    def approval_port(self) -> Any:
        return CallbackApproval(lambda request: self.decision)

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision, *, port: Any = None
    ) -> Any:
        self.decision = decision
        agent = agent_with([native], [(native.name, args, "call_kit_c6")])
        first = await agent.run("go")
        assert [part.tool_name for part in first.output.approvals] == [native.name]
        results = await resolve_deferred(first.output, port, run_id=first.run_id)
        second = await agent.run(
            message_history=first.all_messages(), deferred_tool_results=results
        )
        return result_part(second, "call_kit_c6")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.tool_def.parameters_json_schema

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # Pydantic AI takes the neutral schema


globals().update((item.__name__, item) for item in contract_tests(PydanticAIHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(PydanticAIHarness, runner_factory=Executor)
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

        @registry.tool
        async def lookup(key: str, ctx: ToolContext) -> dict[str, Any]:
            """Look a key up; unknown keys are a tool error."""
            self.ran.append(("lookup", {"key": key}))
            if key == "missing":
                raise ToolError(f"no such key: {key}")
            return {"key": key, "run": ctx.run_id, "thread": ctx.thread_id}

        self.registry = registry

    def export(self, **kwargs: Any) -> dict[str, Any]:
        return {tool.name: tool for tool in pydantic_ai_tools(self.registry, **kwargs)}


async def test_callable_confirm_defers_only_the_calls_it_selects():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    calls = [("pay", {"amount": 5}, "small"), ("pay", {"amount": 500, "to": "bob"}, "big")]
    agent = agent_with([pay], calls)
    first = await agent.run("go")
    assert isinstance(first.output, DeferredToolRequests)
    assert [part.tool_call_id for part in first.output.approvals] == ["big"]
    assert first.output.metadata["big"] == {"tool": "pay"}
    assert tools.ran == [("pay", {"amount": 5, "to": "acme"})]
    port = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"amount": 50})])
    results = await resolve_deferred(first.output, port, run_id=first.run_id)
    assert port.requests[0].run_id == first.run_id
    assert port.requests[0].metadata == {"integration": "pydantic_ai", "native": {"tool": "pay"}}
    second = await agent.run(message_history=first.all_messages(), deferred_tool_results=results)
    assert value_of(result_part(second, "big")) == "paid 50 to acme"
    assert tools.ran[-1] == ("pay", {"amount": 50, "to": "acme"})


async def test_callable_confirm_denial_reaches_the_model_as_an_error():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    agent = agent_with([pay], [("pay", {"amount": 500}, "big")])
    first = await agent.run("go")
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="over budget")])
    results = await resolve_deferred(first.output, port)
    second = await agent.run(message_history=first.all_messages(), deferred_tool_results=results)
    part = result_part(second, "big")
    assert is_error_part(part) and "over budget" in value_of(part)["error"]["message"]
    assert tools.ran == []


async def test_invalid_args_for_a_callable_rule_are_a_model_retry():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    result = await agent_with([pay], [("pay", {"amount": "lots"}, "c1")]).run("go")
    part = result_part(result, "c1")
    assert part.outcome == "retried" and "invalid arguments" in part.content
    assert tools.ran == []


async def test_tool_errors_are_model_visible_and_the_context_carries_native_ids():
    tools = Tools()
    lookup = tools.export(executor=Executor())["lookup"]
    calls = [("lookup", {"key": "missing"}, "c1"), ("lookup", {"key": "k"}, "c2")]
    result = await agent_with([lookup], calls).run("go", conversation_id="conv_1")
    assert value_of(result_part(result, "c1")) == {
        "error": {"kind": "tool-error", "message": "no such key: missing"}
    }
    found = value_of(result_part(result, "c2"))
    assert found == {"key": "k", "run": result.run_id, "thread": "conv_1"}


async def test_pending_decisions_are_raised_after_asking_every_call():
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart("wipe", {"path": "/a"}, tool_call_id="w1"),
            ToolCallPart("wipe", '{"path": "/b"}', tool_call_id="w2"),
        ],
        calls=[ToolCallPart("external", {}, tool_call_id="x1")],
    )
    port = ScriptedApproval(
        [ApprovalPending("apr_1", tool="wipe"), ApprovalDecision(approved=True)]
    )
    with pytest.raises(ApprovalPending) as caught:
        await resolve_deferred(requests, port, run_id="run_9")
    assert caught.value.approval_id == "apr_1"
    assert [(r.call_id, r.arguments, r.run_id) for r in port.requests] == [
        ("w1", {"path": "/a"}, "run_9"),
        ("w2", {"path": "/b"}, "run_9"),
    ]
    both = ScriptedApproval([ApprovalDecision(approved=True), ApprovalDecision(approved=False)])
    results = await resolve_deferred(requests, both)
    assert set(results.approvals) == {"w1", "w2"} and results.calls == {}
    assert results.approvals["w1"].override_args is None


def test_bind_context_without_native_ids():
    from agent_adapter.integrations.pydantic_ai import INTEGRATION

    ctx = INTEGRATION.bind_context(object())
    assert (ctx.run_id, ctx.thread_id, ctx.call_id) == ("", "", "")


def test_targets_and_missing_dependency(monkeypatch):
    from agent_adapter.integrations import pydantic_ai

    elsewhere = build_spec(lambda: 1, name="elsewhere", description="x", targets="mcp")
    with pytest.raises(ToolUsageError, match="not exported"):
        pydantic_ai_tool(elsewhere)
    monkeypatch.setattr(pydantic_ai.PydanticAIIntegration, "requires", ("aa_missing_pkg",))
    with pytest.raises(MissingDependency, match="pydantic-ai"):
        pydantic_ai_tool(Tools().registry.get("pay"))


async def test_default_executor_runs_the_tool():
    tools = Tools()
    lookup = pydantic_ai_tools(tools.registry)[1]  # executor=None: the default executor
    result = await agent_with([lookup], [("lookup", {"key": "k"}, "c1")]).run("go")
    assert value_of(result_part(result, "c1"))["key"] == "k"


# Bridge cases kept in their own module (file size); collected here so the session measures them.
globals().update((n, f) for n, f in vars(pydantic_ai_cases).items() if n.startswith("test_"))
