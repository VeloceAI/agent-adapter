"""Contract kit C1-C8 for CrewAI, through `BaseTool.arun` (plus the sync `run` bridge)."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("crewai")

from crewai.tools import BaseTool

from agent_adapter.core.errors import MissingDependency, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.crewai import crewai_tool, crewai_tools
from tests.fakes.approval import ScriptedApproval
from tests.fakes.hooks import RecordingHook
from tests.integrations.kit import contract_tests
from tests.integrations.sample_tools import add_wipe_registry as registry

pytestmark = pytest.mark.contract


class CrewAIHarness:
    """Drives `BaseTool`s the way CrewAI calls them (`arun`, which validates first)."""

    name = "crewai"
    edits_supported = True  # the executor applies the port's edited_args
    approval = "port"  # the executor's approval port decides
    native_call_ids = False  # run(**kwargs) carries no call id

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return crewai_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return crewai_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await native.arun(**args)

    def value(self, raw: Any) -> Any:
        return raw

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, ValueError):  # CrewAI's own args_schema validation
            return "arguments validation failed" in str(raw)
        return isinstance(raw, dict) and set(raw) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await native.arun(**args)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.args_schema.model_json_schema()

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # args_schema is the spec's own model


globals().update((item.__name__, item) for item in contract_tests(CrewAIHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(CrewAIHarness, runner_factory=Executor)
)


# --- bridge behaviour beyond the kit ---------------------------------------------------------


def test_the_exported_tool_is_a_base_tool_with_the_spec_model():
    reg = registry()
    tool = crewai_tools(reg)[0]
    assert isinstance(tool, BaseTool)
    assert tool.args_schema is reg.get("add").args_model
    assert tool.cache_function({"a": 1}, 3) is False  # every call reaches the executor
    assert "Tool Name: add" in tool.formatted_description


def test_sync_run_outside_a_running_loop():
    tool = crewai_tool(registry().get("add"), executor=Executor())
    assert tool.run(a=1) == 3


async def test_sync_run_inside_a_running_loop():
    hook = RecordingHook()
    tool = crewai_tool(registry().get("add"), executor=Executor(hooks=[hook]))
    assert tool.run(a=5) == 7  # this thread's loop is running: run_sync uses a worker loop
    structured = tool.to_structured_tool()
    assert structured.invoke({"a": 1, "b": 1}) == 2
    assert await structured.ainvoke({"a": 2}) == 4
    assert hook.phases() == ["before_tool", "after_tool"] * 3


async def test_invalid_args_are_rejected_by_crewai_before_the_executor():
    hook = RecordingHook()
    tool = crewai_tool(registry().get("add"), executor=Executor(hooks=[hook]))
    with pytest.raises(ValueError, match="arguments validation failed"):
        await tool.arun(a="x")
    assert hook.calls == []


async def test_approval_comes_from_the_executor_port():
    wipe = registry().get("wipe")
    approve = Executor(approval=ScriptedApproval([ApprovalDecision(approved=True)]))
    assert await crewai_tool(wipe, executor=approve).arun(path="/a") == "wiped /a"
    deny = Executor(approval=ScriptedApproval([ApprovalDecision(approved=False, reason="no")]))
    denied = await crewai_tool(wipe, executor=deny).arun(path="/a")
    assert denied["error"]["kind"] == "approval-denied"


async def test_default_executor_runs_the_tool():
    assert await crewai_tools(registry())[0].arun(a=40) == 42
    assert await asyncio.to_thread(lambda: crewai_tools(registry())[0].run(a=1)) == 3


def test_targets_and_missing_dependency(monkeypatch):
    from agent_adapter.integrations import crewai

    elsewhere = build_spec(lambda: 1, name="elsewhere", description="x", targets="mcp")
    with pytest.raises(ToolUsageError, match="not exported"):
        crewai_tool(elsewhere)
    monkeypatch.setattr(crewai.CrewAIIntegration, "requires", ("aa_missing_pkg",))
    with pytest.raises(MissingDependency, match="crewai"):
        crewai_tool(registry().get("add"))
