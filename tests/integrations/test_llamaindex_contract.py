"""Contract kit C1-C8 for LlamaIndex, through `FunctionTool.acall` (plus the sync bridge)."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.tools import FunctionTool, ToolOutput

from agent_adapter.core.errors import MissingDependency, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.llamaindex import llamaindex_tool, llamaindex_tools
from tests.fakes.approval import ScriptedApproval
from tests.integrations.kit import contract_tests
from tests.integrations.sample_tools import add_wipe_registry as registry

pytestmark = pytest.mark.contract

PARAMETER_KEYS = ("type", "properties", "required", "definitions", "$defs")


class LlamaIndexHarness:
    """Drives `FunctionTool`s the way LlamaIndex agents call them."""

    name = "llamaindex"
    edits_supported = True  # the executor applies the port's edited_args
    approval = "port"  # the executor's approval port decides
    native_call_ids = False  # acall(**kwargs) carries no call id

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return llamaindex_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return llamaindex_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await native.acall(**args)

    def value(self, raw: Any) -> Any:
        return raw.raw_output

    def is_error(self, raw: Any) -> bool:
        output = raw.raw_output if isinstance(raw, ToolOutput) else None
        return isinstance(output, dict) and set(output) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await native.acall(**args)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        metadata = native.metadata
        return metadata.name, metadata.description, metadata.get_parameters_dict()

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        # LlamaIndex shows `fn_schema`'s schema minus `title` / `additionalProperties`.
        return {k: v for k, v in spec.json_schema().items() if k in PARAMETER_KEYS}


globals().update((item.__name__, item) for item in contract_tests(LlamaIndexHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(LlamaIndexHarness, runner_factory=Executor)
)


# --- bridge behaviour beyond the kit ---------------------------------------------------------


def test_the_exported_tool_keeps_the_args_model_and_description():
    reg = registry()
    tool = llamaindex_tools(reg)[0]
    assert isinstance(tool, FunctionTool)
    assert tool.metadata.fn_schema is reg.get("add").args_model
    assert tool.metadata.description == "Add two integers."
    assert tool.metadata.to_openai_tool()["function"]["name"] == "add"


def test_sync_call_outside_a_running_loop():
    tool = llamaindex_tool(registry().get("add"), executor=Executor())
    assert tool.call(a=1).raw_output == 3
    assert tool(a=2, b=2).raw_output == 4


async def test_sync_call_inside_a_running_loop_uses_a_worker_loop():
    tool = llamaindex_tool(registry().get("add"), executor=Executor())
    output = await asyncio.to_thread(lambda: tool.call(a=5))  # a thread without a loop
    assert output.raw_output == 7
    assert tool.call(a=1, b=1).raw_output == 2  # this thread's loop is running


async def test_approval_comes_from_the_executor_port():
    wipe = registry().get("wipe")
    approve = Executor(approval=ScriptedApproval([ApprovalDecision(approved=True)]))
    assert (await llamaindex_tool(wipe, executor=approve).acall(path="/a")).raw_output == "wiped /a"
    deny = Executor(approval=ScriptedApproval([ApprovalDecision(approved=False, reason="no")]))
    denied = (await llamaindex_tool(wipe, executor=deny).acall(path="/a")).raw_output
    assert denied["error"]["kind"] == "approval-denied"


async def test_default_executor_runs_the_tool():
    assert (await llamaindex_tools(registry())[0].acall(a=40)).raw_output == 42


def test_targets_and_missing_dependency(monkeypatch):
    from agent_adapter.integrations import llamaindex

    elsewhere = build_spec(lambda: 1, name="elsewhere", description="x", targets="mcp")
    with pytest.raises(ToolUsageError, match="not exported"):
        llamaindex_tool(elsewhere)
    monkeypatch.setattr(llamaindex.LlamaIndexIntegration, "requires", ("aa_missing_pkg",))
    with pytest.raises(MissingDependency, match="llamaindex"):
        llamaindex_tool(registry().get("add"))
