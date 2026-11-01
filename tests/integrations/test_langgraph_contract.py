"""LangGraph contract: kit C1-C8 for `tool_node` run inside a real graph, approvals by interrupt."""

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import NodeCancelledError

from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.langgraph import INTEGRATION, LangGraphApproval
from tests.integrations.kit import contract_tests
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract


def _input(native: Any, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    name = native.tools[0]["function"]["name"]
    call = {"name": name, "args": args, "id": call_id}
    return {"messages": [AIMessage(content="", tool_calls=[call])]}


class LangGraphHarness:
    """A one-tool `tool_node` in START -> tools -> END, with an in-memory checkpointer."""

    name = "langgraph"
    edits_supported = True  # the executor applies edited args
    approval = "native"  # interrupt() / Command(resume=...)
    native_uses_port = True  # LangGraphApproval is the runner's port
    port_asks = 2  # asked when the call interrupts, and again when the node re-runs

    def approval_port(self) -> ApprovalPort:
        return LangGraphApproval()

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return INTEGRATION.export_tool(spec, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        try:
            out = await one_node(native).ainvoke(_input(native, args, call_id), config())
        except NodeCancelledError as exc:
            # LangGraph reports a node that raised CancelledError as NodeCancelledError; the
            # tool node itself let it through (see test_tools.py)
            if isinstance(exc.__cause__, asyncio.CancelledError):
                raise exc.__cause__ from None
            raise
        message = out["messages"][-1]
        assert isinstance(message, ToolMessage) and message.tool_call_id == call_id
        return message

    def value(self, raw: ToolMessage) -> Any:
        try:
            return json.loads(raw.content)  # type: ignore[arg-type]
        except json.JSONDecodeError:
            return raw.content

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or raw.status == "error"

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        app, cfg = one_node(native), config()
        out = await app.ainvoke(_input(native, args, "call_lg_approval"), cfg, version="v2")
        [interrupt] = out.interrupts
        assert interrupt.value["request"]["arguments"] == args
        out = await app.ainvoke(resume(interrupt, decision), cfg, version="v2")
        assert not out.interrupts
        return out.value["messages"][-1]

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        [tool] = native.tools
        function = tool["function"]
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # bound to a model as an OpenAI-format tool, unchanged


globals().update((item.__name__, item) for item in contract_tests(LangGraphHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(LangGraphHarness, runner_factory=Executor)
)
