"""Contract kit C1-C8 for CAMEL-AI, plus the bridge's own edge cases."""

import json
import time
from typing import Any

import pytest

pytest.importorskip("camel")

from camel.agents import ChatAgent
from camel.models.stub_model import StubModel
from camel.types import ModelType
from openai.types.chat import ChatCompletion

from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations import camel
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.schema.transforms import to_openai_parameters
from tests.fakes.approval import ScriptedApproval
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


class ScriptedModel(StubModel):
    """A CAMEL model that asks for the scripted tool calls, then says it is done."""

    def __init__(self, *calls: tuple[str, str, dict[str, Any]]) -> None:
        super().__init__(ModelType.STUB)
        self.calls = list(calls)

    def _reply(self) -> ChatCompletion:
        message: dict[str, Any] = {"role": "assistant", "content": "done"}
        if self.calls:
            call_id, name, args = self.calls.pop(0)
            function = {"name": name, "arguments": json.dumps(args)}
            calls = [{"id": call_id, "type": "function", "function": function}]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
        return ChatCompletion.model_validate(
            {
                "id": "scripted",
                "model": "stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
            }
        )

    async def _arun(self, messages: Any, response_format: Any = None, tools: Any = None) -> Any:
        return self._reply()

    def _run(self, messages: Any, response_format: Any = None, tools: Any = None) -> Any:
        return self._reply()


def agent(native: Any, *calls: tuple[str, str, dict[str, Any]]) -> ChatAgent:
    return ChatAgent(model=ScriptedModel(*calls), tools=[native])


class CamelHarness:
    """Drives CAMEL's `ChatAgent.astep` with a scripted model."""

    name = "camel"
    edits_supported = True  # the executor's port decides, and the executor applies edits
    approval = "port"
    native_call_ids = False  # CAMEL passes a tool only its arguments

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return camel.camel_tool(spec, executor=executor, strict=False)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return camel.camel_tools(registry, executor=executor, strict=False)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        name = native.get_function_name()
        response = await agent(native, (call_id, name, args)).astep("go")
        (record,) = response.info["tool_calls"]
        return record

    def value(self, raw: Any) -> Any:
        return raw.result

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception):
            return True
        value = raw.result
        if isinstance(value, str):
            return value.startswith("Tool execution failed")
        return isinstance(value, dict) and set(value) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_camel_ok")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        function = native.get_openai_tool_schema()["function"]
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_openai_parameters(spec.json_schema(), strict=False)


globals().update((item.__name__, item) for item in contract_tests(CamelHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(CamelHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


def remove(item: str, note: str = "") -> str:
    """Remove an item."""
    return f"removed {item}"


def test_the_schema_is_the_openai_transform():
    spec = build_spec(remove)
    schema = camel.camel_tool(spec).openai_tool_schema
    assert schema["function"]["strict"] is True
    assert schema["function"]["parameters"] == to_openai_parameters(spec.json_schema())
    plain = camel.camel_tool(spec, strict=False).openai_tool_schema
    assert plain["function"]["strict"] is False
    assert plain["function"]["parameters"]["required"] == ["item"]


def test_an_undescribed_parameter_gives_no_warning():
    # pyproject turns warnings into errors: CAMEL's own check would warn here.
    assert camel.camel_tool(build_spec(remove)).get_openai_tool_schema()["type"] == "function"


async def test_a_confirm_tool_is_decided_by_the_executor_port_only():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="nope")])
    ran: list[str] = []

    def wipe(item: str) -> str:
        """Wipe an item."""
        ran.append(item)
        return f"wiped {item}"

    native = camel.camel_tool(build_spec(wipe, confirm=True), executor=Executor(approval=port))
    response = await agent(native, ("c1", "wipe", {"item": "a"})).astep("go")
    (record,) = response.info["tool_calls"]
    assert record.result["error"]["kind"] == "approval-denied"
    assert ran == [] and [r.tool for r in port.requests] == ["wipe"]


async def test_without_a_port_a_confirm_tool_is_denied():
    native = camel.camel_tool(build_spec(remove, confirm=True), executor=Executor())
    response = await agent(native, ("c1", "remove", {"item": "a"})).astep("go")
    assert response.info["tool_calls"][0].result["error"]["kind"] == "approval-denied"


def test_the_sync_step_runs_through_run_sync():
    runner = FakeRunner()
    native = camel.camel_tool(build_spec(remove), executor=runner)
    response = agent(native, ("c1", "remove", {"item": "a"})).step("go")
    assert response.info["tool_calls"][0].result == "removed a"
    (inv,) = runner.calls
    assert inv.call.arguments == {"item": "a", "note": ""} and inv.ctx.approval is None


async def test_positional_arguments_are_refused():
    native = camel.camel_tool(build_spec(remove), executor=FakeRunner())
    with pytest.raises(ToolUsageError, match="keyword arguments only"):
        await native.async_call("a")
    with pytest.raises(ToolUsageError):
        native("a")


def test_an_unsupported_camel_release_is_refused(monkeypatch):
    import camel as camel_pkg

    monkeypatch.setattr(camel_pkg, "__version__", "0.3.0")
    camel._tool_class.cache_clear()
    try:
        with pytest.raises(IntegrationError, match="outside the supported range"):
            camel.camel_tool(build_spec(remove))
    finally:
        camel._tool_class.cache_clear()


def test_export_tools_honours_targets():
    registry = Registry()
    registry.add(build_spec(remove))
    registry.add(build_spec(remove, name="hidden", targets=("openai",)))
    assert [t.get_function_name() for t in camel.camel_tools(registry)] == ["remove"]


def test_the_example_runs(capsys):
    assert _run("camel_tools.py", capsys).splitlines() == [
        "[('shipping_rate', 5.0)]",
        "[('send_invoice', 'invoiced cus_1024 for 49.50 USD')]",
        "[('send_invoice', {'error': {'kind': 'approval-denied', 'message': "
        "\"approval denied for tool 'send_invoice': unknown customer\"}})]",
    ]
