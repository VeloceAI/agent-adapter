"""DSPy contract: kit C1-C8 through `dspy.Tool.acall`, plus the sync bridge and ReAct offline."""

import asyncio
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

dspy = pytest.importorskip("dspy")

from dspy.utils.dummies import DummyLM  # noqa: E402

from agent_adapter.core.context import ToolContext  # noqa: E402
from agent_adapter.core.errors import (  # noqa: E402
    IntegrationError,
    MissingDependency,
    ToolFailed,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry  # noqa: E402
from agent_adapter.core.tool import ToolSpec, build_spec  # noqa: E402
from agent_adapter.core.types import ApprovalDecision  # noqa: E402
from agent_adapter.exec import Executor  # noqa: E402
from agent_adapter.exec.defaults import default_executor, set_default_executor  # noqa: E402
from agent_adapter.integrations import dspy as dspy_integration  # noqa: E402
from agent_adapter.integrations.base import ToolRunner  # noqa: E402
from agent_adapter.integrations.dspy import INTEGRATION, dspy_tool, dspy_tools  # noqa: E402
from agent_adapter.schema.transforms import to_plain  # noqa: E402
from tests.fakes.hooks import RecordingHook  # noqa: E402
from tests.integrations.kit import FakeRunner, contract_tests  # noqa: E402
from tests.integrations.sample_tools import gst_rate  # noqa: E402

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class DSPyHarness:
    """Drives `dspy.Tool.acall`, the path `dspy.ReAct.aforward` uses."""

    name = "dspy"
    edits_supported = True  # the executor's port applies edited args
    approval = "port"
    native_call_ids = False  # DSPy passes only keyword arguments to a tool

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return dspy_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return dspy_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await native.acall(**args)

    def value(self, raw: Any) -> Any:
        return raw

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, dict) and set(raw) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await native.acall(**args)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        function = native.format_as_litellm_function_call()["function"]
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        plain = to_plain(spec.json_schema())
        return {
            "type": "object",
            "properties": plain.get("properties", {}),
            "required": plain.get("required", []),
        }


globals().update((item.__name__, item) for item in contract_tests(DSPyHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(DSPyHarness, runner_factory=Executor)
)


async def crash(item: str) -> str:
    """Fails with a foreign error."""
    raise RuntimeError("secret detail")


async def echo_extras(tag: str, ctx: ToolContext) -> str:
    """Report what the context holds."""
    return f"{tag}:{type(ctx.extras['dspy']).__name__}:{ctx.call_id.startswith('call_')}"


@pytest.fixture
def rate() -> ToolSpec:
    return build_spec(gst_rate)


def test_export_maps_the_schema_to_args_and_arg_types(rate):
    tool = dspy_tool(rate)
    assert isinstance(tool, dspy.Tool)
    assert (tool.name, tool.desc) == ("gst_rate", "GST rate for an item.")
    assert tool.args == {
        "item": {"type": "string", "description": "What is sold."},
        "region": {"type": "string", "description": "Tax region.", "default": "IN"},
    }
    assert tool.arg_types == {"item": str, "region": str}


@pytest.mark.parametrize(
    ("prop", "expected"),
    [
        ({"type": "integer"}, int),
        ({"type": "number"}, float),
        ({"type": "boolean"}, bool),
        ({"type": "array", "items": {"type": "string"}}, list),
        ({"type": "object"}, dict),
        ({"type": "null"}, Any),
        ({"type": ["string", "null"]}, Any),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, Any),
        ({}, Any),
    ],
)
def test_arg_types_follow_the_json_type(prop, expected):
    assert dspy_integration._python_type(prop) is expected


async def test_acall_returns_the_value_and_model_visible_errors(rate):
    tool = dspy_tool(rate, executor=FakeRunner())
    assert await tool.acall(item="tea") == 18.0
    assert await tool.acall(item="kite") == {
        "error": {"kind": "tool-error", "message": "no rate for 'kite'"}
    }
    # DSPy's own JSON-schema check would raise ValueError quoting the value; the runner's
    # ToolArgsInvalid never holds input values.
    invalid = await tool.acall(item="tea", region=7123)
    assert invalid["error"]["kind"] == "tool-args-invalid"
    assert "7123" not in invalid["error"]["message"]


async def test_foreign_errors_propagate_through_the_executor():
    tool = dspy_tool(build_spec(crash), executor=Executor())
    with pytest.raises(ToolFailed):
        await tool.acall(item="x")


async def test_context_holds_the_dspy_tool_and_a_fresh_call_id():
    tool = dspy_tool(build_spec(echo_extras), executor=FakeRunner())
    assert await tool.acall(tag="t") == "t:AdapterTool:True"


def test_sync_call_outside_a_loop_uses_run_sync(rate):
    tool = dspy_tool(rate, executor=FakeRunner())
    assert tool(item="tea") == 18.0


async def test_sync_call_inside_a_running_loop_uses_a_worker_thread(rate):
    tool = dspy_tool(rate, executor=FakeRunner())
    assert tool(item="tea") == 18.0


async def test_default_executor_is_used_at_call_time(rate):
    previous = default_executor()
    hook = RecordingHook()
    tool = dspy_tool(rate)
    set_default_executor(Executor(hooks=[hook]))
    try:
        assert await tool.acall(item="tea") == 18.0
    finally:
        set_default_executor(previous)
    assert hook.phases() == ["before_tool", "after_tool"]


def test_single_export_honours_targets():
    spec = build_spec(gst_rate, targets="openai")
    with pytest.raises(ToolUsageError, match="not exported to 'dspy'"):
        dspy_tool(spec)


def test_a_dspy_without_the_validation_hook_fails_loudly(monkeypatch, rate):
    class OldTool:
        """A dspy.Tool lacking `_validate_and_parse_args`."""

    dspy_integration._tool_class.cache_clear()
    monkeypatch.setattr(dspy, "Tool", OldTool)
    try:
        with pytest.raises(IntegrationError, match="_validate_and_parse_args"):
            dspy_tool(rate)
    finally:
        dspy_integration._tool_class.cache_clear()


def test_missing_dspy_raises_missing_dependency(monkeypatch, rate):
    dspy_integration._tool_class.cache_clear()
    monkeypatch.setitem(sys.modules, "dspy", None)
    try:
        with pytest.raises(MissingDependency, match=r"agent-adapter\[dspy\]"):
            dspy_tool(rate)
    finally:
        dspy_integration._tool_class.cache_clear()


def test_integration_has_no_native_approval_gate():
    assert INTEGRATION.bind_approval(object()) is None
    assert (INTEGRATION.name, INTEGRATION.requires, INTEGRATION.extra) == (
        "dspy",
        ("dspy",),
        "dspy",
    )


def _react_lm() -> DummyLM:
    return DummyLM(
        [
            {
                "next_thought": "Look up the rate.",
                "next_tool_name": "gst_rate",
                "next_tool_args": {"item": "tea"},
            },
            {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "The tool said 18.", "answer": "18.0"},
        ]
    )


def test_react_runs_the_tool_through_the_executor(rate):
    hook = RecordingHook()
    react = dspy.ReAct(
        "question -> answer", tools=[dspy_tool(rate, executor=Executor(hooks=[hook]))]
    )
    with dspy.context(lm=_react_lm()):
        prediction = react(question="GST on tea?")
    assert prediction.answer == "18.0"
    assert prediction.trajectory["observation_0"] == 18.0
    assert hook.phases() == ["before_tool", "after_tool"]


async def test_react_async_runs_the_tool_through_acall(rate):
    react = dspy.ReAct("question -> answer", tools=[dspy_tool(rate, executor=FakeRunner())])
    with dspy.context(lm=_react_lm()):
        prediction = await react.acall(question="GST on tea?")
    assert prediction.trajectory["observation_0"] == 18.0


async def test_cancelling_acall_cancels_the_tool():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def wait(label: str) -> str:
        """Wait forever."""
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return label

    task = asyncio.ensure_future(dspy_tool(build_spec(wait), executor=Executor()).acall(label="x"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


def test_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "dspy_tools.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "shipping_rate 18.0",
        "{'error': {'kind': 'tool-error', 'message': \"no rate for 'mars'\"}}",
        "answer: 18.0",
    ]
