"""smolagents contract: kit C1-C8 through `Tool.__call__`, the sync bridge and an offline agent."""

import asyncio
import dataclasses
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pydantic
import pytest

smolagents = pytest.importorskip("smolagents")

from smolagents import ChatMessage, ChatMessageToolCall, MessageRole, ToolCallingAgent  # noqa: E402
from smolagents.models import ChatMessageToolCallFunction, get_tool_json_schema  # noqa: E402

from agent_adapter.core.context import ToolContext  # noqa: E402
from agent_adapter.core.errors import (  # noqa: E402
    IntegrationError,
    MissingDependency,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry  # noqa: E402
from agent_adapter.core.tool import ToolSpec, build_spec  # noqa: E402
from agent_adapter.core.types import ApprovalDecision  # noqa: E402
from agent_adapter.exec import Executor  # noqa: E402
from agent_adapter.exec.defaults import default_executor, set_default_executor  # noqa: E402
from agent_adapter.integrations import smolagents as smol_integration  # noqa: E402
from agent_adapter.integrations.base import ToolRunner  # noqa: E402
from agent_adapter.integrations.smolagents import (  # noqa: E402
    INTEGRATION,
    smolagents_tool,
    smolagents_tools,
)
from agent_adapter.schema.transforms import to_plain  # noqa: E402
from tests.fakes.hooks import RecordingHook  # noqa: E402
from tests.integrations.kit import FakeRunner, contract_tests  # noqa: E402
from tests.integrations.sample_tools import gst_rate  # noqa: E402

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def error_payload(raw: Any) -> dict[str, Any] | None:
    """The model-visible error in `raw` (a dict, or its JSON text for string tools), if any."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) and set(raw) == {"error"} else None


class SmolagentsHarness:
    """Drives `Tool.__call__(..., sanitize_inputs_outputs=True)`, as the agents do, in a thread."""

    name = "smolagents"
    edits_supported = True  # the executor's port applies edited args
    approval = "port"
    native_call_ids = False  # `forward` receives only the arguments
    cancels_tool = False  # `forward` is sync: its run_sync thread cannot be cancelled

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return smolagents_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return smolagents_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await asyncio.to_thread(lambda: native(**args, sanitize_inputs_outputs=True))

    def value(self, raw: Any) -> Any:
        return raw

    def is_error(self, raw: Any) -> bool:
        return error_payload(raw) is not None

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        function = get_tool_json_schema(native)["function"]
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        plain = to_plain(spec.json_schema())
        required = plain.get("required", [])
        properties = {
            key: {
                **prop,
                "description": prop.get("description", ""),
                **({} if key in required else {"nullable": True}),
            }
            for key, prop in plain.get("properties", {}).items()
        }
        return {"type": "object", "properties": properties, "required": required}


globals().update((item.__name__, item) for item in contract_tests(SmolagentsHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(SmolagentsHarness, runner_factory=Executor)
)


async def label(item: str) -> str:
    """Label an item."""
    if item == "kite":
        raise ToolError("no label for 'kite'")
    return f"<{item}>"


async def echo_extras(tag: str, ctx: ToolContext) -> str:
    """Report what the context holds."""
    return f"{tag}:{type(ctx.extras['smolagents']).__name__}:{ctx.call_id.startswith('call_')}"


class Address(pydantic.BaseModel):
    city: str


async def shapes(
    count: int,
    ratio: float,
    flag: bool,
    tags: list[str],
    home: Address,
    maybe: int | None = None,
    either: int | str = 1,
    anything: Any = None,
    mode: str = "a",
) -> dict[str, Any]:
    """Accept many shapes."""
    return {"count": count}


@pytest.fixture
def rate() -> ToolSpec:
    return build_spec(gst_rate)


def test_export_builds_a_tool_with_inputs_and_output_type(rate):
    tool = smolagents_tool(rate)
    assert isinstance(tool, smolagents.Tool)
    assert (tool.name, tool.description, tool.output_type) == (
        "gst_rate",
        "GST rate for an item.",
        "number",
    )
    assert tool.inputs == {
        "item": {"type": "string", "description": "What is sold."},
        "region": {
            "type": "string",
            "description": "Tax region.",
            "default": "IN",
            "nullable": True,
        },
    }


def test_inputs_map_every_json_shape_to_authorized_types():
    inputs = smolagents_tool(build_spec(shapes)).inputs
    assert {key: entry["type"] for key, entry in inputs.items()} == {
        "count": "integer",
        "ratio": "number",
        "flag": "boolean",
        "tags": "array",
        "home": "object",
        "maybe": "integer",
        "either": ["integer", "string"],
        "anything": "any",
        "mode": "string",
    }
    assert inputs["home"]["properties"] == {"city": {"type": "string"}}
    assert inputs["tags"]["items"] == {"type": "string"}
    assert inputs["maybe"]["nullable"] is True and "anyOf" not in inputs["maybe"]
    assert inputs["either"]["anyOf"] == [{"type": "integer"}, {"type": "string"}]
    assert all(entry["description"] == "" for entry in inputs.values())
    assert "nullable" not in inputs["count"]


@pytest.mark.parametrize(
    ("prop", "expected"),
    [
        ({"type": "integer"}, "integer"),
        ({"type": "tuple"}, "any"),
        ({"type": ["string", "null"]}, ["string", "null"]),
        ({"type": ["string", "tuple"]}, "string"),
        ({"type": ["tuple"]}, "any"),
        ({"anyOf": [{"type": "integer"}, {"enum": [1]}]}, "any"),
        ({"anyOf": [{"type": "null"}]}, "null"),
        ({"enum": ["a", "b"]}, "any"),
    ],
)
def test_input_types_fall_back_to_any(prop, expected):
    assert smol_integration._entry(prop, required=True)["type"] == expected


def test_union_with_an_untyped_branch_drops_any_of():
    entry = smol_integration._entry({"anyOf": [{"type": "integer"}, {"enum": [1]}]}, required=True)
    assert "anyOf" not in entry and entry["type"] == "any"


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        (int, "integer"),
        (str, "string"),
        (dict[str, int], "object"),
        (list[int], "array"),
        (int | None, "any"),
        (None, "any"),
    ],
)
def test_output_type_follows_the_result_schema(returns, expected):
    async def value() -> Any:
        """Return a value."""
        return None

    spec = dataclasses.replace(build_spec(value), returns=returns)
    assert smolagents_tool(spec).output_type == expected


async def test_numbers_and_errors_come_back_as_values(rate):
    tool = smolagents_tool(rate, executor=FakeRunner())
    assert tool(item="tea") == 18.0
    assert tool(item="kite") == {"error": {"kind": "tool-error", "message": "no rate for 'kite'"}}
    invalid = tool(item="tea", region=7123)
    assert invalid["error"]["kind"] == "tool-args-invalid"
    assert "7123" not in invalid["error"]["message"]


def test_string_tools_return_errors_as_json_text():
    tool = smolagents_tool(build_spec(label), executor=FakeRunner())
    assert tool(item="tea", sanitize_inputs_outputs=True) == "<tea>"
    text = tool(item="kite", sanitize_inputs_outputs=True)
    assert json.loads(text) == {"error": {"kind": "tool-error", "message": "no label for 'kite'"}}


def test_positional_arguments_map_to_inputs_in_order(rate):
    tool = smolagents_tool(rate, executor=FakeRunner())
    assert tool("tea") == 18.0  # agents pass a lone non-dict argument positionally
    assert tool({"item": "tea"}) == 18.0  # smolagents unpacks a matching dict itself
    too_many = tool("tea", "IN", "extra")
    assert too_many["error"]["kind"] == "tool-args-invalid"
    assert "at most 2 positional" in too_many["error"]["message"]


def test_too_many_positional_arguments_on_a_string_tool_are_json_text():
    tool = smolagents_tool(build_spec(label), executor=FakeRunner())
    error = json.loads(tool("tea", "extra"))
    assert error["error"]["kind"] == "tool-args-invalid"
    assert "at most 1 positional" in error["error"]["message"]


async def two(a: str, b: str = "-") -> str:
    """Join two strings."""
    return a + b


def test_positional_and_keyword_for_the_same_input_are_invalid():
    probe = RecordingHook()
    tool = smolagents_tool(build_spec(two), executor=FakeRunner(hooks=[probe]))
    error = json.loads(tool("VALX", a="VALY"))
    assert error["error"]["kind"] == "tool-args-invalid"
    assert "a: given both positionally and by name" in error["error"]["message"]
    assert "VALX" not in error["error"]["message"] and "VALY" not in error["error"]["message"]
    assert probe.calls == []  # never reached the runner
    assert tool("x", b="y") == "xy"


def test_context_holds_the_tool_and_a_fresh_call_id():
    tool = smolagents_tool(build_spec(echo_extras), executor=FakeRunner())
    assert tool(tag="t") == "t:AdapterTool:True"


async def test_sync_bridge_runs_inside_a_running_loop(rate):
    asyncio.get_running_loop()  # a loop is running in this thread
    hook = RecordingHook()
    tool = smolagents_tool(rate, executor=Executor(hooks=[hook]))
    assert tool(item="tea") == 18.0  # forward blocks; run_sync uses a worker thread
    assert hook.phases() == ["before_tool", "after_tool"]


async def test_default_executor_is_used_at_call_time(rate):
    previous = default_executor()
    hook = RecordingHook()
    tool = smolagents_tool(rate)
    set_default_executor(Executor(hooks=[hook]))
    try:
        assert await asyncio.to_thread(tool, item="tea") == 18.0
    finally:
        set_default_executor(previous)
    assert hook.phases() == ["before_tool", "after_tool"]


def test_python_keywords_are_rejected_as_names():
    spec = build_spec(gst_rate, name="lambda")
    with pytest.raises(IntegrationError, match="lambda"):
        smolagents_tool(spec)


def test_single_export_honours_targets():
    with pytest.raises(ToolUsageError, match="not exported to 'smolagents'"):
        smolagents_tool(build_spec(gst_rate, targets="openai"))


def test_missing_smolagents_raises_missing_dependency(monkeypatch, rate):
    smol_integration._tool_class.cache_clear()
    monkeypatch.setitem(sys.modules, "smolagents", None)
    try:
        with pytest.raises(MissingDependency, match=r"agent-adapter\[smolagents\]"):
            smolagents_tool(rate)
    finally:
        smol_integration._tool_class.cache_clear()


def test_integration_has_no_native_approval_gate():
    assert INTEGRATION.bind_approval(object()) is None
    assert (INTEGRATION.name, INTEGRATION.requires) == ("smolagents", ("smolagents",))


class ScriptedModel:
    """A smolagents model that calls one tool, then gives the final answer."""

    model_id = "scripted"

    def __init__(self, tool: str, arguments: dict[str, Any]) -> None:
        self.turns = [(tool, arguments), ("final_answer", {"answer": "18.0"})]

    def generate(self, messages: Any, stop_sequences: Any = None, **kwargs: Any) -> ChatMessage:
        name, arguments = self.turns.pop(0)
        call = ChatMessageToolCall(
            id=f"call_{len(self.turns)}",
            type="function",
            function=ChatMessageToolCallFunction(name=name, arguments=arguments),
        )
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])

    __call__ = generate


def _observations(agent: ToolCallingAgent) -> list[str]:
    return [str(step.observations) for step in agent.memory.steps if hasattr(step, "observations")]


def test_tool_calling_agent_runs_the_tool_through_the_executor(rate):
    hook = RecordingHook()
    tool = smolagents_tool(rate, executor=Executor(hooks=[hook]))
    agent = ToolCallingAgent(
        tools=[tool], model=ScriptedModel("gst_rate", {"item": "tea"}), verbosity_level=0
    )
    assert agent.run("GST on tea?") == "18.0"
    assert hook.phases() == ["before_tool", "after_tool"]
    assert any("18.0" in text for text in _observations(agent))


async def test_tool_calling_agent_inside_a_running_loop(rate):
    tool = smolagents_tool(rate, executor=Executor())
    agent = ToolCallingAgent(
        tools=[tool], model=ScriptedModel("gst_rate", {"item": "kite"}), verbosity_level=0
    )
    assert agent.run("GST on a kite?") == "18.0"
    assert any("no rate for 'kite'" in text for text in _observations(agent))


def test_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "smolagents_tools.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "shipping_rate number 18.0",
        "{'error': {'kind': 'tool-error', 'message': \"no rate for 'mars'\"}}",
        "answer: 18.0",
    ]
