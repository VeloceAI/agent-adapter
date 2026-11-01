"""AutoGen contract: kit C1-C8 through `StaticWorkbench.call_tool`, plus an offline agent run."""

import asyncio
import json
import logging
import runpy
import sys
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

autogen_core = pytest.importorskip("autogen_core")
pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.messages import ToolCallExecutionEvent  # noqa: E402
from autogen_core import EVENT_LOGGER_NAME, CancellationToken, FunctionCall  # noqa: E402
from autogen_core.models import (  # noqa: E402
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import BaseTool, StaticWorkbench, ToolResult  # noqa: E402

from agent_adapter.core.context import ToolContext  # noqa: E402
from agent_adapter.core.errors import (  # noqa: E402
    MissingDependency,
    ToolArgsInvalid,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.imported import spec_from_schema  # noqa: E402
from agent_adapter.core.registry import Registry  # noqa: E402
from agent_adapter.core.tool import ToolSpec, build_spec  # noqa: E402
from agent_adapter.core.types import ApprovalDecision  # noqa: E402
from agent_adapter.exec import Executor  # noqa: E402
from agent_adapter.exec.defaults import default_executor, set_default_executor  # noqa: E402
from agent_adapter.integrations import autogen as autogen_integration  # noqa: E402
from agent_adapter.integrations.autogen import (  # noqa: E402
    INTEGRATION,
    AutoGenCall,
    autogen_tool,
    autogen_tools,
)
from agent_adapter.integrations.base import ToolRunner  # noqa: E402
from agent_adapter.schema.transforms import to_plain  # noqa: E402
from tests.fakes.approval import ScriptedApproval  # noqa: E402
from tests.fakes.hooks import RecordingHook  # noqa: E402
from tests.integrations.kit import FakeRunner, contract_tests  # noqa: E402
from tests.integrations.sample_tools import gst_rate  # noqa: E402

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class AutoGenHarness:
    """Drives `StaticWorkbench.call_tool`, the path `AssistantAgent` uses for its tools."""

    name = "autogen"
    edits_supported = True  # the executor's port applies edited args
    approval = "port"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return autogen_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return autogen_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        workbench = StaticWorkbench([native])
        return await workbench.call_tool(native.name, args, CancellationToken(), call_id=call_id)

    def value(self, raw: Any) -> Any:
        text = raw.result[0].content
        try:
            return json.loads(text)
        except ValueError:
            return text

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, ToolResult) and raw.is_error

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_kit_approval")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        schema = native.schema
        return schema["name"], schema["description"], schema["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        plain = to_plain(spec.json_schema())
        return {
            "type": "object",
            "properties": plain.get("properties", {}),
            "required": plain.get("required", []),
            "additionalProperties": plain.get("additionalProperties", False),
        }


globals().update((item.__name__, item) for item in contract_tests(AutoGenHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AutoGenHarness, runner_factory=Executor)
)


async def echo_call(tag: str, ctx: ToolContext) -> str:
    """Report the call the context holds."""
    call = ctx.extras["autogen"]
    return f"{tag}:{ctx.call_id}:{type(call).__name__}:{call.tool.name}"


@pytest.fixture
def rate() -> ToolSpec:
    return build_spec(gst_rate)


def test_export_builds_a_base_tool_over_the_args_model(rate):
    tool = autogen_tool(rate)
    assert isinstance(tool, BaseTool)
    assert (tool.name, tool.description) == ("gst_rate", "GST rate for an item.")
    assert tool.args_type() is rate.args_model
    assert tool.return_type() is object
    assert tool.schema == {
        "name": "gst_rate",
        "description": "GST rate for an item.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "What is sold."},
                "region": {"type": "string", "description": "Tax region.", "default": "IN"},
            },
            "required": ["item"],
            "additionalProperties": False,
        },
        "strict": False,
    }


def test_imported_tools_export_their_own_schema():
    schema = {
        "type": "object",
        "properties": {"max-results": {"type": "integer"}},
        "additionalProperties": True,
    }

    async def search(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    spec = spec_from_schema("search", "Search.", schema, search)
    parameters = autogen_tool(spec).schema["parameters"]
    assert parameters["properties"] == {"max-results": {"type": "integer"}}
    assert parameters["required"] == []
    assert parameters["additionalProperties"] is True


async def test_run_json_returns_values_and_raises_adapter_errors(rate):
    tool = autogen_tool(rate, executor=FakeRunner())
    token = CancellationToken()
    assert await tool.run_json({"item": "tea"}, token) == 18.0
    with pytest.raises(ToolError, match="no rate for 'kite'"):
        await tool.run_json({"item": "kite"}, token)
    with pytest.raises(ToolArgsInvalid) as invalid:
        await tool.run_json({"item": "tea", "region": 7123}, token)
    assert "7123" not in str(invalid.value)


async def test_run_takes_a_validated_args_model(rate):
    tool = autogen_tool(rate, executor=FakeRunner())
    args = rate.args_model.model_validate({"item": "tea"})
    assert await tool.run(args, CancellationToken()) == 18.0


async def test_workbench_reports_failures_with_is_error(rate):
    workbench = StaticWorkbench([autogen_tool(rate, executor=Executor())])
    ok = await workbench.call_tool("gst_rate", {"item": "tea"}, call_id="call_1")
    failed = await workbench.call_tool("gst_rate", {"item": "kite"}, call_id="call_2")
    assert (ok.is_error, ok.result[0].content) == (False, "18.0")
    assert (failed.is_error, failed.result[0].content) == (True, "no rate for 'kite'")


def test_return_value_as_string_is_json_except_for_text(rate):
    tool = autogen_tool(rate)
    assert tool.return_value_as_string("plain") == "plain"
    assert tool.return_value_as_string({"a": [1, 2]}) == '{"a": [1, 2]}'
    assert tool.return_value_as_string(18.0) == "18.0"


async def test_context_holds_the_autogen_call():
    tool = autogen_tool(build_spec(echo_call), executor=FakeRunner())
    value = await tool.run_json({"tag": "t"}, CancellationToken(), call_id="call_x")
    assert value == "t:call_x:AutoGenCall:echo_call"
    assert INTEGRATION.bind_context(AutoGenCall(call_id="", cancellation_token=None)).call_id == ""


async def test_cancellation_token_cancels_the_tool():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def wait(label: str) -> str:
        """Wait until cancelled."""
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return label

    token = CancellationToken()
    tool = autogen_tool(build_spec(wait), executor=Executor())
    task = asyncio.ensure_future(tool.run_json({"label": "x"}, token))
    await started.wait()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_tool_call_event_hides_arguments_of_confirm_tools(caplog):
    remove = build_spec(gst_rate, name="remove_rate", confirm=True)
    approve = ScriptedApproval([ApprovalDecision(approved=True)] * 2)
    executor = Executor(approval=approve)
    caplog.set_level(logging.INFO, logger=EVENT_LOGGER_NAME)
    await autogen_tool(build_spec(gst_rate), executor=executor).run_json(
        {"item": "tea"}, CancellationToken()
    )
    await autogen_tool(remove, executor=executor).run_json({"item": "tea"}, CancellationToken())
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == EVENT_LOGGER_NAME]
    assert [(e["tool_name"], e["arguments"], e["result"]) for e in events] == [
        ("gst_rate", {"item": "tea"}, "18.0"),
        ("remove_rate", {}, "18.0"),
    ]


async def test_default_executor_is_used_at_call_time(rate):
    previous = default_executor()
    hook = RecordingHook()
    tool = autogen_tool(rate)
    set_default_executor(Executor(hooks=[hook]))
    try:
        assert await tool.run_json({"item": "tea"}, CancellationToken()) == 18.0
    finally:
        set_default_executor(previous)
    assert hook.phases() == ["before_tool", "after_tool"]


def test_single_export_honours_targets():
    with pytest.raises(ToolUsageError, match="not exported to 'autogen'"):
        autogen_tool(build_spec(gst_rate, targets="openai"))


def test_missing_autogen_raises_missing_dependency(monkeypatch, rate):
    autogen_integration._tool_class.cache_clear()
    monkeypatch.setitem(sys.modules, "autogen_core", None)
    try:
        with pytest.raises(MissingDependency, match=r"agent-adapter\[autogen\]"):
            autogen_tool(rate)
    finally:
        autogen_integration._tool_class.cache_clear()


def test_integration_has_no_native_approval_gate():
    assert INTEGRATION.bind_approval(object()) is None
    assert (INTEGRATION.name, INTEGRATION.requires) == ("autogen", ("autogen_core",))


class ScriptedClient(ChatCompletionClient):
    """An offline chat client: each `create` returns the next scripted content."""

    def __init__(self, turns: list[str | list[FunctionCall]]) -> None:
        self.turns = list(turns)
        self.usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages: Sequence[LLMMessage], **kwargs: Any) -> CreateResult:
        content = self.turns.pop(0)
        reason = "function_calls" if isinstance(content, list) else "stop"
        return CreateResult(finish_reason=reason, content=content, usage=self.usage, cached=False)

    def create_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self.usage

    def total_usage(self) -> RequestUsage:
        return self.usage

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 1000

    @property
    def capabilities(self) -> Mapping[str, Any]:  # type: ignore[override]
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=False,
        )


async def test_assistant_agent_runs_tools_with_native_call_ids(rate):
    hook = RecordingHook()
    executor = Executor(hooks=[hook])
    calls = [
        FunctionCall(id="call_a", name="gst_rate", arguments=json.dumps({"item": "tea"})),
        FunctionCall(id="call_b", name="echo_call", arguments=json.dumps({"tag": "t"})),
        FunctionCall(id="call_c", name="gst_rate", arguments=json.dumps({"item": "kite"})),
    ]
    tools = [
        autogen_tool(rate, executor=executor),
        autogen_tool(build_spec(echo_call), executor=executor),
    ]
    agent = AssistantAgent("agent", model_client=ScriptedClient([calls]), tools=tools)
    result = await agent.run(task="GST on tea?")
    executed = next(m for m in result.messages if isinstance(m, ToolCallExecutionEvent))
    assert [(r.call_id, r.content, r.is_error) for r in executed.content] == [
        ("call_a", "18.0", False),
        ("call_b", "t:call_b:AutoGenCall:echo_call", False),
        ("call_c", "no rate for 'kite'", True),
    ]
    seen = sorted(args[0].ctx.call_id for phase, args in hook.calls if phase == "before_tool")
    assert seen == ["call_a", "call_b", "call_c"]


def test_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "autogen_tools.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "shipping_rate 18.0",
        "is_error=True no rate for 'mars'",
        "answer: 18.0",
    ]
