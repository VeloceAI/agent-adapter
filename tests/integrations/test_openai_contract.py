"""OpenAI contract: kit C1-C8 for the host loop over Chat Completions and Responses tool calls.

OpenAI has no framework of its own. The production path: render the tools, take the tool
call the model returns (named after the rendered tool), look the tool up in the registry,
run it through the runner with `bind_context(call)`, and send model-visible errors back as
the tool result (`model_visible_error`); every other error propagates.
"""

import dataclasses
import json
from typing import Any

import pytest

from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import MODEL_VISIBLE_ERRORS, ToolRunner, model_visible_error
from agent_adapter.integrations.openai import (
    INTEGRATION,
    OpenAIIntegration,
    chat_tools,
    responses_tools,
)
from agent_adapter.schema.transforms import to_openai_parameters
from tests.integrations.kit import contract_tests

pytestmark = pytest.mark.contract


async def dispatch(call: Any, spec: ToolSpec, arguments: dict[str, Any], runner: ToolRunner) -> Any:
    """What a host does with one model tool call."""
    ctx = INTEGRATION.bind_context(call)
    try:
        return await runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        return model_visible_error(exc)


@dataclasses.dataclass(frozen=True)
class Exported:
    spec: ToolSpec
    runner: ToolRunner
    tool: dict[str, Any]


class ChatHarness:
    """Chat Completions: `message.tool_calls[i]` with JSON-string arguments."""

    name = "openai"
    edits_supported = True  # the executor applies edited args
    approval = "port"
    integration = INTEGRATION

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        return Exported(spec, executor, self.integration.export_tool(spec, executor=executor))

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return chat_tools(registry)

    def native_call(self, native: Exported, args: dict[str, Any], call_id: str) -> Any:
        """The model's Chat tool call for the rendered tool."""
        function = {"name": native.tool["function"]["name"], "arguments": json.dumps(args)}
        return {"id": call_id, "type": "function", "function": function}

    def parse(self, call: Any) -> tuple[str, dict[str, Any]]:
        return call["function"]["name"], json.loads(call["function"]["arguments"])

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Any:
        registry = Registry()
        registry.add(native.spec)
        call = self.native_call(native, args, call_id)
        name, arguments = self.parse(call)
        value = await dispatch(
            call, registry.for_target("openai").get(name), arguments, native.runner
        )
        return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(value)}

    def value(self, raw: dict[str, Any]) -> Any:
        return json.loads(raw["content"])

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception):
            return True
        value = self.value(raw)
        return isinstance(value, dict) and set(value) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_openai_approval")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        tool = native.tool if isinstance(native, Exported) else native
        function = tool.get("function", tool)  # Chat nests it; Responses is flat
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_openai_parameters(spec.json_schema(), strict=True)


class ResponsesHarness(ChatHarness):
    """Responses API: `function_call` items carry `call_id` and an `fc_` item id."""

    integration = OpenAIIntegration(api="responses")

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return responses_tools(registry)

    def native_call(self, native: Exported, args: dict[str, Any], call_id: str) -> Any:
        """The model's Responses `function_call` item for the rendered tool."""
        return {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": native.tool["name"],
            "arguments": json.dumps(args),
        }

    def parse(self, call: Any) -> tuple[str, dict[str, Any]]:
        return call["name"], json.loads(call["arguments"])


globals().update((item.__name__, item) for item in contract_tests(ChatHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(ChatHarness, runner_factory=Executor)
)
globals().update((f"{item.__name__}_responses", item) for item in contract_tests(ResponsesHarness))
globals().update(
    (f"{item.__name__}_responses_executor", item)
    for item in contract_tests(ResponsesHarness, runner_factory=Executor)
)


def lookup(section: str, year: int | None = None) -> str:
    """Look up a section."""
    return section


def test_unknown_api_is_rejected():
    with pytest.raises(ValueError, match="api must be"):
        OpenAIIntegration(api="assistants")  # type: ignore[arg-type]


def test_non_strict_export_and_the_strict_downgrade():
    tool = OpenAIIntegration(strict=False).export_tool(build_spec(lookup))
    assert tool["function"]["strict"] is False
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    imported = spec_from_schema("imported", "Imported.", schema, lookup, portable=False)
    assert INTEGRATION.export_tool(imported)["function"]["strict"] is False
