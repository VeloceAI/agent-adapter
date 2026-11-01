"""Anthropic contract: kit C1-C8 for the host loop over `tool_use` blocks.

The host sends the exported tool dicts, runs each `tool_use` block through the runner with
`bind_context(block)` and answers with `tool_result_blocks`: model-visible errors and pending
approvals become error blocks, every other error propagates.
"""

import dataclasses
import json
from typing import Any

import pytest

from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ErrorInfo, ToolResult
from agent_adapter.exec import Executor
from agent_adapter.integrations.anthropic import (
    INTEGRATION,
    anthropic_tools,
    tool_result_blocks,
)
from agent_adapter.integrations.base import MODEL_VISIBLE_ERRORS, ToolRunner
from agent_adapter.schema.transforms import to_anthropic_input_schema
from tests.integrations.kit import contract_tests

pytestmark = pytest.mark.contract


@dataclasses.dataclass(frozen=True)
class Exported:
    spec: ToolSpec
    runner: ToolRunner
    tool: dict[str, Any]


async def run_block(block: dict[str, Any], spec: ToolSpec, runner: ToolRunner) -> ToolResult:
    """What a host loop does with one `tool_use` block."""
    ctx = INTEGRATION.bind_context(block)
    try:
        value = await runner.run_spec(spec, block["input"], ctx, call_id=ctx.call_id)
    except (*MODEL_VISIBLE_ERRORS, ApprovalPending) as exc:  # anything else propagates
        status = "pending" if isinstance(exc, ApprovalPending) else "error"
        return ToolResult(
            ctx.call_id, spec.name, status, error=ErrorInfo.from_exception(exc), exception=exc
        )
    return ToolResult(ctx.call_id, spec.name, "ok", value=value)


class AnthropicHarness:
    name = "anthropic"
    edits_supported = True  # the executor applies edited args
    approval = "port"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        return Exported(spec, executor, INTEGRATION.export_tool(spec, executor=executor))

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return anthropic_tools(registry)

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Any:
        block = {"type": "tool_use", "id": call_id, "name": native.spec.name, "input": args}
        [result_block] = tool_result_blocks([await run_block(block, native.spec, native.runner)])
        assert result_block["tool_use_id"] == call_id
        return result_block

    def value(self, raw: dict[str, Any]) -> Any:
        try:
            return json.loads(raw["content"])
        except json.JSONDecodeError:
            return raw["content"]

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or raw["is_error"]

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="toolu_approval")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        tool = native.tool if isinstance(native, Exported) else native
        return tool["name"], tool["description"], tool["input_schema"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_anthropic_input_schema(spec.json_schema())


globals().update((item.__name__, item) for item in contract_tests(AnthropicHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AnthropicHarness, runner_factory=Executor)
)


def test_pending_and_info_less_results_render_as_errors():
    pending = ApprovalPending("apr_1", tool="t")
    blocks = tool_result_blocks(
        [
            ToolResult("t1", "t", "pending", error=ErrorInfo.from_exception(pending)),
            ToolResult("t2", "t", "error"),
        ]
    )
    assert blocks[0]["content"] == "approval required: apr_1" and blocks[0]["is_error"]
    assert json.loads(blocks[1]["content"])["error"]["kind"] == "internal-error"
