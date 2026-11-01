"""Render tools for the Anthropic Messages API as plain dicts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ErrorInfo, ToolResult, to_jsonable
from agent_adapter.integrations.base import IntegrationBase, ToolRunner, _error_payload
from agent_adapter.schema.transforms import to_anthropic_input_schema

TARGET = "anthropic"


def anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    """One tool in Anthropic Messages format."""
    check_target(spec, TARGET)
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": to_anthropic_input_schema(spec.json_schema()),
    }


def anthropic_tools(registry: Registry) -> list[dict[str, Any]]:
    """Every registry tool exported to Anthropic, in Messages format."""
    return [anthropic_tool(spec) for spec in registry.for_target(TARGET)]


def tool_result_blocks(results: Sequence[ToolResult]) -> list[dict[str, Any]]:
    """`tool_result` content blocks answering the model's `tool_use` blocks, in order."""
    return [_result_block(result) for result in results]


def _result_block(result: ToolResult) -> dict[str, Any]:
    if result.ok:
        value = result.value
        content = value if isinstance(value, str) else json.dumps(to_jsonable(value))
    elif result.status == "pending":
        approval_id = result.error.details.get("approval_id", "") if result.error else ""
        content = f"approval required: {approval_id}"
    else:
        info = result.error or ErrorInfo(
            kind="internal-error", message=f"tool {result.name!r} failed"
        )
        content = json.dumps(_error_payload(info))
    return {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": content,
        "is_error": not result.ok,
    }


class AnthropicIntegration(IntegrationBase):
    """Exports Messages API tool dicts; the host runs the model's `tool_use` blocks."""

    name = TARGET
    call_id_fields = ("id",)  # ToolUseBlock.id

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> dict[str, Any]:
        """One tool dict; `executor` is unused."""
        return anthropic_tool(spec)


INTEGRATION = AnthropicIntegration()


__all__ = [
    "INTEGRATION",
    "AnthropicIntegration",
    "anthropic_tool",
    "anthropic_tools",
    "tool_result_blocks",
]
