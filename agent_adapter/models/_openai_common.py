"""Request pieces both OpenAI APIs (Chat Completions and Responses) build the same way."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_adapter.core.messages import LLMRequest, OutputSchema, ToolDef
from agent_adapter.schema.transforms import to_openai_parameters


def add_tools(
    kwargs: dict[str, Any],
    request: LLMRequest,
    tool: Callable[[ToolDef], dict[str, Any]],
    choice: Callable[[str], Any],
) -> None:
    """Set `tools`, `tool_choice` and `parallel_tool_calls` when `request` has tools."""
    if not request.tools:
        return
    kwargs["tools"] = [tool(each) for each in request.tools]
    kwargs["tool_choice"] = choice(request.tool_choice)
    if request.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = request.parallel_tool_calls


def schema_format(schema: OutputSchema) -> dict[str, Any]:
    """The `name` / `schema` / `strict` fields of a JSON-schema output format."""
    return {
        "name": schema.name,
        "schema": to_openai_parameters(schema.json_schema, strict=schema.strict),
        "strict": schema.strict,
    }
