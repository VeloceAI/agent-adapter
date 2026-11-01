"""Render tools for the OpenAI Chat Completions and Responses APIs as plain dicts."""

from __future__ import annotations

from typing import Any, Literal

from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec, strict_downgraded
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.schema.transforms import to_openai_parameters

TARGET = "openai"


def chat_tool(spec: ToolSpec, *, strict: bool = True) -> dict[str, Any]:
    """One tool in Chat Completions format."""
    strict = _strict(spec, strict)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": to_openai_parameters(spec.json_schema(), strict=strict),
            "strict": strict,
        },
    }


def chat_tools(registry: Registry, *, strict: bool = True) -> list[dict[str, Any]]:
    """Every registry tool exported to OpenAI, in Chat Completions format."""
    return [chat_tool(spec, strict=strict) for spec in registry.for_target(TARGET)]


def responses_tool(spec: ToolSpec, *, strict: bool = True) -> dict[str, Any]:
    """One tool in Responses API format (flat, no `function` wrapper)."""
    strict = _strict(spec, strict)
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": to_openai_parameters(spec.json_schema(), strict=strict),
        "strict": strict,
    }


def responses_tools(registry: Registry, *, strict: bool = True) -> list[dict[str, Any]]:
    """Every registry tool exported to OpenAI, in Responses API format."""
    return [responses_tool(spec, strict=strict) for spec in registry.for_target(TARGET)]


def _strict(spec: ToolSpec, strict: bool) -> bool:
    """Check the target; apply the strict-downgrade rule of `strict_downgraded`."""
    check_target(spec, TARGET)
    return strict and not strict_downgraded(spec)


class OpenAIIntegration(IntegrationBase):
    """Exports Chat Completions (default) or Responses tool dicts; the host runs the calls."""

    name = TARGET
    # Responses function_call items carry `call_id` (and an `fc_` item `id`); Chat tool calls `id`.
    call_id_fields = ("call_id", "id")

    def __init__(self, *, api: Literal["chat", "responses"] = "chat", strict: bool = True) -> None:
        if api not in ("chat", "responses"):
            raise ValueError(f"api must be 'chat' or 'responses', got {api!r}")
        self.api: Literal["chat", "responses"] = api
        self.strict = strict

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> dict[str, Any]:
        """One tool dict; `executor` is unused because the host executes the model's call."""
        render = chat_tool if self.api == "chat" else responses_tool
        return render(spec, strict=self.strict)


INTEGRATION = OpenAIIntegration()


__all__ = [
    "INTEGRATION",
    "OpenAIIntegration",
    "chat_tool",
    "chat_tools",
    "responses_tool",
    "responses_tools",
]
