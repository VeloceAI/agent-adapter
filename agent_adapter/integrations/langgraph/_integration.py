"""The LangGraph integration object (`INTEGRATION`), shared by every module of the package."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

TARGET = "langgraph"


class LangGraphIntegration(IntegrationBase):
    """Exports tool nodes and compiles flows; approvals pause the graph with `interrupt()`."""

    name = TARGET
    requires = ("langgraph", "langchain_core")
    extra = "langgraph"
    call_id_fields = ("id",)  # a LangChain ToolCall: {"name", "args", "id"}

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """A `tool_node` over this one tool."""
        from agent_adapter.integrations.langgraph.tools import tool_node

        registry = Registry()
        registry.add(spec)
        return tool_node(registry, executor=executor)

    def compile_flow(self, flow: Any, *, executor: Any = None, **options: Any) -> Any:
        """`compile_flow(flow, executor=executor, checkpoint_key=...)`."""
        from agent_adapter.integrations.langgraph.compile import compile_flow

        key = options.pop("checkpoint_key", None)
        if options:
            from agent_adapter.core.errors import ToolUsageError

            raise ToolUsageError(
                f"compile_flow takes only checkpoint_key for langgraph: {sorted(options)}"
            )
        return compile_flow(flow, executor=executor, checkpoint_key=key)


INTEGRATION = LangGraphIntegration()
