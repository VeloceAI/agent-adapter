"""MCP: serve registry tools to any MCP client, and import tools from MCP servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.integrations.mcp._handlers import raw_context
from agent_adapter.integrations.mcp.annotations import (
    APPROVAL_ID_META,
    CALL_ID_META,
    ERROR_KIND_META,
    RUN_ID_META,
    output_schema_for,
    tool_annotations,
)
from agent_adapter.integrations.mcp.client import import_mcp, mcp_client
from agent_adapter.integrations.mcp.oauth import (
    MemoryTokenStorage,
    client_metadata_document,
    oauth_client,
)
from agent_adapter.integrations.mcp.server import (
    TARGET,
    mcp_server,
    mcp_tool,
    streamable_http_app,
)
from agent_adapter.integrations.mcp.tasks import (
    TASK_ID_META,
    TASKS_EXTENSION,
    MemoryTaskStore,
    RedisTaskStore,
    TaskOptions,
    TaskStore,
)

if TYPE_CHECKING:
    import mcp.types as t

__all__ = [
    "APPROVAL_ID_META",
    "CALL_ID_META",
    "ERROR_KIND_META",
    "INTEGRATION",
    "RUN_ID_META",
    "TASKS_EXTENSION",
    "TASK_ID_META",
    "MCPIntegration",
    "MemoryTaskStore",
    "MemoryTokenStorage",
    "RedisTaskStore",
    "TaskOptions",
    "TaskStore",
    "client_metadata_document",
    "import_mcp",
    "mcp_client",
    "mcp_server",
    "mcp_tool",
    "oauth_client",
    "output_schema_for",
    "streamable_http_app",
    "tool_annotations",
]


class MCPIntegration(IntegrationBase):
    """Exports `mcp.types.Tool` definitions; `mcp_server` runs the calls."""

    name = TARGET
    requires = ("mcp",)
    extra = "mcp"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> t.Tool:
        """The MCP tool definition; `executor` is used by `mcp_server`, not here."""
        check_target(spec, TARGET)
        return mcp_tool(spec)

    def bind_context(self, native: Any) -> ToolContext:
        """A context with the client's raw, unscoped `_meta` keys (for inspection only).

        Never build a server on these ids: `mcp_server` scopes them by principal and honours
        a call key only to resume a pending result.
        """
        return raw_context(native)


INTEGRATION = MCPIntegration()
