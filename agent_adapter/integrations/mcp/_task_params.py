"""MCP tasks request plumbing: the `tasks/*` params models and client capability checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from agent_adapter.integrations.mcp.tasks import TASKS_EXTENSION

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext


def params_types() -> tuple[type[Any], type[Any]]:
    """The `tasks/get`/`tasks/cancel` and the `tasks/update` request params models."""
    from mcp.types import RequestParams

    class TaskIdParams(RequestParams):  # type: ignore[misc,unused-ignore]
        model_config = ConfigDict(extra="ignore")
        task_id: str = Field(alias="taskId", min_length=1, max_length=256)

    class UpdateTaskParams(TaskIdParams):
        input_responses: dict[str, Any] = Field(alias="inputResponses")

    return TaskIdParams, UpdateTaskParams


def client_accepts_tasks(ctx: ServerRequestContext[Any, Any]) -> bool:
    """True when this 2026-07-28 request declares the tasks extension."""
    from mcp.types.version import MODERN_PROTOCOL_VERSIONS

    if ctx.protocol_version not in MODERN_PROTOCOL_VERSIONS:
        return False
    capabilities = ctx.session.client_capabilities
    extensions = capabilities.extensions if capabilities is not None else None
    return isinstance(extensions, Mapping) and TASKS_EXTENSION in extensions


def accepts_url(ctx: ServerRequestContext[Any, Any]) -> bool:
    """True when the client takes URL-mode elicitation."""
    capabilities = ctx.session.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    return elicitation is not None and elicitation.url is not None
