"""What every AG-UI module shares: the integration object, the version guard, lazy SDK access."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING, Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._native_gate import check_framework
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

if TYPE_CHECKING:
    from ag_ui.core import Tool

TARGET = "ag_ui"
SUPPORTED = ((1, 0), (2, 0))  # the `ag-ui` extra's range: the frozen 1.x schema
METADATA_KEY = "agent_adapter"  # our key inside AG-UI metadata objects ("ag-ui" is reserved)


@functools.cache
def core() -> ModuleType:
    """`ag_ui.core`, imported on first use after the version check.

    Raises:
        MissingDependency: `ag-ui-protocol` is not installed.
        IntegrationError: its version is outside the supported range.
    """
    INTEGRATION.require()
    import ag_ui.core as agui_core

    check_framework(
        TARGET,
        str(agui_core.PROTOCOL_VERSION),
        SUPPORTED,
        {"RunFinishedInterruptOutcome": hasattr(agui_core, "RunFinishedInterruptOutcome")},
    )
    return agui_core  # type: ignore[no-any-return,unused-ignore]  # Any without the SDK


def encode(event: Any) -> str:
    """One AG-UI event as an SSE frame (`data: <json>\\n\\n`), as the SDK's `EventEncoder` does."""
    core()
    from ag_ui.encoder import EventEncoder

    return str(EventEncoder().encode(event))


class AGUIIntegration(IntegrationBase):
    """Describes registry tools as AG-UI `Tool`s; runs and approvals go through `AGUIBridge`."""

    name = TARGET
    requires = ("ag_ui",)
    extra = "ag-ui"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Tool:
        """An AG-UI `Tool` (name, description, JSON schema); AG-UI runs tools server side."""
        tool: Tool = core().Tool(
            name=spec.name, description=spec.description, parameters=spec.json_schema()
        )
        return tool


INTEGRATION = AGUIIntegration()
