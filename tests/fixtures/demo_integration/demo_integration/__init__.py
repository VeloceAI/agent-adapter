"""A sample third-party integration for a tiny "demo" framework whose tools are async callables."""

# --8<-- [start:integration]
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.base import IntegrationBase, ToolRunner


@dataclass(frozen=True)
class DemoCall:
    """The framework's record of one tool call; `approval` is set by its own approval gate."""

    id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    approval: ApprovalDecision | None = None


@dataclass(frozen=True)
class DemoTool:
    """The framework's tool: what the model sees, plus an async call."""

    name: str
    description: str
    parameters: dict[str, Any]
    spec: ToolSpec = field(repr=False)
    runner: ToolRunner = field(repr=False)
    integration: DemoIntegration = field(repr=False)

    async def __call__(self, call: DemoCall) -> Any:
        # bind_context -> runner.run_spec(..., call_id=call.id); the framework has no
        # tool-error channel, so model-visible errors come back as the result.
        return await self.integration._run_bound(call, self.spec, call.arguments, self.runner)


class DemoIntegration(IntegrationBase):
    """Exports registry tools as `DemoTool`s that run through the given runner."""

    name = "demo"  # the entry-point name
    call_id_fields = ("id",)  # DemoCall.id becomes ctx.call_id

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> DemoTool:
        """One `DemoTool` for `spec`."""
        runner = runner_for(executor)  # the given runner, else the default Executor
        return DemoTool(spec.name, spec.description, spec.json_schema(), spec, runner, self)

    def bind_context(self, native: Any) -> ToolContext:
        """The base context plus the framework's own approval, so the executor skips its port."""
        ctx = super().bind_context(native)
        ctx.approval = getattr(native, "approval", None)
        return ctx


INTEGRATION = DemoIntegration()
# --8<-- [end:integration]


class NotAnIntegration:
    """Registered on purpose: discovery must skip it (no protocol methods)."""
