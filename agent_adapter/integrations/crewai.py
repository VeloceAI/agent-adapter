"""CrewAI: registry tools as `BaseTool`s; `_run` bridges to the executor with `run_sync`."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

from pydantic import PrivateAttr

from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.runtime.sync import run_sync

if TYPE_CHECKING:
    from crewai.tools import BaseTool

TARGET = "crewai"


def crewai_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> BaseTool:
    """A `BaseTool` with `args_schema=spec.args_model`: `_arun` is async, `_run` uses `run_sync`.

    CrewAI validates the arguments against `args_schema` first (a `ValueError` on failure).
    Its result cache is off, so every call reaches the executor. Approval comes from the
    executor's port (CrewAI has no gate of its own). Tool errors, denials, guardrails and
    run limits come back as `{"error": {...}}`; other errors propagate.

    Raises:
        MissingDependency: `crewai` is not installed.
        ToolUsageError: `spec.targets` excludes `crewai`.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    tool = _tool_class()(
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_model,
        cache_function=_never_cache,
    )
    tool._aa_spec = spec
    tool._aa_runner = lazy_runner(executor)
    return cast("BaseTool", tool)  # built from `_tool_class()`, typed `type[Any]` for lazy import


def crewai_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[BaseTool]:
    """`crewai_tool` for every registry tool exported to `crewai`."""
    return [crewai_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]


def _never_cache(_args: Any = None, _result: Any = None) -> bool:
    """CrewAI `cache_function`: never reuse a result, so policy and approval run every time."""
    return False


@functools.cache
def _tool_class() -> type[Any]:
    """The one `BaseTool` subclass, created on first use (CrewAI imports slowly)."""
    from crewai.tools import BaseTool

    class AgentAdapterTool(BaseTool):  # type: ignore[misc, unused-ignore]  # Any without crewai
        """A registry tool run through the Agent Adapter executor."""

        _aa_spec: ToolSpec = PrivateAttr()
        _aa_runner: ToolRunner = PrivateAttr()

        def _run(self, **arguments: Any) -> Any:
            return run_sync(lambda: self._arun(**arguments))

        async def _arun(self, **arguments: Any) -> Any:
            return await INTEGRATION._run_bound(self, self._aa_spec, arguments, self._aa_runner)

    return AgentAdapterTool


class CrewAIIntegration(IntegrationBase):
    """Exports `BaseTool`s; CrewAI has no call id at the tool boundary."""

    name = TARGET
    requires = ("crewai",)
    extra = "crewai"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> BaseTool:
        """`crewai_tool(spec, executor=executor)`."""
        return crewai_tool(spec, executor=executor)


INTEGRATION = CrewAIIntegration()


__all__ = ["INTEGRATION", "CrewAIIntegration", "crewai_tool", "crewai_tools"]
