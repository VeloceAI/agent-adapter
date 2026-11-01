"""DSPy: registry tools as `dspy.Tool`s that run through the executor."""

from __future__ import annotations

import functools
from typing import Any

from agent_adapter.core.errors import IntegrationError
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.runtime.sync import run_sync
from agent_adapter.schema.transforms import to_plain

TARGET = "dspy"

_PYTHON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def dspy_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """One `dspy.Tool` for `spec`; `acall` is native, sync `__call__` goes through `run_sync`."""
    return INTEGRATION.export_tool(spec, executor=executor)


def dspy_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """A `dspy.Tool` for every registry tool exported to DSPy."""
    return INTEGRATION.export_tools(registry, executor=executor)


class DSPyIntegration(IntegrationBase):
    """Exports `dspy.Tool`s; DSPy has no approval gate, so the executor's port decides."""

    name = TARGET
    requires = ("dspy",)
    extra = "dspy"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `dspy.Tool` for `spec` (`ctx.extras["dspy"]` is that tool)."""
        check_target(spec, TARGET)
        tool_class = _tool_class()
        properties: dict[str, Any] = to_plain(spec.json_schema()).get("properties", {})
        tool: Any = None

        async def call(**kwargs: Any) -> Any:
            return await self._run_bound(tool, spec, kwargs, runner_for(executor))

        tool = tool_class(
            call,
            name=spec.name,
            desc=spec.description,
            args=properties,
            arg_types={key: _python_type(prop) for key, prop in properties.items()},
        )
        return tool


def _python_type(prop: dict[str, Any]) -> Any:
    """The Python type DSPy shows for a JSON-schema property; `Any` when it is not one type."""
    kind = prop.get("type")
    return _PYTHON_TYPES.get(kind, Any) if isinstance(kind, str) else Any


@functools.cache
def _tool_class() -> Any:
    """The `dspy.Tool` subclass, built on first use so `dspy` is imported lazily.

    Raises:
        IntegrationError: `dspy.Tool` lacks the validation hook this subclass overrides.
    """
    base: Any = _dspy().Tool
    if not callable(getattr(base, "_validate_and_parse_args", None)):
        # Without this hook DSPy would validate first, quoting input values to the model.
        raise IntegrationError(
            "this dspy version has no dspy.Tool._validate_and_parse_args to override; "
            "the dspy integration supports dspy>=3.4,<4",
            integration=TARGET,
        )

    class AdapterTool(base):  # type: ignore[misc]
        """A `dspy.Tool` whose arguments are validated by the tool's own model, in the runner."""

        def _validate_and_parse_args(self, **kwargs: Any) -> dict[str, Any]:
            # DSPy's own JSON-schema check would raise ValueError with the input values and
            # skip hooks; the runner reports `ToolArgsInvalid` as a model-visible error.
            return kwargs

        def __call__(self, **kwargs: Any) -> Any:
            return run_sync(self.acall(**kwargs))

    return AdapterTool


def _dspy() -> Any:
    """The `dspy` module (typed `Any`: the type check must not depend on the extra)."""
    INTEGRATION.require()
    import dspy

    return dspy


INTEGRATION = DSPyIntegration()


__all__ = ["INTEGRATION", "DSPyIntegration", "dspy_tool", "dspy_tools"]
