"""AutoGen 0.7: registry tools as `BaseTool` subclasses that run through the executor."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import to_jsonable
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.autogen_compress import (
    autogen_model_context,  # registers the format
)
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.schema.transforms import to_plain

TARGET = "autogen"


@dataclass(frozen=True)
class AutoGenCall:
    """One AutoGen tool call, as `ctx.extras["autogen"]` holds it."""

    call_id: str
    cancellation_token: Any  # autogen_core.CancellationToken
    tool: Any = None  # the exported BaseTool


def autogen_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """One `autogen_core.tools.BaseTool` for `spec`."""
    return INTEGRATION.export_tool(spec, executor=executor)


def autogen_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """A `BaseTool` for every registry tool exported to AutoGen."""
    return INTEGRATION.export_tools(registry, executor=executor)


class AutoGenIntegration(IntegrationBase):
    """Exports `BaseTool`s; AutoGen has no per-tool approval gate, so the executor's port decides.

    AutoGen's tool-error channel is an exception from `run_json`: its workbench turns it into
    `ToolResult(is_error=True)` with the message, so adapter errors propagate.
    """

    name = TARGET
    requires = ("autogen_core",)
    extra = "autogen"
    call_id_fields = ("call_id",)

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `BaseTool` for `spec`; its arguments are validated by the runner."""
        check_target(spec, TARGET)
        return _tool_class()(spec, executor, self)


def _parameters(spec: ToolSpec) -> dict[str, Any]:
    """AutoGen's `ParametersSchema` for the tool's plain (ref-free) JSON schema."""
    plain = to_plain(spec.json_schema())
    return {
        "type": "object",
        "properties": plain.get("properties", {}),
        "required": plain.get("required", []),
        "additionalProperties": plain.get("additionalProperties", False),
    }


@functools.cache
def _tool_class() -> Any:
    """The `BaseTool` subclass, built on first use so `autogen_core` is imported lazily."""
    core = _autogen_core()
    base: Any = core.tools.BaseTool
    events = logging.getLogger(core.EVENT_LOGGER_NAME)

    class AdapterTool(base):  # type: ignore[misc]
        """A `BaseTool` for one `ToolSpec`; `run_json` sends the raw arguments to the runner."""

        def __init__(
            self, spec: ToolSpec, executor: ToolRunner | None, integration: AutoGenIntegration
        ) -> None:
            super().__init__(spec.args_model, object, spec.name, spec.description)
            self.spec = spec
            self._executor = executor
            self._integration = integration

        @property
        def schema(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": _parameters(self.spec),
                "strict": False,
            }

        def return_value_as_string(self, value: Any) -> str:
            return value if isinstance(value, str) else json.dumps(to_jsonable(value))

        async def run(self, args: BaseModel, cancellation_token: Any) -> Any:
            arguments = args.model_dump(by_alias=True, exclude_unset=True)
            return await self.run_json(arguments, cancellation_token)

        async def run_json(
            self, args: Mapping[str, Any], cancellation_token: Any, call_id: str | None = None
        ) -> Any:
            native = AutoGenCall(call_id or "", cancellation_token, self)
            ctx = self._integration.bind_context(native)
            runner = runner_for(self._executor)
            with core.trace_tool_span(
                tool_name=self.name, tool_description=self.description, tool_call_id=call_id
            ):
                task = asyncio.ensure_future(
                    runner.run_spec(self.spec, args, ctx, call_id=ctx.call_id or None)
                )
                cancellation_token.link_future(task)
                value = await task
            # Arguments of tools that may need approval never reach a log.
            shown = {} if self.spec.confirm is not False else dict(args)
            event = core.logging.ToolCallEvent(
                tool_name=self.name, arguments=shown, result=self.return_value_as_string(value)
            )
            events.info(event)
            return value

    return AdapterTool


def _autogen_core() -> Any:
    """The `autogen_core` package (typed `Any`: the type check must not depend on the extra)."""
    INTEGRATION.require()
    import autogen_core
    import autogen_core.logging
    import autogen_core.tools

    return autogen_core


INTEGRATION = AutoGenIntegration()


__all__ = [
    "INTEGRATION",
    "AutoGenCall",
    "AutoGenIntegration",
    "autogen_model_context",
    "autogen_tool",
    "autogen_tools",
]
