"""LangChain: registry tools as `StructuredTool`s, and LangChain tools imported as `ToolSpec`s."""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolArgsInvalid, ToolDefinitionError, ToolError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations.base import MODEL_VISIBLE_ERRORS, IntegrationBase, ToolRunner
from agent_adapter.runtime.current import current_context
from agent_adapter.runtime.sync import run_sync

logger = logging.getLogger(__name__)

TARGET = "langchain"
CONFIG_PARAM = "_aa_config"  # receives LangChain's RunnableConfig; never a tool argument
_MAX_NAME = 64

# The id of the LangChain ToolCall being run; set by the exported tool's `arun` / `run`.
_TOOL_CALL_ID: ContextVar[str | None] = ContextVar("agent_adapter_langchain_call_id", default=None)


def structured_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """`spec` as a LangChain `StructuredTool` that runs through the executor.

    The model sees `spec.json_schema()` unchanged and validation is the spec's own.
    Model-visible errors raise `ToolException` (`handle_tool_error=True` turns it
    into an error `ToolMessage`); other exceptions propagate. Sync `invoke` uses `run_sync`.
    The run id comes from `current_context()` inside an adapter run, else from the
    RunnableConfig: `configurable["run_id"]`, else `configurable["thread_id"]` (LangGraph's);
    without either, every call is a fresh run (a deferred approval cannot resume).

    Raises:
        ToolUsageError: `spec.targets` excludes langchain.
        MissingDependency: langchain-core is not installed.
    """
    check_target(spec, TARGET)
    INTEGRATION.require()
    runner = lazy_runner(executor)

    from langchain_core.runnables import RunnableConfig

    async def coroutine(_aa_config: Any = None, **kwargs: Any) -> Any:
        return await _run_exported(spec, runner, kwargs, _aa_config)

    def func(_aa_config: Any = None, **kwargs: Any) -> Any:
        return run_sync(lambda: _run_exported(spec, runner, kwargs, _aa_config))

    # LangChain passes the RunnableConfig to a parameter annotated with exactly that class.
    coroutine.__annotations__ = func.__annotations__ = {CONFIG_PARAM: RunnableConfig}

    return _tool_class()(
        name=spec.name,
        description=spec.description,
        args_schema=spec.json_schema(),
        coroutine=coroutine,
        func=func,
        handle_tool_error=True,
        tags=sorted(spec.tags),
        metadata={"agent_adapter_kind": spec.kind},
    )


def structured_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """`structured_tool` for every registry tool exported to LangChain."""
    return [structured_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]


async def _run_exported(
    spec: ToolSpec, runner: ToolRunner, args: dict[str, Any], config: Any
) -> Any:
    from langchain_core.tools import ToolException

    native = {"name": spec.name, "args": args, "id": _TOOL_CALL_ID.get(), "type": "tool_call"}
    ctx = _context(INTEGRATION.bind_context(native), config)
    try:
        return await runner.run_spec(spec, args, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        logger.debug("langchain tool %s: %s", spec.name, type(exc).__name__)
        raise ToolException(str(exc)) from exc


def _context(bound: ToolContext, config: Any) -> ToolContext:
    """The call's context: joins the current adapter run, else the config's run, else none."""
    outer = current_context()
    if outer is not None:
        return outer.child(call_id=bound.call_id, extras={**outer.extras, **bound.extras})
    configurable = (config or {}).get("configurable") or {}
    thread_id = str(configurable.get("thread_id") or "")
    run_id = str(configurable.get("run_id") or thread_id)
    return dataclasses.replace(bound, run_id=run_id, thread_id=thread_id)


@functools.cache
def _tool_class() -> type[Any]:
    """A `StructuredTool` subclass that exposes the ToolCall id to the running coroutine."""
    from langchain_core.tools import StructuredTool

    base: Any = StructuredTool  # Any: the type check must not depend on the extra

    class AdapterTool(base):  # type: ignore[misc]
        """A registry tool; `tool_call_id` (from a ToolCall input) reaches `ctx.call_id`."""

        async def arun(
            self, tool_input: Any, *args: Any, tool_call_id: str | None = None, **kwargs: Any
        ) -> Any:
            token = _TOOL_CALL_ID.set(tool_call_id)
            try:
                return await super().arun(tool_input, *args, tool_call_id=tool_call_id, **kwargs)
            finally:
                _TOOL_CALL_ID.reset(token)

        def run(
            self, tool_input: Any, *args: Any, tool_call_id: str | None = None, **kwargs: Any
        ) -> Any:
            token = _TOOL_CALL_ID.set(tool_call_id)
            try:
                return super().run(tool_input, *args, tool_call_id=tool_call_id, **kwargs)
            finally:
                _TOOL_CALL_ID.reset(token)

    return AdapterTool


def import_langchain(
    registry: Registry, tools: Iterable[Any], *, prefix: str = "", **policy: Any
) -> list[ToolSpec]:
    """Add LangChain `BaseTool`s to `registry` (all or nothing) and return their specs.

    The schema is what LangChain sends a model (`convert_to_openai_function`), so
    `InjectedToolArg` parameters stay out; the importer does not supply them. Names are
    sanitised to `[A-Za-z0-9_]`. A `ToolException` becomes `ToolError`; arguments the tool's
    own argument model rejects become `ToolArgsInvalid` (checked before the call).

    Raises:
        ToolDefinitionError: an item is not a `BaseTool`, or a bad name or option.
        DuplicateTool: a name is already registered.
        MissingDependency: langchain-core is not installed.
    """
    INTEGRATION.require()
    specs = [_import_one(tool, prefix, policy) for tool in tools]
    return registry.add_many(specs)


def _import_one(tool: Any, prefix: str, policy: Mapping[str, Any]) -> ToolSpec:
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_function

    if not isinstance(tool, BaseTool):
        raise ToolDefinitionError(
            f"import_langchain needs LangChain BaseTool objects, got {tool!r}"
        )
    name = _tool_name(prefix, tool.name)
    parameters = convert_to_openai_function(tool)["parameters"]
    options = {"kind": "langchain", "tags": {"langchain", *(tool.tags or ())}, **policy}

    async def call(**kwargs: Any) -> Any:
        return await _run_imported(tool, name, kwargs)

    return spec_from_schema(name, tool.description or tool.name, parameters, call, **options)


async def _run_imported(tool: Any, name: str, args: dict[str, Any]) -> Any:
    from langchain_core.tools import ToolException

    _check_arguments(tool, name, args)
    try:
        return await tool.ainvoke(args)
    except ToolException as exc:
        logger.debug("imported langchain tool %s raised ToolException", name)
        raise ToolError(str(exc), tool=name) from exc


def _check_arguments(tool: Any, name: str, args: dict[str, Any]) -> None:
    """Validate with the tool's own argument model: only argument errors are ToolArgsInvalid."""
    schema = tool.tool_call_schema  # without InjectedToolArg fields
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return  # a JSON-schema args_schema: LangChain does not validate it either
    try:
        schema.model_validate(args)
    except ValidationError as exc:
        logger.debug("imported langchain tool %s rejected its arguments", name)
        raise ToolArgsInvalid(name, exc.errors(include_input=False)) from exc


def _tool_name(prefix: str, name: str) -> str:
    """`prefix + name` with every character outside `[A-Za-z0-9_]` replaced by `_`."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", f"{prefix}{name}")
    if not clean or clean[0].isdigit():
        clean = f"t_{clean}"
    return clean[:_MAX_NAME]


class LangChainIntegration(IntegrationBase):
    """Exports `StructuredTool`s; the executor's approval port decides `confirm` tools."""

    name = TARGET
    requires = ("langchain_core",)
    extra = "langchain"
    call_id_fields = ("id",)  # ToolCall["id"]

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `StructuredTool` (see `structured_tool`)."""
        return structured_tool(spec, executor=executor)


INTEGRATION = LangChainIntegration()


__all__ = [
    "INTEGRATION",
    "LangChainIntegration",
    "import_langchain",
    "structured_tool",
    "structured_tools",
]
