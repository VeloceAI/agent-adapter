"""The integration protocol: what every framework bridge provides, and a helper base class."""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalDenied,
    GuardrailBlocked,
    MissingDependency,
    RunLimitExceeded,
    ToolArgsInvalid,
    ToolError,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ErrorInfo

# Errors a framework without a native tool-error channel returns to the model as
# `model_visible_error(exc)`; every other exception propagates.
MODEL_VISIBLE_ERRORS: tuple[type[AdapterError], ...] = (
    ToolArgsInvalid,
    ToolError,
    ApprovalDenied,
    GuardrailBlocked,
    RunLimitExceeded,
)


class ToolRunner(Protocol):
    """Runs one tool call through the full policy pipeline (an `Executor` is one)."""

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any: ...


@runtime_checkable
class Integration(Protocol):
    """A framework bridge; `name` is the canonical target name and the entry-point name.

    The attributes are read-only properties here so that class attributes satisfy them.
    """

    @property
    def name(self) -> str: ...

    @property
    def requires(self) -> tuple[str, ...]: ...  # import names that must be importable

    @property
    def extra(self) -> str: ...  # pip extra named in MissingDependency messages

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any: ...

    def bind_context(self, native: Any) -> ToolContext: ...

    def bind_approval(self, port: ApprovalPort) -> Any: ...  # native gate, or None


@runtime_checkable
class FlowCompiler(Protocol):
    """An integration that also compiles flows; `flow` is a `FlowSpec` (typed Any here)."""

    def compile_flow(
        self, flow: Any, *, executor: ToolRunner | None = None, **options: Any
    ) -> Any: ...


class IntegrationBase(ABC):
    """Helper base for built-in and third-party integrations; subclasses add `export_tool`."""

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()
    extra: ClassVar[str] = ""
    # Attribute or key names on the native call object holding its call id, first non-empty wins.
    call_id_fields: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """The framework's native tool object for `spec`."""

    def export_tools(self, registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
        """Native tools for every registry tool exported to this integration's target."""
        return [
            self.export_tool(spec, executor=executor) for spec in registry.for_target(self.name)
        ]

    def bind_context(self, native: Any) -> ToolContext:
        """A context holding `native` in `extras[name]` and its call id in `call_id`."""
        return ToolContext(
            call_id=_native_call_id(native, self.call_id_fields), extras={self.name: native}
        )

    def bind_approval(self, port: ApprovalPort) -> Any:
        """None: the executor's approval port decides."""
        return None

    def require(self) -> None:
        """Raise MissingDependency naming the extra when a required module is missing."""
        _check_requires(self)

    async def _run_bound(
        self,
        native: Any,
        spec: ToolSpec,
        args: Mapping[str, Any] | None,
        runner: ToolRunner,
        *,
        visible: Callable[[BaseException], Any] | None = None,
    ) -> Any:
        """Run one native call through `runner`; MODEL_VISIBLE_ERRORS become the tool result.

        For frameworks without a native tool-error channel; `visible` formats the error
        (default `model_visible_error`). Every other exception, and control flow such as
        `asyncio.CancelledError`, propagates.
        """
        ctx = self.bind_context(native)
        try:
            return await runner.run_spec(spec, args, ctx, call_id=ctx.call_id or None)
        except MODEL_VISIBLE_ERRORS as exc:
            return (visible or model_visible_error)(exc)


def _native_call_id(native: Any, fields: tuple[str, ...]) -> str:
    """The first non-empty string among `native`'s attributes or keys named in `fields`."""
    for name in fields:
        value = native.get(name) if isinstance(native, Mapping) else getattr(native, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _check_requires(integration: Integration) -> None:
    """Raise MissingDependency when any of `integration.requires` cannot be imported."""
    missing = [name for name in integration.requires if not _importable(name)]
    if not missing:
        return
    hint = (
        f'pip install "agent-adapter[{integration.extra}]"'
        if integration.extra
        else f"install {', '.join(missing)}"
    )
    raise MissingDependency(f"integration {integration.name!r} needs {', '.join(missing)}: {hint}")


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False
    except ValueError:  # already imported with __spec__ = None
        return name in sys.modules


def model_visible_error(exc: BaseException) -> dict[str, Any]:
    """`{"error": {"kind", "message"}}` for a tool result; foreign errors show only their type."""
    return _error_payload(ErrorInfo.from_exception(exc))


def _error_payload(info: ErrorInfo) -> dict[str, Any]:
    """The model-visible part of `info`: kind and message, never details or input values."""
    return {"error": {"kind": info.kind, "message": info.message}}


__all__ = [
    "MODEL_VISIBLE_ERRORS",
    "FlowCompiler",
    "Integration",
    "IntegrationBase",
    "ToolRunner",
    "model_visible_error",
]
