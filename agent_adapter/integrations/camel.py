"""CAMEL-AI: ToolSpecs as `FunctionTool`s (the executor's port approves), a compression format.

CAMEL hands a tool only its arguments (no call id) and has no approval gate of its own,
so nothing CAMEL does can approve a `confirm` tool: the executor asks its own port.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Mapping
from typing import Any

from agent_adapter.context.formats import register_format
from agent_adapter.context.formats._items import as_text
from agent_adapter.context.formats._shared import (
    Converter,
    Group,
    Present,
    arguments_of,
    not_a_message,
)
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, strict_downgraded
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import check_framework, run_bound
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.runtime.sync import run_sync
from agent_adapter.schema.transforms import to_openai_parameters

TARGET = "camel"
_SUPPORTED = ((0, 2, 90), (0, 3, 0))  # the `camel` extra's range


class CamelIntegration(IntegrationBase):
    """Exports `FunctionTool`s; `confirm` tools are decided by the executor's port."""

    name = TARGET
    requires = ("camel",)
    extra = "camel"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `FunctionTool` running `spec` through the executor."""
        return camel_tool(spec, executor=executor)


INTEGRATION = CamelIntegration()


def camel_tool(spec: ToolSpec, *, executor: ToolRunner | None = None, strict: bool = True) -> Any:
    """A CAMEL `FunctionTool` for `spec`, with the OpenAI-transformed schema.

    `strict` follows the OpenAI integration's rule (a spec that cannot be strict is not).

    Raises:
        MissingDependency: CAMEL is not installed.
        IntegrationError: the installed CAMEL is outside the extra's range or lacks
            `FunctionTool.async_call`.
    """
    INTEGRATION.require()
    return _tool_class()(spec, lazy_runner(executor), strict=strict)


def camel_tools(
    registry: Registry, *, executor: ToolRunner | None = None, strict: bool = True
) -> list[Any]:
    """CAMEL `FunctionTool`s for every registry tool exported to this target."""
    INTEGRATION.require()
    return [
        camel_tool(spec, executor=executor, strict=strict) for spec in registry.for_target(TARGET)
    ]


def openai_tool_schema(spec: ToolSpec, *, strict: bool = True) -> dict[str, Any]:
    """The Chat Completions tool schema CAMEL sends to the model for `spec`."""
    strict = strict and not strict_downgraded(spec)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": to_openai_parameters(spec.json_schema(), strict=strict),
            "strict": strict,
        },
    }


@functools.cache
def _tool_class() -> type[Any]:
    """The `FunctionTool` subclass, built on first use (the framework is imported lazily)."""
    import camel
    from camel.toolkits import FunctionTool
    from jsonschema.validators import Draft202012Validator  # type: ignore[import-untyped, unused-ignore]

    check_framework(
        TARGET,
        str(getattr(camel, "__version__", "0")),
        _SUPPORTED,
        {"FunctionTool.async_call": callable(getattr(FunctionTool, "async_call", None))},
    )

    class AdapterFunctionTool(FunctionTool):  # type: ignore[misc, unused-ignore]
        """A `FunctionTool` that runs a `ToolSpec` through a `ToolRunner`."""

        def __init__(self, spec: ToolSpec, runner: ToolRunner, *, strict: bool) -> None:
            def call(**arguments: Any) -> Any:
                return run_sync(lambda: self.async_call(**arguments))

            call.__name__ = spec.name
            call.__doc__ = spec.description
            super().__init__(call, openai_tool_schema=openai_tool_schema(spec, strict=strict))
            self.spec = spec
            self.runner = runner

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Run the call synchronously (CAMEL's `step`); arguments are keyword-only."""
            return self.func(**_keywords(args, kwargs))

        async def async_call(self, *args: Any, **kwargs: Any) -> Any:
            """Run the call (CAMEL's `astep`); MODEL_VISIBLE_ERRORS come back as results."""
            ctx = INTEGRATION.bind_context(None)
            return await run_bound(self.runner, self.spec, _keywords(args, kwargs), ctx)

        @staticmethod
        def validate_openai_tool_schema(openai_tool_schema: dict[str, Any]) -> None:
            """Check the parameters are a JSON Schema (CAMEL warns on undescribed fields)."""
            Draft202012Validator.check_schema(openai_tool_schema["function"]["parameters"])

    return AdapterFunctionTool


def _keywords(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The call's arguments; CAMEL always passes keywords, positional ones are refused."""
    if args:
        raise ToolUsageError("agent_adapter CAMEL tools take keyword arguments only")
    return dict(kwargs)


FORMAT = "camel"
_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
_ROLES |= {"function": "tool", "tool": "tool"}
_STEP = 1e-6  # seconds between a new record and its neighbour; CAMEL orders records by time


def is_camel_record(item: Any) -> bool:
    """True for a CAMEL `MemoryRecord`, found by class, without importing CAMEL."""
    return any(
        cls.__name__ == "MemoryRecord" and cls.__module__.startswith("camel.")
        for cls in type(item).__mro__
    )


class CamelConverter(Converter):
    """`MemoryRecord` lists (`[c.memory_record for c in memory.retrieve()]`).

    Tool calls come from `meta_dict["tool_calls"]` or a `FunctionCallingMessage`; results
    are `FunctionCallingMessage`s with the function backend role. CAMEL orders records by
    time, so new records, and records a step moved, get a timestamp between their neighbours.
    """

    name = FORMAT

    def __init__(self, items: Any) -> None:
        super().__init__(items)
        self._rendered: set[int] = set()

    def groups(self) -> list[Group]:
        groups: list[Group] = []
        for index, record in enumerate(self.items):
            role = _role(record) if is_camel_record(record) else None
            if role is None:
                raise not_a_message(FORMAT, index, record)
            groups.append(Group((index,), (_message(role, record.message),)))
        return groups

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        record = self.items[group.natives[0]]
        part = present[0]
        if part.unchanged:
            return [record]
        message = record.message
        is_result = part.message.role == "tool" and getattr(message, "result", None) is not None
        field = "result" if is_result else "content"
        changed = dataclasses.replace(message, **{field: part.message.content})
        return [record.model_copy(update={"message": changed})]

    def render(self, messages: list[Message]) -> list[Any]:
        out = [self._record(message) for message in messages]
        self._rendered.update(id(record) for record in out)
        return out

    def finish(self, natives: list[Any]) -> list[Any]:
        """The list in CAMEL's order: its first system record, then the rest by time.

        Kept records keep their timestamps. A new record goes right after the kept record
        before it, with a timestamp between that record's and the next one's.
        """
        kept = [r for r in natives if id(r) not in self._rendered]
        after: dict[int | None, list[Any]] = {}  # id(kept record before) -> new records
        before: int | None = None
        for record in natives:
            if id(record) in self._rendered:
                after.setdefault(before, []).append(record)
            else:
                before = id(record)
        system = next((r for r in kept if _role(r) == "system"), None)
        rest = sorted((r for r in kept if r is not system), key=lambda r: r.timestamp)
        ordered = [system, *rest] if system is not None else rest
        out = _stamped(after.get(None, []), None, ordered[0].timestamp if ordered else None)
        for index, record in enumerate(ordered):
            following = ordered[index + 1].timestamp if index + 1 < len(ordered) else None
            out += [record, *_stamped(after.get(id(record), []), record.timestamp, following)]
        return out

    def _record(self, message: Message) -> Any:
        """A new `MemoryRecord` for a message a step created, named like its role's records."""
        INTEGRATION.require()
        from camel.memories import MemoryRecord
        from camel.messages import BaseMessage, FunctionCallingMessage
        from camel.types import OpenAIBackendRole, RoleType

        like = next((r.message for r in self.items if _role(r) == message.role), None)
        role_name = like.role_name if like is not None else message.role
        role_type = like.role_type if like is not None else RoleType.DEFAULT
        backend = OpenAIBackendRole("function" if message.role == "tool" else message.role)
        native: Any
        if message.role == "tool":
            native = FunctionCallingMessage(
                role_name=role_name,
                role_type=role_type,
                meta_dict=None,
                content="",
                func_name=message.name,
                result=message.content,
                tool_call_id=message.tool_call_id,
            )
        else:
            calls = [_openai_call(call) for call in message.tool_calls]
            meta = {"tool_calls": calls} if calls else None
            native = BaseMessage(role_name, role_type, meta, message.content)
        agent_id = self.items[0].agent_id if self.items else ""
        return MemoryRecord(message=native, role_at_backend=backend, agent_id=agent_id)


def _stamped(records: list[Any], low: float | None, high: float | None) -> list[Any]:
    """`records` with timestamps rising strictly between `low` and `high` (either may be open)."""
    out: list[Any] = []
    for number, record in enumerate(records, start=1):
        if high is not None and (low is None or low >= high):
            low = high - _STEP * (len(records) - number + 2)
        stamp = record.timestamp if low is None else low + _STEP
        if high is not None and low is not None:
            stamp = (low + high) / 2
        out.append(record.model_copy(update={"timestamp": stamp}))
        low = stamp
    return out


def _role(record: Any) -> str | None:
    backend = record.role_at_backend
    return _ROLES.get(str(getattr(backend, "value", backend)))


def _message(role: str, message: Any) -> Message:
    """The `Message` for one record's CAMEL message."""
    if role == "tool":
        result = getattr(message, "result", None)
        masked = bool(getattr(message, "mask_output", False))
        text = "[MASKED]" if masked else as_text(message.content if result is None else result)
        return Message(
            role="tool",
            tool_call_id=str(getattr(message, "tool_call_id", None) or "null"),
            name=str(getattr(message, "func_name", None) or ""),
            content=text,
        )
    if role != "assistant":
        return Message(role=role, content=as_text(message.content))
    calls = [_meta_call(call) for call in (message.meta_dict or {}).get("tool_calls") or ()]
    if not calls and getattr(message, "func_name", None) and getattr(message, "args", None):
        calls = [
            ToolCall(
                id=str(message.tool_call_id or "null"),
                name=str(message.func_name),
                arguments=arguments_of(message.args),
            )
        ]
    return Message(role="assistant", content=as_text(message.content), tool_calls=tuple(calls))


def _meta_call(call: Mapping[str, Any]) -> ToolCall:
    """A call from `meta_dict["tool_calls"]` (OpenAI chat format)."""
    function = call.get("function") or {}
    return ToolCall(
        id=str(call.get("id") or "null"),
        name=str(function.get("name") or ""),
        arguments=arguments_of(function.get("arguments")),
    )


def _openai_call(call: ToolCall) -> dict[str, Any]:
    arguments = json.dumps(dict(call.arguments), ensure_ascii=False)
    function = {"name": call.name, "arguments": arguments}
    return {"id": call.id, "type": "function", "function": function}


register_format(FORMAT, CamelConverter, detect=is_camel_record)


__all__ = [
    "INTEGRATION",
    "CamelIntegration",
    "camel_tool",
    "camel_tools",
    "openai_tool_schema",
]
