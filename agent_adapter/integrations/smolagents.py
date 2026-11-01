"""smolagents: registry tools as `smolagents.Tool`s whose sync `forward` runs the executor."""

from __future__ import annotations

import functools
import json
import keyword
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.errors import IntegrationError, ToolArgsInvalid
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.base import (
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.integrations.smolagents_compress import (
    smolagents_compressed_model,  # registers the format
)
from agent_adapter.runtime.sync import run_sync
from agent_adapter.schema.transforms import to_plain

TARGET = "smolagents"

# smolagents.tools.AUTHORIZED_TYPES minus the media types, which JSON schemas never name.
_JSON_TYPES = frozenset({"string", "boolean", "integer", "number", "array", "object", "null"})


def smolagents_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """One `smolagents.Tool` for `spec`; its sync `forward` runs the executor via `run_sync`."""
    return INTEGRATION.export_tool(spec, executor=executor)


def smolagents_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """A `smolagents.Tool` for every registry tool exported to smolagents."""
    return INTEGRATION.export_tools(registry, executor=executor)


class SmolagentsIntegration(IntegrationBase):
    """Exports `smolagents.Tool`s; no native approval gate, so the executor's port decides."""

    name = TARGET
    requires = ("smolagents",)
    extra = "smolagents"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `smolagents.Tool` for `spec` (`ctx.extras["smolagents"]` is that tool).

        Raises:
            IntegrationError: the tool name is not a Python identifier (smolagents needs one).
        """
        check_target(spec, TARGET)
        if not spec.name.isidentifier() or keyword.iskeyword(spec.name):
            raise IntegrationError(
                f"smolagents needs tool names that are Python identifiers, got {spec.name!r}; "
                "export it under another name (build_spec(..., name=...))",
                integration=TARGET,
            )
        return _tool_class()(spec, executor, self)


def _visible(tool: Any, exc: BaseException) -> Any:
    """`model_visible_error(exc)`, as JSON text when the tool's `output_type` is "string"."""
    error = model_visible_error(exc)
    return json.dumps(error) if tool.output_type == "string" else error


def _arguments(tool: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Positional args mapped onto `tool.inputs` in order, plus the keyword args.

    Raises:
        ToolArgsInvalid: too many positional args, or one input given both ways.
    """
    names = list(tool.inputs)
    if len(args) > len(names):
        msg = f"takes at most {len(names)} positional arguments"
        raise ToolArgsInvalid(tool.name, [{"type": "too_many_arguments", "loc": [], "msg": msg}])
    positional = dict(zip(names, args, strict=False))
    clashes = [name for name in positional if name in kwargs]
    if clashes:
        raise ToolArgsInvalid(
            tool.name,
            [
                {
                    "type": "multiple_values",
                    "loc": [name],
                    "msg": "given both positionally and by name",
                }
                for name in clashes
            ],
        )
    return {**positional, **kwargs}


def _inputs(spec: ToolSpec) -> dict[str, dict[str, Any]]:
    """`Tool.inputs`: each property's plain schema with an authorized `type` and `nullable`."""
    plain = to_plain(spec.json_schema())
    required = set(plain.get("required", ()))
    return {
        key: _entry(prop, required=key in required)
        for key, prop in plain.get("properties", {}).items()
    }


def _entry(prop: Mapping[str, Any], *, required: bool) -> dict[str, Any]:
    """One input entry; `nullable` marks an argument the model may omit or send as null."""
    entry = dict(prop)
    branches = entry.get("anyOf")
    if isinstance(branches, list):
        typed = [b for b in branches if isinstance(b, dict) and isinstance(b.get("type"), str)]
        non_null = [b for b in typed if b["type"] != "null"]
        if len(typed) < len(branches):
            del entry["anyOf"]  # get_tool_json_schema reads every branch's "type"
            entry["type"] = "any"
        elif len(non_null) == 1:  # Optional[X]: inline X so nested schemas survive
            del entry["anyOf"]
            entry = {**non_null[0], **entry}
            required = False
        else:
            entry["type"] = _authorized([b["type"] for b in non_null]) if non_null else "null"
            required = required and len(non_null) == len(typed)
    kind = entry["type"] = _authorized(entry.get("type"))
    entry.setdefault("description", "")
    if not required or kind == "null" or (isinstance(kind, list) and "null" in kind):
        entry["nullable"] = True
    return entry


def _authorized(kind: Any) -> str | list[str]:
    """`kind` when smolagents accepts it; a list keeps its known members; else `"any"`."""
    if isinstance(kind, str):
        return kind if kind in _JSON_TYPES or kind == "any" else "any"
    if isinstance(kind, list):
        known = [k for k in kind if k in _JSON_TYPES]
        if len(known) > 1:
            return known
        return known[0] if known else "any"
    return "any"


def _output_type(spec: ToolSpec) -> str:
    """The JSON type of the result, or `"any"` when it is not a single known type."""
    kind = (spec.result_schema() or {}).get("type")
    return kind if isinstance(kind, str) and kind in _JSON_TYPES else "any"


@functools.cache
def _tool_class() -> Any:
    """The `smolagents.Tool` subclass, built on first use so `smolagents` is imported lazily."""
    base: Any = _smolagents().Tool

    class AdapterTool(base):  # type: ignore[misc]
        """A `smolagents.Tool` for one `ToolSpec`; `forward` blocks until the runner finishes."""

        skip_forward_signature_validation = True  # forward takes **kwargs, not the inputs

        def __init__(
            self, spec: ToolSpec, executor: ToolRunner | None, integration: SmolagentsIntegration
        ) -> None:
            self.spec = spec
            self.name = spec.name
            self.description = spec.description
            self.inputs = _inputs(spec)
            self.output_type = _output_type(spec)
            self._executor = executor
            self._integration = integration
            super().__init__()

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            try:
                arguments = _arguments(self, args, kwargs)
            except ToolArgsInvalid as exc:
                return _visible(self, exc)
            runner = runner_for(self._executor)
            visible = functools.partial(_visible, self)
            work = self._integration._run_bound(self, self.spec, arguments, runner, visible=visible)
            return run_sync(work)

    return AdapterTool


def _smolagents() -> Any:
    """The `smolagents` module (typed `Any`: the type check must not depend on the extra)."""
    INTEGRATION.require()
    import smolagents

    return smolagents


INTEGRATION = SmolagentsIntegration()


__all__ = [
    "INTEGRATION",
    "SmolagentsIntegration",
    "smolagents_compressed_model",
    "smolagents_tool",
    "smolagents_tools",
]
