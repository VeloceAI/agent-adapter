"""Registry: the one place an app defines the tools it exposes."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar, overload

from agent_adapter.core.errors import (
    DuplicateTool,
    SchemaNotPortable,
    ToolNotFound,
    ToolUsageError,
)
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.schema.portable import check_portable

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
SPEC_ATTR = "__agent_tool__"
# Fix-it notes, shown in tracebacks.
RENAME_IT = (
    "pass name= to register it under another name, use a separate Registry(), or "
    "registry.remove(name) first (e.g. a notebook cell that runs twice)"
)
TYPE_IT = (
    "give each flagged field a concrete type (str, int, list[str], a TypedDict or a "
    "pydantic model), or pass portable=False when every target provider accepts it"
)


def exported_to(spec: ToolSpec, target: str) -> bool:
    """True when `spec.targets` is None or names `target`."""
    return spec.targets is None or target in spec.targets


def check_target(spec: ToolSpec, target: str) -> None:
    """Raise ToolUsageError when `spec` is not exported to `target`."""
    if not exported_to(spec, target):
        raise ToolUsageError(
            f"tool {spec.name!r} is not exported to {target!r} "
            f"(targets: {sorted(spec.targets or ())})"
        )


class Registry:
    """Holds ToolSpecs by name; executors and exports read from here."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def __reduce__(self) -> tuple[Any, ...]:
        """Refuse pickling with a clear error (it holds locks and loop-bound state).

        Raises:
            ToolUsageError: always.
        """
        raise ToolUsageError(
            f"{type(self).__name__} cannot be pickled: build it on the worker with a "
            "module-level functools.cache factory (see docs/backends/pipelines.md)"
        )

    @overload
    def tool(self, func: F, /) -> F: ...

    @overload
    def tool(self, func: None = None, /, **options: Any) -> Callable[[F], F]: ...

    def tool(self, func: F | None = None, /, **options: Any) -> F | Callable[[F], F]:
        """Decorator: register a function as a tool and return it unchanged."""

        def register(target: F) -> F:
            spec = self.add(build_spec(target, **options))
            if inspect.isfunction(target):
                setattr(target, SPEC_ATTR, spec)
            return target

        return register(func) if func is not None else register

    def wrap(self, func: Callable[..., Any], **options: Any) -> ToolSpec:
        """Register existing code as a tool without editing it."""
        return self.add(build_spec(func, **options))

    def add(self, spec: ToolSpec) -> ToolSpec:
        """Register a spec after checking its name and schema portability.

        Raises:
            DuplicateTool: the name is taken.
            SchemaNotPortable: the schema would fail on some provider.
        """
        self._check(spec, set(self._tools))
        self._insert(spec)
        logger.debug("registered tool %s (kind=%s)", spec.name, spec.kind)
        return spec

    def add_many(self, specs: Iterable[ToolSpec]) -> list[ToolSpec]:
        """Register several specs, all or nothing.

        Raises:
            DuplicateTool: a name is taken or repeated; nothing is registered.
            SchemaNotPortable: a schema would fail on some provider; nothing is registered.
        """
        batch = list(specs)
        taken = set(self._tools)
        for spec in batch:
            self._check(spec, taken)
            taken.add(spec.name)
        for spec in batch:
            self._insert(spec)
        logger.debug("registered %d tools: %s", len(batch), [spec.name for spec in batch])
        return batch

    @staticmethod
    def _check(spec: ToolSpec, taken: set[str]) -> None:
        if spec.name in taken:
            duplicate = DuplicateTool(f"tool {spec.name!r} is already registered")
            duplicate.add_note(RENAME_IT)
            raise duplicate
        if spec.portable:
            problems = check_portable(spec.json_schema())
            if problems:
                error = SchemaNotPortable(problems, tool=spec.name)
                error.add_note(TYPE_IT)
                raise error

    def remove(self, name: str) -> ToolSpec:
        """Unregister `name` and return its spec.

        Raises:
            ToolNotFound: no tool has that name.
        """
        spec = self.get(name)
        del self._tools[name]
        logger.debug("removed tool %s", name)
        return spec

    def _insert(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Return the spec for `name`.

        Raises:
            ToolNotFound: no tool has that name.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"no tool named {name!r}; known: {sorted(self._tools)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Tool names in registration order."""
        return list(self._tools)

    def subset(
        self,
        *,
        names: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> Registry:
        """A new registry holding only tools that match every given filter.

        Raises:
            ToolNotFound: a requested name is not registered.
        """
        wanted_names = set(names) if names is not None else None
        wanted_tags = set(tags) if tags is not None else None
        wanted_kinds = set(kinds) if kinds is not None else None
        if wanted_names:
            missing = wanted_names - set(self._tools)
            if missing:
                raise ToolNotFound(f"no tools named {sorted(missing)}")
        sub = Registry()
        for spec in self:
            if wanted_names is not None and spec.name not in wanted_names:
                continue
            if wanted_tags is not None and not spec.tags & wanted_tags:
                continue
            if wanted_kinds is not None and spec.kind not in wanted_kinds:
                continue
            sub._insert(spec)
        return sub

    def for_target(self, target: str) -> Registry:
        """A new registry with the tools exported to `target` (`targets` None or listing it)."""
        sub = Registry()
        for spec in self:
            if exported_to(spec, target):
                sub._insert(spec)
        return sub

    def catalogue(self) -> str:
        """Markdown tool list for planner prompts, generated so it never drifts from code."""
        blocks: list[str] = []
        for spec in self:
            schema = spec.json_schema()
            required = set(schema.get("required", []))
            lines = [f"### {spec.name}", spec.description]
            if spec.prompt_hint:
                lines.append(f"When to use: {spec.prompt_hint}")
            properties = schema.get("properties", {})
            if properties:
                lines.append("Arguments:")
                for arg, prop in properties.items():
                    flag = "required" if arg in required else "optional"
                    text = prop.get("description")
                    lines.append(f"- `{arg}` ({flag})" + (f": {text}" if text else ""))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
