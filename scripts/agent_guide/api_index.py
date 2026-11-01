"""The API index: every public name, its signature and its first docstring line, by module."""

from __future__ import annotations

import enum
import importlib
import inspect
import pkgutil
import re
import types
import typing
from collections.abc import Iterator
from typing import Any

import agent_adapter

ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")
# `agent_adapter.core.messages.Message` -> `Message`: agents must never import core paths.
QUALIFIED = re.compile(r"\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")
MAX_DEFAULT = 48
_EMPTY = inspect.Parameter.empty


def public_modules() -> list[str]:
    """`agent_adapter`, `.errors`, every public subpackage and integration with an `__all__`."""
    found = ["agent_adapter", "agent_adapter.errors"]
    for info in sorted(pkgutil.iter_modules(agent_adapter.__path__), key=lambda i: i.name):
        if info.ispkg and not info.name.startswith("_"):
            found.append(f"agent_adapter.{info.name}")
    integrations = importlib.import_module("agent_adapter.integrations")
    for info in sorted(pkgutil.iter_modules(integrations.__path__), key=lambda i: i.name):
        if not info.name.startswith("_"):
            found.append(f"agent_adapter.integrations.{info.name}")
    return [name for name in found if hasattr(importlib.import_module(name), "__all__")]


def exported(module: str) -> list[str]:
    """The module's `__all__`, sorted, without dunders."""
    names = importlib.import_module(module).__all__
    return sorted(name for name in names if not name.startswith("__"))


def annotation(value: Any) -> str:
    """A short, version-stable rendering of a type annotation."""
    if isinstance(value, str):
        return QUALIFIED.sub("", value)
    if value is None or value is type(None):
        return "None"
    if value is Ellipsis:
        return "..."
    if isinstance(value, list):
        return "[" + ", ".join(annotation(item) for item in value) + "]"
    metadata = getattr(value, "__metadata__", None)
    origin = typing.get_origin(value)
    args = typing.get_args(value)
    if metadata is not None and origin is not None:  # Annotated[T, ...] -> T
        return annotation(args[0])
    if origin is typing.Union or isinstance(value, types.UnionType):
        return " | ".join(annotation(arg) for arg in args)
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(arg) for arg in args) + "]"
    if origin is not None:
        name = getattr(origin, "__name__", None) or QUALIFIED.sub("", str(origin))
        inner = ", ".join(annotation(arg) for arg in args)
        return f"{name}[{inner}]" if args else name
    if isinstance(value, type | typing.TypeVar):
        return value.__name__
    return QUALIFIED.sub("", str(value)).replace("typing.", "")


def default(value: Any) -> str:
    """A default value's repr without memory addresses, shortened when long."""
    if inspect.isroutine(value) or inspect.isclass(value):
        return getattr(value, "__name__", "...")
    text = ADDRESS.sub("", repr(value))
    return text if len(text) <= MAX_DEFAULT else "..."


def signature(obj: Any) -> str:
    """`(params) -> return` from `inspect.signature`, rendered with `annotation`/`default`."""
    sig = inspect.signature(obj)
    parts: list[str] = []
    star = False
    params = [p for p in sig.parameters.values() if not p.name.startswith("_")]  # internals
    for index, param in enumerate(params):
        kind = param.kind
        if kind is param.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        text = param.name
        if kind is param.VAR_POSITIONAL:
            text, star = f"*{param.name}", True
        elif kind is param.VAR_KEYWORD:
            text = f"**{param.name}"
        if param.annotation is not _EMPTY:
            text += f": {annotation(param.annotation)}"
        if param.default is not _EMPTY:
            text += f" = {default(param.default)}" if ":" in text else f"={default(param.default)}"
        parts.append(text)
        following = params[index + 1 :]
        if kind is param.POSITIONAL_ONLY and not any(
            p.kind is param.POSITIONAL_ONLY for p in following
        ):
            parts.append("/")
    out = f"({', '.join(parts)})"
    if sig.return_annotation is not _EMPTY:
        out += f" -> {annotation(sig.return_annotation)}"
    return out


def first_line(obj: Any) -> str:
    """The first docstring sentence of a class, function or module; "" for plain values."""
    if not (inspect.isclass(obj) or inspect.isroutine(obj) or inspect.ismodule(obj)):
        return ""
    doc = obj.__doc__ if inspect.isclass(obj) else inspect.getdoc(obj)
    paragraph: list[str] = []
    for line in inspect.cleandoc(doc).splitlines() if doc else []:
        if not line.strip():
            break
        paragraph.append(line.strip())
    return SENTENCE_END.split(" ".join(paragraph), maxsplit=1)[0]


def _protocol_methods(cls: type) -> str:
    names = sorted(
        name
        for name, value in vars(cls).items()
        if not name.startswith("_") and (inspect.isroutine(value) or isinstance(value, property))
    )
    return ", ".join(names)


def describe(name: str, obj: Any) -> str:
    """One index line body: the name with its signature or kind, without the docstring."""
    if inspect.ismodule(obj):
        return f"`{name}` (module)"
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            members = ", ".join(member.name for member in obj)
            return f"`{name}` (enum: {members})"
        if getattr(obj, "_is_protocol", False):
            methods = _protocol_methods(obj)
            return f"`{name}` (protocol: {methods})" if methods else f"`{name}` (protocol)"
        if issubclass(obj, BaseException):
            return f"`{name}` (error)"
    if _is_type_alias(obj):
        return f"`{name} = {annotation(obj)}`"
    if inspect.isclass(obj) or inspect.isroutine(obj):
        prefix = "async " if inspect.iscoroutinefunction(obj) else ""
        try:
            return f"`{prefix}{name}{signature(obj)}`"
        except (TypeError, ValueError):
            return f"`{name}(...)`"
    if isinstance(obj, str | int | float | bool | bytes | frozenset | tuple):
        return f"`{name} = {default(obj)}`"
    return f"`{name}`: `{type(obj).__name__}` instance"


def _is_type_alias(obj: Any) -> bool:
    """`Literal[...]`, `A | B`, `Callable[...]` and other typing constructs."""
    alias_type = getattr(typing, "TypeAliasType", None)  # Python 3.12+
    if alias_type is not None and isinstance(obj, alias_type):
        return True
    return typing.get_origin(obj) is not None or isinstance(obj, types.UnionType)


def entries(module: str, seen: dict[int, str]) -> Iterator[str]:
    """One bullet per exported name; a name shown in an earlier module points back to it."""
    loaded = importlib.import_module(module)
    for name in exported(module):
        obj = getattr(loaded, name)
        home = seen.get(id(obj))
        real = getattr(obj, "__name__", name)
        if real != name and (inspect.isclass(obj) or inspect.isroutine(obj)):
            where = f" in `{home}`" if home and home != module else ""
            yield f"- `{name}`: alias of `{real}`{where}"
            continue
        if home is not None and (inspect.isclass(obj) or inspect.isroutine(obj)):
            yield f"- `{name}`: see `{home}`"
            continue
        if inspect.isclass(obj) or inspect.isroutine(obj):
            seen[id(obj)] = module
        doc = first_line(obj)
        yield f"- {describe(name, obj)}" + (f": {doc}" if doc else "")


def render() -> str:
    """The `## API index` section."""
    out = [
        "## API index",
        "",
        "Every public name, grouped by the module to import it from. Signatures come from",
        "`inspect.signature`; long default values are shortened to `...`.",
        "",
    ]
    seen: dict[int, str] = {}
    for module in public_modules():
        doc = first_line(importlib.import_module(module))
        out += [f"### `{module}`", ""]
        if doc:
            out += [doc, ""]
        out += [*entries(module, seen), ""]
    return "\n".join(out)
