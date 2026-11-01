"""Turn a callable's signature into the pydantic fields of a tool's argument model."""

from __future__ import annotations

import copy
import functools
import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Union

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolDefinitionError

# Intersection of the OpenAI, Gemini and Anthropic rules for tool names and property keys.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SELF_NAMES = frozenset({"self", "cls"})


@dataclass(frozen=True)
class Options:
    """The parameter-shaping options of build_spec, keyed by Python parameter name."""

    fixed: dict[str, Any]
    hidden: set[str]
    renamed: dict[str, str]


@dataclass(frozen=True)
class Fields:
    """What a signature contributes to a ToolSpec."""

    definitions: dict[str, tuple[Any, FieldInfo]]
    llm_to_python: dict[str, str]
    context_param: str | None
    returns: Any


def unwrap(func: Callable[..., Any]) -> Any:
    """The object whose name, hints and docstring describe `func` (partials unwrapped)."""
    target: Any = func
    while isinstance(target, functools.partial):
        target = target.func
    return target


def partial_keywords(func: Callable[..., Any]) -> dict[str, Any]:
    """Keywords bound by `functools.partial` layers; an outer binding wins over an inner one."""
    layers: list[Mapping[str, Any]] = []
    target: Any = func
    while isinstance(target, functools.partial):
        layers.append(target.keywords)
        target = target.func
    bound: dict[str, Any] = {}
    for keywords in reversed(layers):
        bound.update(keywords)
    return bound


def args_model_name(tool_name: str) -> str:
    """CamelCase model name for a tool's arguments: `get_weather` -> `GetWeatherArgs`."""
    return "".join(part.capitalize() for part in tool_name.split("_") if part) + "Args"


def callable_doc(target: Any) -> str | None:
    """Docstring of a function, or of a callable object's class or `__call__`."""
    doc = inspect.getdoc(target)
    plain = inspect.isfunction(target) or inspect.ismethod(target)
    if doc is None and not plain and callable(target):
        doc = inspect.getdoc(target.__call__)
    return doc


def signature(func: Callable[..., Any], tool_name: str) -> inspect.Signature:
    """`inspect.signature`, with failures reported as ToolDefinitionError."""
    try:
        return inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise ToolDefinitionError(f"tool {tool_name!r}: cannot read the signature ({exc})") from exc


def build_fields(
    func: Callable[..., Any],
    tool_name: str,
    sig: inspect.Signature,
    param_docs: Mapping[str, str],
    options: Options,
) -> Fields:
    """Field definitions for every parameter the LLM supplies, plus context and return type."""
    hints = _type_hints(unwrap(func), tool_name)
    definitions: dict[str, tuple[Any, FieldInfo]] = {}
    llm_to_python: dict[str, str] = {}
    context_param: str | None = None
    for param in sig.parameters.values():
        annotation = hints.get(param.name, param.annotation)
        _check_kind(tool_name, param)
        if is_context(annotation):
            context_param = _context_param(tool_name, param.name, context_param, options)
            continue
        if not _exposed(tool_name, param, options):
            continue
        if annotation is param.empty:
            raise _missing_annotation(tool_name, param.name)
        llm_name = options.renamed.get(param.name, param.name)
        check_arg_name(tool_name, llm_name, param.name, definitions)
        definitions[llm_name] = _definition(param, annotation, param_docs.get(param.name))
        if llm_name != param.name:
            llm_to_python[llm_name] = param.name
    return Fields(definitions, llm_to_python, context_param, _returns(hints))


def check_arg_name(
    tool_name: str, llm_name: Any, param_name: str, taken: Mapping[str, Any]
) -> None:
    """Reject an LLM-facing argument name that a provider or pydantic cannot take."""
    where = f"tool {tool_name!r}: argument {llm_name!r}"
    if llm_name != param_name:
        where += f" (renamed from {param_name!r})"
    if not isinstance(llm_name, str) or not NAME_RE.match(llm_name):
        raise ToolDefinitionError(
            f"{where}: use letters, digits and '_' (max 64, no leading digit)"
        )
    if llm_name.startswith("_"):
        raise ToolDefinitionError(f"{where}: must not start with '_'")
    if llm_name.startswith("model_") or hasattr(BaseModel, llm_name):
        raise ToolDefinitionError(f"{where}: clashes with a pydantic BaseModel attribute")
    if llm_name in taken:
        raise ToolDefinitionError(f"tool {tool_name!r}: argument name {llm_name!r} is used twice")


def is_context(annotation: Any) -> bool:
    """True for `ToolContext`, a subclass, or an Optional/union containing one."""
    if typing.get_origin(annotation) is Annotated:
        return is_context(typing.get_args(annotation)[0])
    if typing.get_origin(annotation) in (Union, types.UnionType):
        return any(is_context(arg) for arg in typing.get_args(annotation))
    return inspect.isclass(annotation) and issubclass(annotation, ToolContext)


def _check_kind(tool_name: str, param: inspect.Parameter) -> None:
    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
        raise ToolDefinitionError(f"tool {tool_name!r}: *args/**kwargs are not supported")
    if param.kind is param.POSITIONAL_ONLY:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: positional-only parameter {param.name!r} is not supported"
        )


def _context_param(tool_name: str, name: str, current: str | None, options: Options) -> str:
    if current is not None:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: more than one ToolContext parameter ({current!r}, {name!r})"
        )
    if name in options.fixed or name in options.hidden or name in options.renamed:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: {name!r} is the ToolContext parameter; "
            "it cannot be fixed, hidden or renamed"
        )
    return name


def _exposed(tool_name: str, param: inspect.Parameter, options: Options) -> bool:
    """False for fixed and hidden parameters, after checking the options make sense."""
    if param.name not in options.fixed and param.name not in options.hidden:
        return True
    if param.name in options.renamed:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: renaming {param.name!r} has no effect; it is fixed or hidden"
        )
    if param.name in options.hidden and param.name not in options.fixed:
        if param.default is param.empty:
            raise ToolDefinitionError(
                f"tool {tool_name!r}: hidden parameter {param.name!r} needs a default"
            )
        if isinstance(param.default, FieldInfo):
            raise ToolDefinitionError(
                f"tool {tool_name!r}: hidden parameter {param.name!r} has a Field() default; "
                "pass its value with fixed= instead"
            )
    return False


def _missing_annotation(tool_name: str, name: str) -> ToolDefinitionError:
    message = f"tool {tool_name!r}: parameter {name!r} needs a type annotation"
    if name in _SELF_NAMES:
        message += (
            "; to expose a method, register the bound method with registry.wrap(obj.method)"
            " instead of decorating it in the class body"
        )
    return ToolDefinitionError(message)


def _definition(
    param: inspect.Parameter, annotation: Any, doc_text: str | None
) -> tuple[Any, FieldInfo]:
    """(annotation, FieldInfo); a Field() default or Annotated description beats the docstring."""
    default = param.default
    if isinstance(default, FieldInfo):
        info = copy.copy(default)
    elif default is param.empty:
        info = Field()
    else:
        info = Field(default)
    if doc_text and info.description is None and not _annotated_description(annotation):
        annotation = Annotated[annotation, Field(description=doc_text)]
    return annotation, info


def _annotated_description(annotation: Any) -> bool:
    if typing.get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(m, FieldInfo) and m.description for m in annotation.__metadata__)


def _returns(hints: Mapping[str, Any]) -> Any:
    returns = hints.get("return")
    return None if returns is type(None) else returns


def _type_hints(target: Any, tool_name: str) -> dict[str, Any]:
    """Resolved hints; an unresolvable return annotation is dropped, not fatal."""
    if not (inspect.isfunction(target) or inspect.ismethod(target)) and callable(target):
        target = target.__call__
    try:
        return typing.get_type_hints(target, include_extras=True)
    except (NameError, TypeError) as exc:
        error = exc
    annotations = dict(getattr(target, "__annotations__", {}))
    if annotations.pop("return", None) is not None:
        params_only = types.SimpleNamespace(
            __annotations__=annotations, __globals__=getattr(target, "__globals__", {})
        )
        try:
            return typing.get_type_hints(params_only, include_extras=True)
        except (NameError, TypeError) as exc:
            error = exc
    raise ToolDefinitionError(f"tool {tool_name!r}: cannot resolve type hints ({error})") from error
