"""ToolSpec: a callable plus everything needed to expose it as an LLM tool."""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic.errors import PydanticUndefinedAnnotation

from agent_adapter.core._attempt import ATTEMPT, Attempt
from agent_adapter.core._docstring import parse_docstring
from agent_adapter.core._nulls import drop_strict_nulls, imported_kwargs
from agent_adapter.core._policy import validate_policy
from agent_adapter.core._signature import (
    NAME_RE,
    Options,
    args_model_name,
    build_fields,
    callable_doc,
    partial_keywords,
    signature,
    unwrap,
)
from agent_adapter.core.callables import is_async
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ToolArgsInvalid,
    ToolDefinitionError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.messages import ToolDef
from agent_adapter.core.types import CacheScope

logger = logging.getLogger(__name__)

ConfirmRule = bool | Callable[..., Any]
Effects = Literal["read_only", "additive", "destructive"]


@dataclass(frozen=True, eq=False)
class ToolSpec:
    """A registered tool: callable, argument model and execution policy.

    Equality and hashing are by identity: two specs built from the same function are
    different tools (they may differ in `fixed`, `rename` or policy).
    """

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., Any] = field(repr=False)
    context_param: str | None = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    llm_to_python: Mapping[str, str] = field(default_factory=dict)
    result: Callable[[Any], Any] | None = field(default=None, repr=False)
    returns: Any = None
    kind: str = "tool"
    tags: frozenset[str] = frozenset()
    parallel_safe: bool = False
    once_per_run: bool = False
    confirm: ConfirmRule = False
    timeout: float | None = None
    retries: int = 0
    cache_ttl: float | None = None
    prompt_hint: str | None = None
    portable: bool = True
    cache_scope: CacheScope = "run"
    targets: frozenset[str] | None = None  # target names this tool is exported to; None = all
    input_schema: Mapping[str, Any] | None = field(default=None, repr=False)  # imported tools
    output_schema: Mapping[str, Any] | None = field(default=None, repr=False)
    effects: Effects | None = None
    idempotent: bool | None = None

    @functools.cached_property
    def _schema(self) -> dict[str, Any]:
        if self.input_schema is not None:
            return dict(self.input_schema)
        return self.args_model.model_json_schema()

    @functools.cached_property
    def _result_schema(self) -> dict[str, Any] | None:
        if self.output_schema is not None:
            return dict(self.output_schema)
        if self.returns is None or self.result is not None:
            return None
        try:
            return TypeAdapter(self.returns).json_schema()
        except (PydanticUserError, PydanticUndefinedAnnotation) as exc:
            # Return types are informational; a type pydantic cannot describe must not block
            # registration.
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "tool %s: no result schema for %r",
                self.name,
                self.returns,
            )
            return None

    def json_schema(self) -> dict[str, Any]:
        """Neutral JSON schema of the LLM-supplied arguments (`input_schema` if set); a copy."""
        return copy.deepcopy(self._schema)

    def result_schema(self) -> dict[str, Any] | None:
        """`output_schema`, else the return annotation's schema; None if absent or `result` set."""
        return copy.deepcopy(self._result_schema)

    def tool_def(self, *, strict: bool = True) -> ToolDef:
        """The tool as model ports take it (see `strict_downgraded`)."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters=self.json_schema(),
            strict=strict and not strict_downgraded(self),
        )

    def validate_args(self, args: Mapping[str, Any] | None) -> BaseModel:
        """Validate LLM-supplied arguments against the tool's model.

        Raises:
            ToolArgsInvalid: arguments are missing, unknown or the wrong type.
        """
        data = copy.deepcopy(dict(args or {}))
        drop_strict_nulls(self.args_model, data)
        try:
            return self.args_model.model_validate(data)
        except ValidationError as exc:
            raise ToolArgsInvalid(
                self.name, exc.errors(include_url=False, include_input=False)
            ) from exc

    async def call(self, validated: BaseModel, ctx: ToolContext | None = None) -> Any:
        """Call the function with already-validated args (sync ones in a worker thread).

        A direct call is no executor attempt: `current_attempt()` is None inside it.

        Raises:
            ToolUsageError: `validated` is not an instance of this tool's `args_model`.
        """
        return await self._call(validated, ctx, None)

    async def _call(
        self, validated: BaseModel, ctx: ToolContext | None, attempt: Attempt | None
    ) -> Any:
        token = ATTEMPT.set(attempt)
        try:
            return await self._run(validated, ctx)
        finally:
            ATTEMPT.reset(token)

    async def _run(self, validated: BaseModel, ctx: ToolContext | None) -> Any:
        if not isinstance(validated, self.args_model):
            raise ToolUsageError(
                f"tool {self.name!r}: expected {self.args_model.__name__}, "
                f"got {type(validated).__name__}"
            )
        if self.input_schema is not None:
            kwargs = imported_kwargs(validated, self.input_schema)
        else:
            kwargs = {
                self.llm_to_python.get(name, name): getattr(validated, name)
                for name in self.args_model.model_fields
            }
        kwargs.update(self.fixed)
        if self.context_param is not None:
            kwargs[self.context_param] = ctx if ctx is not None else ToolContext()
        if is_async(self.func):
            value = await self.func(**kwargs)
        else:
            value = await asyncio.to_thread(self.func, **kwargs)
            if inspect.isawaitable(value):
                value = await value
        return self.result(value) if self.result is not None else value

    async def invoke(
        self, args: Mapping[str, Any] | None = None, ctx: ToolContext | None = None
    ) -> Any:
        """Validate args, call the function, apply `result`.

        Raises:
            ToolArgsInvalid: arguments are missing, unknown or the wrong type.
        """
        return await self.call(self.validate_args(args), ctx)


async def call_in_attempt(
    spec: ToolSpec, validated: BaseModel, ctx: ToolContext | None, attempt: Attempt
) -> Any:
    """`spec.call` as the executor's `attempt` (what `current_attempt()` returns inside)."""
    return await spec._call(validated, ctx, attempt)


def strict_downgraded(spec: ToolSpec) -> bool:
    """True when strict rendering must be off: an imported schema that is not portable."""
    return spec.input_schema is not None and not spec.portable


def build_spec(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    fixed: Mapping[str, Any] | None = None,
    hide: Iterable[str] | str = (),
    rename: Mapping[str, str] | None = None,
    result: Callable[[Any], Any] | None = None,
    args_model: type[BaseModel] | None = None,
    **policy: Any,
) -> ToolSpec:
    """Build a ToolSpec from a function's signature and docstring.

    Args:
        hide: Parameter names (or one name) hidden from the LLM; their defaults apply.
        rename: Python parameter name -> name the LLM sees.
        args_model: Use this model as the argument schema instead of the signature's types;
            its field names must equal the LLM-facing parameter names.
        **policy: Execution options such as `kind`, `tags`, `parallel_safe`, `timeout`.

    Raises:
        ToolDefinitionError: the function cannot be exposed as a tool.
    """
    target = unwrap(func)
    tool_name = _resolve_name(func, target, name)
    options = validate_policy(tool_name, policy)
    sig = signature(func, tool_name)
    doc_description, param_docs = parse_docstring(callable_doc(target), sig.parameters)
    shaping = _options(tool_name, sig, fixed, hide, rename, partial_keywords(func))
    fields = build_fields(func, tool_name, sig, param_docs, shaping)
    if args_model is None:
        model = _args_model(tool_name, fields.definitions)
    else:
        model = _given_args_model(tool_name, args_model, fields.definitions)
    spec = ToolSpec(
        name=tool_name,
        description=_resolve_description(tool_name, description, doc_description),
        args_model=model,
        func=func,
        context_param=fields.context_param,
        fixed=shaping.fixed,
        llm_to_python=fields.llm_to_python,
        result=result,
        returns=fields.returns,
        **options,
    )
    _check_schema(spec)
    return spec


def _resolve_name(func: Callable[..., Any], target: Any, name: Any) -> str:
    if name is not None and not isinstance(name, str):
        raise ToolDefinitionError(f"tool name must be a str, got {name!r}")
    tool_name = name or getattr(func, "__name__", None) or getattr(target, "__name__", "")
    if not tool_name:
        raise ToolDefinitionError(f"{func!r} has no __name__; pass name=")
    if not NAME_RE.match(tool_name):
        raise ToolDefinitionError(
            f"invalid tool name {tool_name!r}: use letters, digits and '_' (max 64)"
        )
    return str(tool_name)


def _resolve_description(tool_name: str, description: Any, doc_description: str) -> str:
    if description is not None and not isinstance(description, str):
        raise ToolDefinitionError(f"tool {tool_name!r}: description must be a str")
    text = (description or doc_description).strip()
    if not text:
        raise ToolDefinitionError(f"tool {tool_name!r} needs a description or a docstring")
    return text


def _options(
    tool_name: str,
    sig: inspect.Signature,
    fixed: Mapping[str, Any] | None,
    hide: Iterable[str] | str,
    rename: Mapping[str, str] | None,
    bound: Mapping[str, Any],
) -> Options:
    """Shaping options keyed by parameter; partial keyword bindings count as fixed values."""
    explicit = dict(fixed or {})
    clash = sorted(set(bound) & set(explicit))
    if clash:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: {clash[0]!r} is bound by the partial; "
            "do not also pass it in fixed="
        )
    hidden = {hide} if isinstance(hide, str) else set(hide)
    shaping = Options(fixed={**bound, **explicit}, hidden=hidden, renamed=dict(rename or {}))
    named = set(shaping.fixed) | shaping.hidden | set(shaping.renamed)
    unknown = named - set(sig.parameters)
    if unknown:
        raise ToolDefinitionError(f"tool {tool_name!r}: unknown parameters {sorted(unknown)}")
    return shaping


def _args_model(tool_name: str, definitions: Mapping[str, Any]) -> type[BaseModel]:
    try:
        model: type[BaseModel] = create_model(
            args_model_name(tool_name), __config__=ConfigDict(extra="forbid"), **definitions
        )
    except PydanticUserError as exc:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: cannot build the argument model ({exc})"
        ) from exc
    _reject_aliases(tool_name, model)
    return model


def _given_args_model(
    tool_name: str, model: Any, definitions: Mapping[str, Any]
) -> type[BaseModel]:
    """Check an `args_model=` against the LLM-facing names the signature produces."""
    if not (inspect.isclass(model) and issubclass(model, BaseModel)):
        raise ToolDefinitionError(f"tool {tool_name!r}: args_model must be a BaseModel subclass")
    if model.model_config.get("extra") != "forbid":
        raise ToolDefinitionError(
            f"tool {tool_name!r}: args_model {model.__name__} needs extra='forbid' in model_config"
        )
    _reject_aliases(tool_name, model)
    missing = sorted(set(definitions) - set(model.model_fields))
    extra = sorted(set(model.model_fields) - set(definitions))
    if missing or extra:
        raise ToolDefinitionError(
            f"tool {tool_name!r}: args_model {model.__name__} must have one field per "
            f"LLM-facing parameter; missing {missing}, extra {extra}"
        )
    return model


def _reject_aliases(tool_name: str, model: type[BaseModel]) -> None:
    for field_name, info in model.model_fields.items():
        if info.alias or info.validation_alias or info.serialization_alias:
            raise ToolDefinitionError(
                f"tool {tool_name!r}: argument {field_name!r} sets an alias; use rename= instead"
            )


def _check_schema(spec: ToolSpec) -> None:
    try:
        spec.json_schema()
    except PydanticUserError as exc:
        raise ToolDefinitionError(
            f"tool {spec.name!r}: cannot build the argument model ({exc})"
        ) from exc
