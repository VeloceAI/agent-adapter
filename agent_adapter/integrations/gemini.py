"""Gemini: function declarations, genai automatic-function-calling proxies, call dispatch."""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import keyword
import logging
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, ClassVar

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, MissingDependency, ToolUsageError
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import to_jsonable
from agent_adapter.exec.defaults import runner_for
from agent_adapter.exec.executor import Executor
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.runtime.current import current_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.schema.transforms import to_gemini_parameters

logger = logging.getLogger(__name__)

TARGET = "gemini"
_GENAI_HINT = 'pip install "agent-adapter[gemini]"'


def function_declaration(spec: ToolSpec) -> dict[str, Any]:
    """One tool as a Gemini function declaration dict."""
    check_target(spec, TARGET)
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters_json_schema": to_gemini_parameters(spec.json_schema()),
    }


def function_declarations(registry: Registry) -> list[dict[str, Any]]:
    """Every registry tool exported to Gemini, as function declaration dicts."""
    return [function_declaration(spec) for spec in registry.for_target(TARGET)]


def genai_tool(registry: Registry) -> Any:
    """A `google.genai.types.Tool` holding every registry tool exported to Gemini.

    Raises:
        MissingDependency: google-genai is not installed (`pip install "agent-adapter[gemini]"`).
    """
    types = _genai_types("genai_tool")
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**decl) for decl in function_declarations(registry)
        ]
    )


# --- automatic function calling (AFC) ----------------------------------------------------


class _AnyValueMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return True  # genai checks `isinstance(arg, annotation)`; our own validation runs later


class _SchemaArg(metaclass=_AnyValueMeta):
    """A parameter annotation whose JSON schema is a fixed property schema."""

    schema: ClassVar[dict[str, Any]] = {}

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> Any:
        return handler(Any)

    @classmethod
    def __get_pydantic_json_schema__(cls, core: Any, handler: GetJsonSchemaHandler) -> Any:
        return copy.deepcopy(cls.schema)


def afc_callable(
    spec: ToolSpec, *, executor: ToolRunner | None = None
) -> Callable[..., Awaitable[Any]]:
    """An async function for genai's `GenerateContentConfig(tools=[...])` that runs `spec`.

    genai builds the declaration from the synthesised `__signature__` (each property keeps
    the Gemini schema; optional ones carry their default, `None` when they have none) and
    `__doc__`. Calls go through the runner; genai turns any exception into an `error` reply.

    Raises:
        ToolUsageError: `spec.targets` excludes gemini.
    """
    check_target(spec, TARGET)
    runner = lazy_runner(executor)
    parameters, to_property = _afc_parameters(spec)

    async def proxy(**kwargs: Any) -> Any:
        args = {to_property.get(name, name): value for name, value in kwargs.items()}
        bound = INTEGRATION.bind_context({"name": spec.name, "args": args})
        outer = current_context()  # inside an adapter run, the call joins it
        ctx = (
            bound
            if outer is None
            else outer.child(call_id="", extras={**outer.extras, **bound.extras})
        )
        return to_jsonable(await runner.run_spec(spec, args, ctx))

    proxy.__name__ = proxy.__qualname__ = spec.name
    proxy.__doc__ = spec.description
    proxy.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    proxy.__annotations__ = {p.name: p.annotation for p in parameters}
    return proxy


def afc_callables(
    registry: Registry, *, executor: ToolRunner | None = None
) -> list[Callable[..., Awaitable[Any]]]:
    """`afc_callable` for every registry tool exported to Gemini."""
    return [afc_callable(spec, executor=executor) for spec in registry.for_target(TARGET)]


def _afc_parameters(spec: ToolSpec) -> tuple[list[inspect.Parameter], dict[str, str]]:
    """Keyword-only parameters for the proxy, and parameter name -> schema property name."""
    neutral = spec.json_schema().get("properties", {})
    gemini = to_gemini_parameters(spec.json_schema())
    required = set(gemini.get("required", []))
    parameters: list[inspect.Parameter] = []
    to_property: dict[str, str] = {}
    for prop, prop_schema in gemini.get("properties", {}).items():
        name = _parameter_name(prop, to_property)
        to_property[name] = prop
        # genai marks every property without a `default` as required.
        default = (
            inspect.Parameter.empty if prop in required else neutral[prop].get("default", None)
        )
        annotation = type(f"{spec.name}_{name}", (_SchemaArg,), {"schema": prop_schema})
        parameters.append(
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation
            )
        )
    return parameters, to_property


def _parameter_name(prop: str, taken: Mapping[str, str]) -> str:
    """`prop` when it is a usable Python parameter name, else a sanitised unique one."""
    name = prop
    if not prop.isidentifier() or keyword.iskeyword(prop):
        name = re.sub(r"\W", "_", prop)
        if not name or name[0].isdigit() or keyword.iskeyword(name):
            name = f"p_{name}"
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


# --- manual dispatch -------------------------------------------------------------------------


async def run_function_calls(
    calls: Iterable[Any],
    *,
    executor: ToolRunner | None = None,
    ctx: ToolContext | None = None,
    registry: Registry | None = None,
) -> list[Any]:
    """Run the model's function calls; one function-response `types.Part` per call, in order.

    `calls` holds `types.FunctionCall`s, parts with a `function_call`, or `{"name", "args",
    "id"}` mappings; a call without an id gets a stable one (name, arguments, and which
    occurrence of that same call it is), so a repeated or reordered turn gets the same ids.
    Tools are looked up in `registry`, else the executor's registry. An `Executor` runs the
    calls concurrently in one run; another runner runs them in order. Model-visible errors
    become `{"error": {"kind", "message"}}` responses; any other exception
    (`ApprovalPending`, `ToolNotFound`, `ToolTimeout`, ...) propagates.

    `ctx=None` uses `current_context()` when one is set (inside an adapter run), else a
    fresh run per batch. For approval, pass a `ctx` with a stable `run_id` and repeat the
    same calls after the decision: calls that completed in a batch stopped by
    `ApprovalPending` are answered from memory on the retry, not run again.

    Raises:
        ToolUsageError: no registry, or an item is not a function call.
        MissingDependency: google-genai is not installed.
    """
    types = _genai_types("run_function_calls")
    runner = runner_for(executor)
    tools = _registry_for(runner, registry)
    batch = [(fc, tools.get(fc.name)) for fc in _function_calls(calls)]
    ctx = ctx if ctx is not None else current_context()
    if isinstance(runner, Executor):
        responses = await _run_concurrently(runner, batch, ctx)
    else:
        keep = ctx is not None and bool(ctx.run_id)
        responses = await _run_batch(runner, batch, _run_context(ctx), keep=keep)
    return [
        types.Part(
            function_response=types.FunctionResponse(id=fc.id, name=fc.name, response=response)
        )
        for (fc, _), response in zip(batch, responses, strict=True)
    ]


@dataclasses.dataclass(frozen=True)
class _Call:
    name: str
    args: dict[str, Any]
    id: str | None  # the model's id, echoed in the response
    call_id: str  # the id the executor sees: the model's, else a stable one
    digest: str  # name + arguments, so a kept response only answers the same call


def _function_calls(items: Iterable[Any]) -> list[_Call]:
    """Parsed calls; id-less ones are told apart by how often the same call came before."""
    seen: Counter[str] = Counter()
    calls = []
    for item in items:
        call = _function_call(item, 0)
        calls.append(_function_call(item, seen[call.digest]) if call.id is None else call)
        seen[call.digest] += 1
    return calls


def _function_call(item: Any, occurrence: int) -> _Call:
    call = getattr(item, "function_call", None) or item
    if isinstance(call, Mapping):
        name, args, call_id = call.get("name"), call.get("args"), call.get("id")
    else:
        name = getattr(call, "name", None)
        args, call_id = getattr(call, "args", None), getattr(call, "id", None)
    if not isinstance(name, str) or not name:
        raise ToolUsageError(f"run_function_calls needs function calls with a name, got {item!r}")
    arguments = dict(args or {})
    text = json.dumps({"name": name, "args": arguments}, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stable = call_id or f"call_{digest[:16]}_{occurrence}"
    return _Call(name, arguments, call_id or None, stable, digest)


def _registry_for(runner: Any, registry: Registry | None) -> Registry:
    found = registry if registry is not None else getattr(runner, "registry", None)
    if not isinstance(found, Registry):
        raise ToolUsageError(
            "run_function_calls needs a registry: pass registry= or an Executor that has one"
        )
    return found.for_target(TARGET)


def _run_context(ctx: ToolContext | None) -> ToolContext:
    """The caller's context, with a fresh run id when it has none."""
    base = ctx if ctx is not None else ToolContext()
    return base if base.run_id else dataclasses.replace(base, run_id=new_id("run_"))


async def _run_concurrently(
    executor: Executor, batch: list[tuple[_Call, ToolSpec]], ctx: ToolContext | None
) -> list[dict[str, Any]]:
    """Every call at once in one run (owned and ended here when the caller gave none)."""
    owned = ctx is None or not ctx.run_id
    base = executor.context() if ctx is None else _run_context(ctx)
    try:
        return await _run_batch(executor, batch, base, keep=not owned)
    finally:
        if owned:
            executor.end_run(base.run_id)


async def _run_batch(
    runner: ToolRunner, batch: list[tuple[_Call, ToolSpec]], base: ToolContext, *, keep: bool
) -> list[dict[str, Any]]:
    """Responses in order: kept ones are reused, the rest run, the first error raises."""
    responses: list[dict[str, Any] | None] = [
        _COMPLETED.take(base.run_id, fc) if keep else None for fc, _ in batch
    ]
    todo = [i for i, response in enumerate(responses) if response is None]
    outcomes = await _run_todo(runner, [batch[i] for i in todo], base)
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    for i, outcome in zip(todo, outcomes, strict=False):
        if not isinstance(outcome, Exception):
            responses[i] = outcome
    if errors:
        if keep and any(isinstance(error, ApprovalPending) for error in errors):
            for (fc, _), response in zip(batch, responses, strict=True):
                if response is not None:
                    _COMPLETED.keep(base.run_id, fc, response)
        logger.debug("gemini function call failed: %s", type(errors[0]).__name__)
        raise errors[0]
    return [response for response in responses if response is not None]


async def _run_todo(
    runner: ToolRunner, batch: list[tuple[_Call, ToolSpec]], base: ToolContext
) -> list[dict[str, Any] | Exception]:
    """An Executor runs the calls at once; another runner in order, stopping at an error."""
    if isinstance(runner, Executor):
        work = [functools.partial(_respond, runner, spec, fc, base) for fc, spec in batch]
        return await runner.parallel(*work, on_error="collect")
    outcomes: list[dict[str, Any] | Exception] = []
    for fc, spec in batch:
        try:
            outcomes.append(await _respond(runner, spec, fc, base))
        except ApprovalPending as exc:
            outcomes.append(exc)  # siblings still run; _run_batch raises it at the end
        except Exception as exc:
            outcomes.append(exc)  # _run_batch raises it; later calls do not run
            break
    return outcomes


async def _respond(
    runner: ToolRunner, spec: ToolSpec, fc: _Call, base: ToolContext
) -> dict[str, Any]:
    bound = INTEGRATION.bind_context(fc)
    child = base.child(call_id=fc.call_id, extras={**base.extras, **bound.extras})
    try:
        value = await runner.run_spec(spec, fc.args, child, call_id=fc.call_id)
    except MODEL_VISIBLE_ERRORS as exc:
        return model_visible_error(exc)
    return {"result": to_jsonable(value)}


class _CompletedCalls:
    """Responses of calls that finished in a batch stopped by ApprovalPending (bounded LRU)."""

    def __init__(self, limit: int = 4096) -> None:
        self.limit = limit
        self._items: OrderedDict[tuple[str, str], tuple[str, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()  # run_sync loops in several threads share it

    def keep(self, run_id: str, fc: _Call, response: dict[str, Any]) -> None:
        with self._lock:
            self._items[(run_id, fc.call_id)] = (fc.digest, response)
            self._items.move_to_end((run_id, fc.call_id))
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def take(self, run_id: str, fc: _Call) -> dict[str, Any] | None:
        """The kept response for this exact call (same id, name and arguments), once."""
        with self._lock:
            kept = self._items.pop((run_id, fc.call_id), None)
        if kept is None or kept[0] != fc.digest:
            return None
        return kept[1]


_COMPLETED = _CompletedCalls()


def _genai_types(caller: str) -> Any:
    try:
        from google.genai import types
    except ImportError as exc:
        raise MissingDependency(f"{caller} needs google-genai: {_GENAI_HINT}") from exc
    return types


class GeminiIntegration(IntegrationBase):
    """Exports function declaration dicts; the host runs the calls (`run_function_calls`)."""

    name = TARGET
    requires = ("google.genai",)
    extra = "gemini"
    call_id_fields = ("id",)  # FunctionCall.id, when the API sets one

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> dict[str, Any]:
        """One function declaration dict; `executor` is unused."""
        return function_declaration(spec)


INTEGRATION = GeminiIntegration()


__all__ = [
    "INTEGRATION",
    "GeminiIntegration",
    "afc_callable",
    "afc_callables",
    "function_declaration",
    "function_declarations",
    "genai_tool",
    "run_function_calls",
]
