"""Gemini contract: kit C1-C8 for `run_function_calls` and for genai AFC proxies, plus edges."""

import asyncio
import dataclasses
import datetime
import json
import sys
from typing import Any

import pytest

pytest.importorskip("google.genai")
httpx = pytest.importorskip("httpx")

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from agent_adapter.core.context import ToolContext  # noqa: E402
from agent_adapter.core.errors import (  # noqa: E402
    ApprovalPending,
    MissingDependency,
    ToolNotFound,
    ToolUsageError,
)
from agent_adapter.core.imported import spec_from_schema  # noqa: E402
from agent_adapter.core.registry import Registry  # noqa: E402
from agent_adapter.core.tool import ToolSpec, build_spec  # noqa: E402
from agent_adapter.core.types import ApprovalDecision  # noqa: E402
from agent_adapter.exec import Executor  # noqa: E402
from agent_adapter.integrations.base import ToolRunner  # noqa: E402
from agent_adapter.integrations.gemini import (  # noqa: E402
    INTEGRATION,
    afc_callable,
    afc_callables,
    function_declarations,
    run_function_calls,
)
from agent_adapter.schema.transforms import to_gemini_parameters  # noqa: E402
from tests.fakes.approval import ScriptedApproval  # noqa: E402
from tests.integrations.gemini_dispatch_cases import *  # noqa: E402, F403
from tests.integrations.kit import FakeRunner, contract_tests  # noqa: E402

pytestmark = pytest.mark.contract


def _registry(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


def _declaration(native: Any) -> types.FunctionDeclaration:
    if isinstance(native, dict):
        return types.FunctionDeclaration(**native)
    if isinstance(native, Exported):
        return types.FunctionDeclaration(**native.declaration)
    return types.FunctionDeclaration.from_callable_with_api_option(
        callable=native, use_json_schema=True
    )


class _GeminiResults:
    """Shared result reading: a function-response Part (or the exception a call raised)."""

    name = "gemini"
    edits_supported = True  # the executor applies edited args
    approval = "port"

    def value(self, raw: Any) -> Any:
        response = raw.function_response.response
        return response.get("result", response)

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception):
            return True
        return "error" in raw.function_response.response

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        fd = _declaration(native)
        return fd.name or "", fd.description or "", fd.parameters_json_schema or {}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_gemini_approval")  # type: ignore[attr-defined]


@dataclasses.dataclass(frozen=True)
class Exported:
    spec: ToolSpec
    runner: ToolRunner
    declaration: dict[str, Any]


class FunctionCallHarness(_GeminiResults):
    """The host path: declarations to the model, its calls through `run_function_calls`."""

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        return Exported(spec, executor, INTEGRATION.export_tool(spec, executor=executor))

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return function_declarations(registry)

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Any:
        fc = types.FunctionCall(id=call_id, name=native.spec.name, args=args)
        [part] = await run_function_calls(
            [fc], executor=native.runner, registry=_registry(native.spec)
        )
        assert part.function_response.id == call_id
        return part

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_gemini_parameters(spec.json_schema())


def model_turn(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}}]}


def genai_client(*replies: dict[str, Any]) -> tuple[genai.Client, list[dict[str, Any]]]:
    """A genai client whose API answers with `replies` in order; records request bodies."""
    queue, bodies = list(replies), []

    def wire(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    options = types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key="test", http_options=options), bodies


async def afc_round_trip(native: Any, args: dict[str, Any], *, call_id: str = "") -> Any:
    """One AFC hop through `generate_content`: the model calls `native`, then says done."""
    call = {"name": native.__name__, "args": args, **({"id": call_id} if call_id else {})}
    client, _ = genai_client(model_turn({"functionCall": call}), model_turn({"text": "done"}))
    config = types.GenerateContentConfig(tools=[native])
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents="go", config=config
    )
    assert response.text == "done"
    history = response.automatic_function_calling_history or []
    return next(p for c in history for p in c.parts or () if p.function_response)


class AfcHarness(_GeminiResults):
    """genai automatic function calling: the SDK invokes the proxy itself."""

    native_call_ids = False  # AFC does not hand the proxy the call id

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return afc_callable(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return afc_callables(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await afc_round_trip(native, args, call_id=call_id)

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        neutral = spec.json_schema()["properties"]
        gemini = to_gemini_parameters(spec.json_schema())
        required = set(gemini.get("required", []))
        properties = {
            name: prop if name in required else {**prop, "default": neutral[name].get("default")}
            for name, prop in gemini["properties"].items()
        }
        return {
            "type": "object",
            "properties": properties,
            "required": [name for name in properties if name in required],
        }


globals().update((item.__name__, item) for item in contract_tests(FunctionCallHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(FunctionCallHarness, runner_factory=Executor)
)
globals().update((f"{item.__name__}_afc", item) for item in contract_tests(AfcHarness))
globals().update(
    (f"{item.__name__}_afc_executor", item)
    for item in contract_tests(AfcHarness, runner_factory=Executor)
)


# --- AFC proxy details ------------------------------------------------------------------------


def odd_names_spec(record: list[dict[str, Any]]) -> ToolSpec:
    async def lookup(**kwargs: Any) -> dict[str, Any]:
        record.append(kwargs)
        return {"seen": sorted(kwargs)}

    schema = {
        "type": "object",
        "properties": {
            "X-Trace": {"type": "string"},
            "class": {"type": "integer"},
            "9lives": {"type": "boolean"},
            "X_Trace": {"type": "string"},
        },
        "required": ["X-Trace"],
    }
    return spec_from_schema("odd_lookup", "Look up with odd names.", schema, lookup)


async def test_afc_sanitises_parameter_names_and_maps_them_back():
    record: list[dict[str, Any]] = []
    native = afc_callable(odd_names_spec(record), executor=FakeRunner())
    fd = _declaration(native)
    assert list(fd.parameters_json_schema["properties"]) == [
        "X_Trace",
        "p_class",
        "p_9lives",
        "X_Trace_2",
    ]
    assert fd.parameters_json_schema["required"] == ["X_Trace"]
    args = {"X_Trace": "t", "p_class": 2, "p_9lives": True, "X_Trace_2": "u"}
    part = await afc_round_trip(native, args)
    assert record == [{"X-Trace": "t", "class": 2, "9lives": True, "X_Trace": "u"}]
    assert part.function_response.response == {
        "result": {"seen": ["9lives", "X-Trace", "X_Trace", "class"]}
    }


async def test_afc_proxy_returns_json_values_and_ignores_unknown_arguments():
    async def when(day: str) -> dict[str, Any]:
        """Echo a date."""
        return {"at": datetime.date(2026, 10, 29), "tags": {"a"}, "day": day}

    native = afc_callable(build_spec(when), executor=FakeRunner())
    part = await afc_round_trip(native, {"day": "thu", "extra": 1})
    assert part.function_response.response == {
        "result": {"at": "2026-10-29", "tags": ["a"], "day": "thu"}
    }


def echo_q(q: str) -> str:
    """Echo."""
    return q


def test_afc_callable_refuses_a_spec_not_exported_to_gemini():
    spec = build_spec(echo_q, name="openai_only", targets="openai")
    with pytest.raises(ToolUsageError):
        afc_callable(spec)


def test_afc_callable_without_an_executor_uses_the_default_one():
    native = afc_callable(build_spec(echo_q))
    assert native.__name__ == "echo_q" and native.__doc__ == "Echo."


# --- run_function_calls --------------------------------------------------------------------


def echo_spec(name: str = "echo", **policy: Any) -> ToolSpec:
    async def echo(text: str, ctx: ToolContext) -> dict[str, Any]:
        """Echo text with the call id and the native call."""
        native = ctx.extras.get("gemini")
        return {"text": text, "call_id": ctx.call_id, "run_id": ctx.run_id, "native": native.name}

    return build_spec(echo, name=name, **policy)


async def test_run_function_calls_accepts_parts_mappings_and_uses_the_executor_registry():
    registry = _registry(echo_spec())
    executor = Executor(registry)
    calls = [
        types.Part(function_call=types.FunctionCall(id="c1", name="echo", args={"text": "a"})),
        {"name": "echo", "args": {"text": "b"}, "id": "c2"},
        {"name": "echo", "args": {"text": "c"}},
    ]
    parts = await run_function_calls(calls, executor=executor)
    results = [p.function_response.response["result"] for p in parts]
    assert [(r["text"], r["native"]) for r in results] == [("a", "echo")] * 1 + [
        ("b", "echo"),
        ("c", "echo"),
    ]
    assert [r["call_id"] for r in results[:2]] == ["c1", "c2"]
    assert results[2]["call_id"].startswith("call_")
    assert len({r["run_id"] for r in results}) == 1  # one run per batch
    assert [p.function_response.id for p in parts] == ["c1", "c2", None]


async def test_run_function_calls_runs_executor_calls_concurrently_in_input_order():
    started, release = [], asyncio.Event()

    async def slow(n: int) -> int:
        """Wait for the others, then return n."""
        started.append(n)
        if len(started) == 3:
            release.set()
        await asyncio.wait_for(release.wait(), 2)
        return n

    executor = Executor(_registry(build_spec(slow, parallel_safe=True)))
    calls = [{"name": "slow", "args": {"n": n}} for n in (3, 1, 2)]
    parts = await run_function_calls(calls, executor=executor)
    assert [p.function_response.response["result"] for p in parts] == [3, 1, 2]


async def test_run_function_calls_ends_only_the_runs_it_owns(monkeypatch):
    executor = Executor(_registry(echo_spec()))
    ended: list[str] = []
    monkeypatch.setattr(executor, "end_run", ended.append)
    call = {"name": "echo", "args": {"text": "x"}}
    [part] = await run_function_calls([call], executor=executor)
    assert ended == [part.function_response.response["result"]["run_id"]]
    ctx = executor.context(run_id="run_host")
    [part] = await run_function_calls([call], executor=executor, ctx=ctx)
    assert part.function_response.response["result"]["run_id"] == "run_host"
    [part] = await run_function_calls([call], executor=executor, ctx=ToolContext())
    assert len(ended) == 2 and ended[1] == part.function_response.response["result"]["run_id"]


async def test_run_function_calls_with_a_plain_runner_runs_in_order_in_one_run():
    parts = await run_function_calls(
        [{"name": "echo", "args": {"text": t}} for t in "ab"],
        executor=FakeRunner(),
        registry=_registry(echo_spec()),
    )
    runs = {p.function_response.response["result"]["run_id"] for p in parts}
    assert len(runs) == 1 and runs.pop().startswith("run_")
    ctx = ToolContext(run_id="run_mine")
    [part] = await run_function_calls(
        [{"name": "echo", "args": {"text": "c"}}],
        executor=FakeRunner(),
        registry=_registry(echo_spec()),
        ctx=ctx,
    )
    assert part.function_response.response["result"]["run_id"] == "run_mine"


async def test_run_function_calls_turns_model_visible_errors_into_error_responses():
    executor = Executor(_registry(echo_spec()))
    [part] = await run_function_calls([{"name": "echo", "args": {}}], executor=executor)
    error = part.function_response.response["error"]
    assert error["kind"] == "tool-args-invalid" and "text" in error["message"]


async def test_run_function_calls_propagates_pending_approval_and_unknown_tools():
    pending = ApprovalPending("apr_1", tool="guarded")
    approval = ScriptedApproval([pending])
    executor = Executor(_registry(echo_spec("guarded", confirm=True)), approval=approval)
    with pytest.raises(ApprovalPending):
        await run_function_calls([{"name": "guarded", "args": {"text": "x"}}], executor=executor)
    with pytest.raises(ToolNotFound):
        await run_function_calls([{"name": "nope", "args": {}}], executor=executor)


async def test_run_function_calls_skips_tools_not_exported_to_gemini():
    executor = Executor(_registry(echo_spec("hidden", targets="openai")))
    with pytest.raises(ToolNotFound):
        await run_function_calls([{"name": "hidden", "args": {"text": "x"}}], executor=executor)


@pytest.mark.parametrize("item", [{"args": {}}, types.FunctionCall(args={}), 42])
async def test_run_function_calls_rejects_items_without_a_name(item):
    with pytest.raises(ToolUsageError, match="with a name"):
        await run_function_calls([item], executor=Executor(Registry()))


async def test_run_function_calls_needs_a_registry():
    with pytest.raises(ToolUsageError, match="needs a registry"):
        await run_function_calls([{"name": "echo", "args": {}}], executor=FakeRunner())


async def test_run_function_calls_needs_google_genai(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[gemini\]"):
        await run_function_calls([], executor=Executor(Registry()))
