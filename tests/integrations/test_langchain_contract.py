"""LangChain contract: kit C1-C8 for exported StructuredTools (async and sync), plus import."""

import asyncio
import json
import sys
from typing import Annotated, Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import ToolMessage
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalPending,
    DuplicateTool,
    MissingDependency,
    ToolArgsInvalid,
    ToolDefinitionError,
    ToolError,
    ToolFailed,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.langchain import (
    INTEGRATION,
    import_langchain,
    structured_tool,
    structured_tools,
)
from agent_adapter.runtime.current import use_context
from tests.integrations.kit import FakeRunner, contract_tests

pytestmark = pytest.mark.contract


def without_titles(node: Any) -> Any:
    """`node` without `title` keywords (property names called "title" stay)."""
    if isinstance(node, list):
        return [without_titles(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: (
            {name: without_titles(sub) for name, sub in value.items()}
            if key == "properties"
            else without_titles(value)
        )
        for key, value in node.items()
        if key != "title"
    }


def tool_call(native: Any, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": native.name, "args": args, "id": call_id, "type": "tool_call"}


class LangChainHarness:
    """Exported tools run as a LangChain agent runs them: `ainvoke(ToolCall)`."""

    name = "langchain"
    edits_supported = True  # the executor applies edited args
    approval = "port"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return structured_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return structured_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        message = await native.ainvoke(tool_call(native, args, call_id))
        assert isinstance(message, ToolMessage) and message.tool_call_id == call_id
        return message

    def value(self, raw: ToolMessage) -> Any:
        try:
            return json.loads(raw.content)  # type: ignore[arg-type]
        except json.JSONDecodeError:
            return raw.content

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or raw.status == "error"

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_langchain_approval")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        function = convert_to_openai_function(native)
        return function["name"], function["description"], function["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return without_titles(spec.json_schema())  # what bind_tools sends: no `title`s


class SyncLangChainHarness(LangChainHarness):
    """The sync path: `invoke(ToolCall)` from a worker thread (`run_sync` inside)."""

    cancels_tool = False  # the tool runs on the worker thread's own loop

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        message = await asyncio.to_thread(native.invoke, tool_call(native, args, call_id))
        assert message.tool_call_id == call_id
        return message


globals().update((item.__name__, item) for item in contract_tests(LangChainHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(LangChainHarness, runner_factory=Executor)
)
globals().update((f"{item.__name__}_sync", item) for item in contract_tests(SyncLangChainHarness))


# --- export details --------------------------------------------------------------------------


async def lookup(section: str, ctx: ToolContext) -> dict[str, Any]:
    """Look up a section."""
    return {"section": section, "call_id": ctx.call_id, "native": ctx.extras["langchain"]["id"]}


async def test_dict_input_returns_the_raw_value_and_a_generated_call_id():
    native = structured_tool(build_spec(lookup), executor=FakeRunner())
    value = await native.ainvoke({"section": "80C"})
    assert value["section"] == "80C" and value["native"] is None
    assert value["call_id"].startswith("call_")


async def test_sync_invoke_inside_a_running_loop_uses_a_worker_thread():
    native = structured_tool(build_spec(lookup), executor=FakeRunner())
    assert native.invoke({"section": "80D"})["section"] == "80D"


def test_export_carries_tags_and_kind_and_honours_targets():
    native = structured_tool(build_spec(lookup, tags={"tax"}, kind="search"))
    assert native.tags == ["tax"] and native.metadata == {"agent_adapter_kind": "search"}
    assert INTEGRATION.export_tool(build_spec(lookup)).name == "lookup"
    with pytest.raises(ToolUsageError):
        structured_tool(build_spec(lookup, targets="openai"))


def test_export_needs_langchain_core(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[langchain\]"):
        structured_tool(build_spec(lookup))


# --- import ----------------------------------------------------------------------------------


class AddArgs(BaseModel):
    a: int = Field(description="First addend.", gt=0)
    b: int = 0


@tool("add-numbers", args_schema=AddArgs)
async def add_numbers(a: int, b: int = 0) -> int:
    """Add two integers."""
    return a + b


@tool
def with_injected(x: int, secret: Annotated[str, InjectedToolArg] = "s3") -> str:
    """Has an argument the model never sees."""
    return f"{x}:{secret}"


@tool
def explode(reason: str) -> str:
    """Always fails with a tool error."""
    raise ToolException(f"cannot: {reason}")


class Plain(BaseTool):
    name: str = "plain"
    description: str = ""

    def _run(self, q: str) -> str:
        return q.upper()


dict_tool = StructuredTool.from_function(
    func=lambda **kw: sorted(kw.items()),
    name="dict_schema",
    description="Takes a raw JSON schema.",
    args_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    infer_schema=False,
)


async def test_import_builds_specs_from_the_model_facing_schema():
    registry = Registry()
    specs = import_langchain(registry, [add_numbers, with_injected, dict_tool, Plain()])
    assert [s.name for s in specs] == ["add_numbers", "with_injected", "dict_schema", "plain"]
    add = registry.get("add_numbers")
    assert add.json_schema()["properties"]["a"] == {
        "description": "First addend.",
        "exclusiveMinimum": 0,
        "type": "integer",
    }
    assert add.kind == "langchain" and add.tags == frozenset({"langchain"})
    assert list(registry.get("with_injected").json_schema()["properties"]) == ["x"]
    assert registry.get("plain").description == "plain"  # empty description -> the name
    executor = Executor(registry)
    assert await executor.run("add_numbers", {"a": 2, "b": 3}) == 5
    assert await executor.run("with_injected", {"x": 1}) == "1:s3"
    assert await executor.run("dict_schema", {"q": "hi"}) == [("q", "hi")]
    assert await executor.run("plain", {"q": "hi"}) == "HI"


async def test_import_maps_tool_exceptions_and_validation_errors():
    registry = Registry()
    import_langchain(registry, [add_numbers, explode])
    with pytest.raises(ToolError, match="cannot: x") as info:
        await registry.get("explode").invoke({"reason": "x"})
    assert info.value.tool == "explode"
    # the loose model accepts a=0; LangChain's own schema (gt=0) rejects it
    with pytest.raises(ToolArgsInvalid) as invalid:
        await registry.get("add_numbers").invoke({"a": 0})
    assert invalid.value.tool == "add_numbers" and invalid.value.errors[0]["loc"] == ["a"]


@tool
def builds_a_model(n: int) -> int:
    """Fails inside its body with a pydantic error that is not about its arguments."""
    return AddArgs.model_validate({"a": n - 10}).a


async def test_validation_errors_inside_the_tool_body_are_not_argument_errors():
    registry = Registry()
    import_langchain(registry, [builds_a_model])
    assert await registry.get("builds_a_model").invoke({"n": 15}) == 5
    with pytest.raises(ToolFailed) as info:
        await Executor(registry).run("builds_a_model", {"n": 1})
    assert info.value.cause_type == "pydantic_core._pydantic_core.ValidationError"


def test_import_applies_prefix_policy_and_sanitised_names():
    registry = Registry()
    [spec] = import_langchain(registry, [add_numbers], prefix="lc.", confirm=True, tags={"x"})
    assert spec.name == "lc_add_numbers" and spec.confirm is True and spec.tags == {"x"}
    numbered = StructuredTool.from_function(func=lambda: 1, name="9-lives", description="Nine.")
    assert import_langchain(Registry(), [numbered])[0].name == "t_9_lives"


def test_import_is_all_or_nothing():
    registry = Registry()
    import_langchain(registry, [dict_tool])
    with pytest.raises(DuplicateTool):
        import_langchain(registry, [add_numbers, dict_tool])
    assert registry.names() == ["dict_schema"]


def test_import_rejects_objects_that_are_not_langchain_tools():
    with pytest.raises(ToolDefinitionError, match="BaseTool"):
        import_langchain(Registry(), [lambda: 1])


async def test_round_trip_export_then_import_runs_the_original_tool():
    source = Registry()
    source.add(build_spec(lookup))
    target = Registry()
    import_langchain(target, structured_tools(source, executor=FakeRunner()), prefix="re_")
    value = await target.get("re_lookup").invoke({"section": "80C"})
    assert value["section"] == "80C"


# --- run ids: resuming a deferred approval -------------------------------------------------


async def guarded_echo(item: str, ctx: ToolContext) -> dict[str, str]:
    """Echo an item and the run it ran in (needs approval)."""
    ran.append(item)
    return {"item": item, "run_id": ctx.run_id, "thread_id": ctx.thread_id}


ran: list[str] = []


@pytest.mark.parametrize(
    ("configurable", "run_id"),
    [({"thread_id": "t-1"}, "t-1"), ({"thread_id": "t-1", "run_id": "r-9"}, "r-9")],
)
async def test_a_deferred_approval_resumes_through_the_runnable_config(configurable, run_id):
    ran.clear()
    store = MemoryApprovalStore()
    executor = Executor(approval=DeferredApproval(store))
    native = structured_tool(build_spec(guarded_echo, confirm=True), executor=executor)
    call, config = tool_call(native, {"item": "x"}, "call_7"), {"configurable": configurable}
    with pytest.raises(ApprovalPending) as pending:
        await native.ainvoke(call, config=config)
    await store.decide(pending.value.approval_id, ApprovalDecision(approved=True))
    message = await native.ainvoke(call, config=config)
    assert ran == ["x"]
    assert json.loads(message.content) == {"item": "x", "run_id": run_id, "thread_id": "t-1"}


async def test_without_a_run_id_each_invocation_is_a_new_run():
    ran.clear()
    store = MemoryApprovalStore()
    executor = Executor(approval=DeferredApproval(store))
    native = structured_tool(build_spec(guarded_echo, confirm=True), executor=executor)
    call = tool_call(native, {"item": "y"}, "call_8")
    with pytest.raises(ApprovalPending) as first:
        await native.ainvoke(call)
    await store.decide(first.value.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ApprovalPending) as second:
        await native.ainvoke(call)
    assert second.value.approval_id != first.value.approval_id and ran == []


async def test_the_current_context_is_the_base_inside_an_adapter_run():
    native = structured_tool(build_spec(guarded_echo), executor=FakeRunner())
    config = {"configurable": {"thread_id": "ignored"}}
    with use_context(ToolContext(run_id="run_outer", call_id="call_outer", thread_id="t-0")):
        message = await native.ainvoke(tool_call(native, {"item": "n"}, "call_9"), config=config)
    assert json.loads(message.content) == {"item": "n", "run_id": "run_outer", "thread_id": "t-0"}


async def test_the_sync_path_reads_the_runnable_config_too():
    ran.clear()
    native = structured_tool(build_spec(guarded_echo), executor=FakeRunner())
    config = {"configurable": {"thread_id": "t-2"}}
    message = await asyncio.to_thread(native.invoke, tool_call(native, {"item": "s"}, "c"), config)
    assert json.loads(message.content)["run_id"] == "t-2"
