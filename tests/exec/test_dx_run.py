"""DX sugar: the default registry, run by function, sync twins, configure, call and acall."""

import asyncio
from collections.abc import Iterator

import pytest

import agent_adapter as aa
from agent_adapter import AutoApprove, Executor, Registry, ToolCall
from agent_adapter.core._default import default_registry
from agent_adapter.errors import ApprovalDenied, ToolUsageError
from agent_adapter.exec import _sugar, defaults
from agent_adapter.runtime import LoopRunner


@pytest.fixture(autouse=True)
def _isolated_defaults(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test gets a clean default registry and no default executor."""
    saved = dict(default_registry._tools)
    monkeypatch.setattr(defaults, "_default", None)
    yield
    default_registry._tools.clear()
    default_registry._tools.update(saved)


def _add() -> object:
    @aa.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return add


def test_executor_without_a_registry_uses_the_default_registry() -> None:
    add = _add()
    executor = Executor()
    assert executor.registry is default_registry
    assert executor.run_sync(add, {"a": 1, "b": 2}) == 3
    assert executor.run_sync("add", {"a": 2, "b": 2}) == 4


def test_an_explicit_registry_wins_over_the_default() -> None:
    _add()
    mine = Registry()
    executor = Executor(mine)
    assert executor.registry is mine
    with pytest.raises(aa.errors.ToolNotFound):
        executor.run_sync("add", {"a": 1, "b": 2})


def test_replace_keeps_the_default_registry() -> None:
    assert Executor().replace(namespace="x").registry is default_registry


def test_default_executor_knows_the_default_registry() -> None:
    assert defaults.default_executor().registry is default_registry


async def test_run_accepts_the_decorated_function() -> None:
    registry = Registry()

    @registry.tool
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"hi {name}"

    executor = Executor(registry)
    assert await executor.run(greet, {"name": "ana"}) == "hi ana"
    assert await executor.run(registry.get("greet"), {"name": "bo"}) == "hi bo"


async def test_run_accepts_a_wrapped_function() -> None:
    registry = Registry()

    def legacy(x: int) -> int:
        """Double x."""
        return 2 * x

    registry.wrap(legacy)
    assert await Executor(registry).run(legacy, {"x": 4}) == 8


async def test_run_of_an_undecorated_function_says_how_to_register_it() -> None:
    def plain(x: int) -> int:
        return x

    with pytest.raises(ToolUsageError) as caught:
        await Executor(Registry()).run(plain, {"x": 1})
    assert str(caught.value) == (
        "'test_run_of_an_undecorated_function_says_how_to_register_it.<locals>.plain' is not "
        "a tool: decorate it with @tool (or registry.tool), or register it with "
        "registry.wrap(fn), then run it by function or by name"
    )


async def test_run_of_a_function_from_another_registry_names_the_fix() -> None:
    other = Registry()

    @other.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    with pytest.raises(ToolUsageError) as caught:
        await Executor(Registry()).run(ping)
    assert str(caught.value) == (
        "tool 'ping' is registered in another Registry, not this executor's: "
        "pass that registry to Executor(registry)"
    )


async def test_run_of_a_spec_from_another_registry_points_at_run_spec() -> None:
    other = Registry()
    spec = other.wrap(lambda: 1, name="one", description="One.")
    with pytest.raises(ToolUsageError) as caught:
        await Executor(Registry()).run(spec)
    assert str(caught.value) == (
        "tool spec 'one' is not in this executor's registry: "
        "run it with executor.run_spec(spec, args), or register it with registry.add(spec)"
    )


async def test_run_of_a_function_registered_twice_asks_for_the_name() -> None:
    registry = Registry()

    def twice(x: int) -> int:
        """Echo."""
        return x

    registry.wrap(twice, name="first")
    registry.wrap(twice, name="second")
    with pytest.raises(
        ToolUsageError, match=r"is registered as several tools \['first', 'second'\]"
    ):
        await Executor(registry).run(twice, {"x": 1})
    assert await Executor(registry).run("second", {"x": 3}) == 3


@pytest.mark.parametrize("bad", [42, None, b"add"])
async def test_run_of_something_else_is_a_usage_error(bad: object) -> None:
    with pytest.raises(ToolUsageError) as caught:
        await Executor(Registry()).run(bad, {})  # type: ignore[arg-type]
    assert str(caught.value) == (
        f"run needs a tool name (str) or a function registered with @tool, got {type(bad).__name__}"
    )


async def test_run_with_arguments_that_are_not_a_mapping_says_so() -> None:
    _add()
    with pytest.raises(ToolUsageError) as caught:
        await Executor().run("add", [1, 2])  # type: ignore[arg-type]
    assert str(caught.value) == (
        "args must be a mapping of argument names to values, e.g. {'a': 1}, got list"
    )


@pytest.mark.parametrize("bad", [[], ["add"], "tools", object()])
def test_executor_rejects_a_registry_that_is_not_a_registry(bad: object) -> None:
    with pytest.raises(ToolUsageError) as caught:
        Executor(bad)  # type: ignore[arg-type]
    assert str(caught.value) == (
        f"Executor's registry must be a Registry, got {type(bad).__name__}: register "
        "functions with @tool (the default registry, used by Executor()) or "
        "Registry().tool, then pass that Registry"
    )


@pytest.mark.parametrize("bad", [True, "sqlite:///a.db", object()])
def test_executor_rejects_an_approval_that_is_not_a_port(bad: object) -> None:
    with pytest.raises(ToolUsageError) as caught:
        Executor(approval=bad)  # type: ignore[arg-type]
    assert str(caught.value) == (
        f"approval must be an ApprovalPort (an object with async request()), got "
        f"{type(bad).__name__}: use AutoApprove() in tests, DeferredApproval(store) for "
        "reviews (DeferredApproval.sqlite(path) for a file), or CallbackApproval(fn)"
    )


def test_run_sync_twins_return_values_and_results() -> None:
    add = _add()
    executor = Executor()
    assert executor.run_sync(add, {"a": 2, "b": 3}, timeout=5) == 5
    result = executor.run_call_sync(ToolCall(name="add", arguments={"a": 1, "b": 1}, id="c1"))
    assert (result.ok, result.value, result.call_id) == (True, 2, "c1")
    calls = [ToolCall(name="add", arguments={"a": i, "b": i}) for i in range(3)]
    assert [r.value for r in executor.run_many_sync(calls, concurrency=2)] == [0, 2, 4]


def test_sync_twins_share_one_long_lived_loop() -> None:
    loops: list[asyncio.AbstractEventLoop] = []

    @aa.tool
    async def where() -> str:
        """Which loop runs me."""
        loops.append(asyncio.get_running_loop())
        return "here"

    Executor().run_sync(where)
    aa.call(where)
    assert len(loops) == 2 and loops[0] is loops[1]
    assert isinstance(_sugar.shared_runner(), LoopRunner)


def test_shared_runner_is_replaced_once_closed() -> None:
    first = _sugar.shared_runner()
    first.close()
    try:
        assert _sugar.shared_runner() is not first
    finally:
        assert _sugar.shared_runner().closed is False


async def test_a_sync_twin_inside_a_running_loop_still_works() -> None:
    add = _add()
    assert Executor().run_sync(add, {"a": 1, "b": 1}) == 2  # blocks this loop; await instead


def test_call_runs_a_function_on_the_default_executor() -> None:
    add = _add()
    assert aa.call(add, a=1, b=2) == 3
    assert aa.call("add", a=5, b=5) == 10


async def test_acall_is_the_async_twin() -> None:
    add = _add()
    assert await aa.acall(add, a=4, b=4) == 8


def test_call_raises_like_run() -> None:
    @aa.tool(confirm=True)
    def wipe() -> str:
        """Wipe everything."""
        return "gone"

    with pytest.raises(ApprovalDenied):
        aa.call(wipe)


def test_configure_sets_the_default_executor_on_the_default_registry() -> None:
    @aa.tool(confirm=True)
    def wipe() -> str:
        """Wipe everything."""
        return "gone"

    executor = aa.configure(approval=AutoApprove())
    assert defaults.default_executor() is executor
    assert executor.registry is default_registry
    assert aa.call(wipe) == "gone"


def test_configure_accepts_an_explicit_registry() -> None:
    mine = Registry()
    assert aa.configure(mine).registry is mine


def test_configure_rejects_unknown_options() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'aproval'"):
        aa.configure(aproval=AutoApprove())
