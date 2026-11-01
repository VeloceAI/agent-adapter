import asyncio

import pytest

from agent_adapter import Registry, ToolError
from agent_adapter.errors import PlanInvalid
from agent_adapter.exec import Executor, Step
from agent_adapter.exec.plan import _plan_waves
from tests.exec.helpers import GraphBubbleUp

registry = Registry()
started: list[str] = []


@registry.tool(parallel_safe=True)
async def record(label: str, fail: bool = False, bubble: bool = False) -> str:
    """Record the label, then return it."""
    started.append(label)
    await asyncio.sleep(0)
    if bubble:
        raise GraphBubbleUp(label)
    if fail:
        raise ToolError(f"{label} failed")
    return label


@pytest.fixture(autouse=True)
def _reset():
    started.clear()


def _step(id, *deps, **args):
    return Step(id=id, tool="record", args={"label": id, **args}, depends_on=deps)


def test_waves_are_kahn_layers_in_step_order():
    steps = [_step("c", "a"), _step("a"), _step("d", "a", "b"), _step("b")]
    assert [[s.id for s in wave] for wave in _plan_waves(steps)] == [["a", "b"], ["c", "d"]]


def test_validation_lists_every_problem_together():
    with pytest.raises(PlanInvalid) as info:
        _plan_waves([_step("a"), _step("a"), _step("b", "missing"), _step("c", "nope")])
    assert info.value.problems == [
        "duplicate step id 'a'",
        "step 'b' depends on unknown step 'missing'",
        "step 'c' depends on unknown step 'nope'",
    ]


@pytest.mark.parametrize("steps", [[_step("a", "b"), _step("b", "a")], [_step("a", "a")]])
def test_cycles_are_rejected_before_anything_runs(steps):
    with pytest.raises(PlanInvalid, match="cycle"):
        _plan_waves(steps)


async def test_run_plan_runs_waves_and_returns_results_in_step_order():
    steps = [
        _step("rerank", "acts", "cases"),
        _step("acts"),
        _step("cases"),
        _step("sections", "acts"),
    ]
    results = await Executor(registry).run_plan(steps)
    assert list(results) == ["rerank", "acts", "cases", "sections"]
    assert all(r.ok for r in results.values())
    assert set(started[:2]) == {"acts", "cases"} and set(started[2:]) == {"rerank", "sections"}


async def test_run_plan_rejects_invalid_plans_without_running():
    with pytest.raises(PlanInvalid):
        await Executor(registry).run_plan([_step("a", "a")])
    assert started == []


async def test_callable_args_receive_only_their_dependencies():
    seen = {}

    def args(deps):
        seen.update(deps)
        return {"label": deps["a"].value + "+b"}

    steps = [_step("a"), _step("x"), Step(id="b", tool="record", args=args, depends_on=["a"])]
    results = await Executor(registry).run_plan(steps)
    assert results["b"].value == "a+b" and list(seen) == ["a"]


async def test_a_failed_dependency_blocks_its_dependents():
    steps = [_step("a", fail=True), _step("b", "a"), _step("c", "b"), _step("d")]
    results = await Executor(registry).run_plan(steps)
    assert [results[k].status for k in "abcd"] == ["error", "error", "error", "ok"]
    blocked = results["b"].error
    assert blocked.kind == "dependency-failed" and blocked.details == {"dependencies": ["a"]}
    assert "b" not in started and "c" not in started


async def test_a_failing_args_callable_is_the_step_error():
    def args(deps):
        raise KeyError("missing")

    steps = [Step(id="a", tool="record", args=args), _step("b", "a")]
    results = await Executor(registry).run_plan(steps)
    assert isinstance(results["a"].exception, KeyError)
    assert results["b"].error.kind == "dependency-failed"
    with pytest.raises(KeyError):
        await Executor(registry).run_plan(steps, on_error="fail_fast")


async def test_fail_fast_raises_the_first_error_unwrapped():
    steps = [_step("a", fail=True), _step("b", "a")]
    with pytest.raises(ToolError, match="a failed"):
        await Executor(registry).run_plan(steps, on_error="fail_fast")
    assert "b" not in started


async def test_control_flow_escapes_run_plan_unwrapped():
    steps = [_step("a", bubble=True), _step("b")]
    with pytest.raises(GraphBubbleUp):
        await Executor(registry).run_plan(steps)


async def test_plan_steps_share_one_run():
    ex = Executor(registry)
    ctx = ex.context(run_id="plan_run")
    seen = []

    class Spy:
        def emit(self, event):
            seen.append(event.run_id)

    ctx.sink = Spy()
    await ex.run_plan([_step("a"), _step("b", "a")], ctx)
    assert set(seen) == {"plan_run"}
