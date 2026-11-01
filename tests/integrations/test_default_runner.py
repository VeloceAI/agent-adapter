"""Exported tools given no executor find the default executor at call time."""

import pytest

from agent_adapter.core.tool import build_spec
from agent_adapter.exec.defaults import default_executor, set_default_executor
from agent_adapter.exec.executor import Executor
from agent_adapter.integrations._default_runner import DEFAULT_RUNNER, DefaultRunner, lazy_runner
from tests.fakes.approval import ScriptedApproval
from tests.fakes.hooks import RecordingHook


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


SPEC = build_spec(add, name="add")


@pytest.fixture
def restore_default():
    previous = default_executor()
    yield
    set_default_executor(previous)


def test_lazy_runner_keeps_a_given_executor():
    executor = Executor()
    assert lazy_runner(executor) is executor
    assert lazy_runner(None) is DEFAULT_RUNNER


async def test_default_runner_uses_a_default_set_after_export(restore_default):
    runner = lazy_runner(None)  # what an exporter keeps
    hook = RecordingHook()
    set_default_executor(Executor(hooks=[hook]))

    assert await runner.run_spec(SPEC, {"a": 1, "b": 2}) == 3
    assert hook.phases() == ["before_tool", "after_tool"]


async def test_default_runner_with_a_port_keeps_it_and_follows_the_default(restore_default):
    port = ScriptedApproval([])
    runner = DefaultRunner(approval=port)
    first = runner.executor()
    assert first.approval is port
    assert runner.executor() is first  # reused while the default is unchanged

    hook = RecordingHook()
    set_default_executor(Executor(hooks=[hook]))
    second = runner.executor()
    assert second is not first
    assert second.approval is port
    assert await runner.run_spec(SPEC, {"a": 2, "b": 2}) == 4
    assert hook.phases() == ["before_tool", "after_tool"]


def test_default_runner_without_a_port_is_the_default_itself(restore_default):
    executor = Executor()
    set_default_executor(executor)
    assert DEFAULT_RUNNER.executor() is executor
    assert repr(DEFAULT_RUNNER) == "DefaultRunner(approval=None)"
