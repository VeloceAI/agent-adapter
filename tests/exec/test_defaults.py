import pytest

from agent_adapter import default_registry
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import (
    Executor,
    default_executor,
    defaults,
    runner_for,
    set_default_executor,
)


@pytest.fixture(autouse=True)
def _fresh_default(monkeypatch):
    monkeypatch.setattr(defaults, "_default", None)


def test_default_executor_is_created_lazily_once():
    first = default_executor()
    assert isinstance(first, Executor) and first.registry is default_registry
    assert default_executor() is first


def test_set_default_executor_replaces_it():
    mine = Executor()
    set_default_executor(mine)
    assert default_executor() is mine


def test_set_default_executor_rejects_other_objects():
    with pytest.raises(ToolUsageError):
        set_default_executor(object())  # type: ignore[arg-type]


def test_runner_for_prefers_the_given_runner():
    class Runner:
        async def run_spec(self, spec, args=None, ctx=None, *, call_id=None):
            return None

    runner = Runner()
    assert runner_for(runner) is runner
    assert runner_for(None) is default_executor()
