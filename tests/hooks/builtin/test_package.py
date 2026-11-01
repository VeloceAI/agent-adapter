import pytest

import agent_adapter.hooks as hooks
import agent_adapter.hooks.builtin as builtin
from agent_adapter.core.errors import GuardrailBlocked
from tests.hooks.builtin.helpers import ok, tool_inv


def test_built_ins_are_exported_from_both_packages():
    names = {"LoggingHook", "BudgetHook", "GuardrailHook", "RateLimitHook", "Rate"}
    assert names <= set(builtin.__all__) and names <= set(hooks.__all__)
    for name in names:
        assert getattr(hooks, name) is getattr(builtin, name)
    assert all(hasattr(builtin, name) for name in builtin.__all__)
    assert all(hasattr(hooks, name) for name in hooks.__all__)


def test_built_ins_subclass_hook():
    for cls in (hooks.LoggingHook, hooks.BudgetHook, hooks.GuardrailHook, hooks.RateLimitHook):
        assert issubclass(cls, hooks.Hook) and cls.name


async def test_built_ins_compose_through_the_dispatchers():
    chain = [
        hooks.LoggingHook(),
        hooks.RateLimitHook({}),
        hooks.GuardrailHook(lambda stage, payload: "password" not in payload, on=["args"]),
    ]
    inv = tool_inv()
    assert await hooks.run_after(chain, "after_tool", inv, ok()) == ok()
    with pytest.raises(GuardrailBlocked):
        await hooks.run_before(chain, "before_tool", inv)
