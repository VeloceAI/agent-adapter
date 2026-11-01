import asyncio
import re

from agent_adapter.core.context import ToolContext
from agent_adapter.runtime import current_context, new_id, use_context


def test_no_context_by_default():
    assert current_context() is None


async def test_use_context_sets_and_restores():
    outer, inner = ToolContext(call_id="outer"), ToolContext(call_id="inner")
    with use_context(outer) as ctx:
        assert ctx is outer and current_context() is outer
        with use_context(inner):
            assert current_context() is inner
        assert current_context() is outer
    assert current_context() is None


async def test_child_tasks_read_the_parent_value_and_writes_stay_local():
    parent = ToolContext(call_id="parent")

    async def child():
        seen = current_context()
        with use_context(ToolContext(call_id="child")):
            pass
        return seen

    with use_context(parent):
        seen = await asyncio.create_task(child())
        assert seen is parent and current_context() is parent


def test_use_context_restores_after_an_error():
    try:
        with use_context(ToolContext()):
            raise ValueError
    except ValueError:
        pass
    assert current_context() is None


def test_new_id_is_prefix_plus_hex():
    assert re.fullmatch(r"run_[0-9a-f]{32}", new_id("run_"))
    assert re.fullmatch(r"[0-9a-f]{32}", new_id())
    assert new_id("call_") != new_id("call_")
