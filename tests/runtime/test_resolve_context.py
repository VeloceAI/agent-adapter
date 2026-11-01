from agent_adapter.core.context import ToolContext
from agent_adapter.runtime.current import resolve_context, use_context


def test_bound_context_wins_over_the_current_one():
    bound, current = ToolContext(run_id="b"), ToolContext(run_id="c")
    with use_context(current):
        assert resolve_context(bound) is bound
        assert resolve_context(None) is current


def test_no_context_gives_a_fresh_run_and_call():
    ctx = resolve_context(None)
    assert ctx.run_id.startswith("run_") and ctx.call_id.startswith("call_")
    assert resolve_context(None).run_id != ctx.run_id


def test_an_empty_run_id_is_filled_and_the_rest_kept():
    extras: dict[str, object] = {"k": 1}
    ctx = resolve_context(ToolContext(call_id="c1", thread_id="t", extras=extras))
    assert ctx.run_id.startswith("run_")
    assert (ctx.call_id, ctx.thread_id) == ("c1", "t") and ctx.extras is extras
