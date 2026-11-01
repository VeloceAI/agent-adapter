"""The executor tells a running tool which attempt it is (`current_attempt()`)."""

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.exec import Executor
from agent_adapter.exec.pipeline import Attempt, current_attempt


async def no_sleep(_: float) -> None:
    return None


async def test_each_attempt_sees_its_number_the_total_and_its_call_id():
    seen: list[Attempt | None] = []

    async def flaky(ctx: ToolContext) -> str:
        """Fail once."""
        seen.append(current_attempt())
        if len(seen) == 1:
            raise RuntimeError("once")
        return ctx.call_id

    registry = Registry()
    registry.add(build_spec(flaky, retries=4))
    call_id = await Executor(registry, sleep=no_sleep).run("flaky", call_id="call_a")
    assert seen == [Attempt("call_a", 1, 5), Attempt("call_a", 2, 5)]
    assert call_id == "call_a" and current_attempt() is None
    assert seen[0] is not None and not seen[0].last and Attempt("x", 5, 5).last


async def test_a_direct_invoke_has_no_attempt():
    async def probe() -> bool:
        """Report whether an attempt is running."""
        return current_attempt() is None

    assert await build_spec(probe, retries=2).invoke({}) is True


async def test_a_direct_invoke_nested_in_an_attempt_sees_none():
    inner: list[Attempt | None] = []

    async def child() -> str:
        """Report the attempt."""
        inner.append(current_attempt())
        return "ok"

    child_spec = build_spec(child, retries=2)

    async def parent() -> Attempt | None:
        """Call the child directly, then report the attempt."""
        await child_spec.invoke({})
        await child_spec.call(child_spec.validate_args({}))
        return current_attempt()

    registry = Registry()
    registry.add(build_spec(parent, retries=1))
    after = await Executor(registry, sleep=no_sleep).run("parent", call_id="call_p")
    assert inner == [None, None]
    assert after == Attempt("call_p", 1, 2)  # restored after the nested calls


async def test_a_sync_tool_in_a_worker_thread_sees_its_attempt():
    def blocking() -> Attempt | None:
        """Report the attempt from a thread."""
        return current_attempt()

    registry = Registry()
    registry.add(build_spec(blocking))
    assert await Executor(registry).run("blocking", call_id="call_s") == Attempt("call_s", 1, 1)
