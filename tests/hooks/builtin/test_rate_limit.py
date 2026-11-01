import asyncio
import threading

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.rate_limit import Rate, RateLimitHook
from agent_adapter.runtime import run_sync
from tests.fakes.clock import FakeClock
from tests.hooks.builtin.helpers import llm_inv, tool_inv


def limited(clock: FakeClock, **rates: Rate) -> RateLimitHook:
    return RateLimitHook(rates, clock=clock, sleep=clock.sleep)


async def test_burst_then_waits_at_the_refill_rate():
    clock = FakeClock()
    hook = limited(clock, tool=Rate(calls=2, per=1.0))
    for _ in range(4):
        assert await hook.before_tool(tool_inv()) is None
    assert clock.sleeps == [0.5, 0.5]


async def test_tokens_refill_up_to_capacity_only():
    clock = FakeClock()
    hook = limited(clock, tool=Rate(2, 1.0))
    await hook.before_tool(tool_inv())
    clock.advance(100)
    for _ in range(3):
        await hook.before_tool(tool_inv())
    assert clock.sleeps == [0.5]


async def test_buckets_are_per_kind_and_llm_has_its_own():
    clock = FakeClock()
    hook = limited(clock, search=Rate(1, 2.0), llm=Rate(1, 4.0))
    await hook.before_tool(tool_inv(kind="search"))
    await hook.before_llm(llm_inv())
    await hook.before_tool(tool_inv(kind="search"))  # waits 2 s
    await hook.before_llm(llm_inv())  # 2 s of 4 already passed: waits 2 s
    assert clock.sleeps == [2.0, 2.0]


async def test_kinds_without_a_rate_never_wait():
    clock = FakeClock()
    hook = limited(clock, search=Rate(1, 60.0))
    for _ in range(5):
        await hook.before_tool(tool_inv())  # kind "tool"
        await hook.before_llm(llm_inv())
    assert clock.sleeps == []


class BlockingSleep:
    """Records each wait and blocks until released, so waits can be cancelled mid-queue."""

    def __init__(self) -> None:
        self.waits: list[float] = []
        self.release = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        await self.release.wait()


async def queue(hook: RateLimitHook, n: int) -> list[asyncio.Task]:
    tasks = []
    for _ in range(n):
        tasks.append(asyncio.create_task(hook.before_tool(tool_inv())))
        await asyncio.sleep(0)
    return tasks


async def cancel(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    ("cancelled", "d_waits"),
    [
        (1, 3.0),  # B is mid-queue: C keeps its slot at 2 s, so D must not reuse 2 s
        (2, 2.0),  # C is the tail: its slot is rolled back and D takes it
    ],
)
async def test_a_cancelled_wait_never_double_books(cancelled, d_waits):
    sleep = BlockingSleep()
    hook = RateLimitHook({"tool": Rate(1, 1.0)}, clock=lambda: 0.0, sleep=sleep)
    a, b, c = await queue(hook, 3)  # A runs now; B waits 1 s; C waits 2 s
    await cancel((a, b, c)[cancelled])
    (d,) = await queue(hook, 1)
    assert sleep.waits == [1.0, 2.0, d_waits]
    sleep.release.set()
    await asyncio.gather(*(t for t in (a, b, c, d) if not t.cancelled()))


def test_shared_by_two_loops_in_two_threads():
    sleeps: list[float] = []
    lock = threading.Lock()

    async def sleep(seconds: float) -> None:
        with lock:
            sleeps.append(seconds)
        await asyncio.sleep(0)

    hook = RateLimitHook({"tool": Rate(2, 1.0)}, clock=lambda: 0.0, sleep=sleep)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    async def two_calls() -> None:
        await hook.before_tool(tool_inv())
        await hook.before_tool(tool_inv())

    def worker() -> None:
        barrier.wait()
        try:
            run_sync(two_calls, timeout=5)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert sorted(sleeps) == [0.5, 1.0]  # one bucket across both loops


@pytest.mark.parametrize(
    ("calls", "per"), [(0, 1.0), (-1, 1.0), (True, 1.0), (1.5, 1.0), (1, 0), (1, float("inf"))]
)
def test_bad_rates_are_rejected(calls, per):
    with pytest.raises(ConfigError):
        Rate(calls, per)


def test_per_kind_values_must_be_rates():
    with pytest.raises(ConfigError, match="Rate"):
        RateLimitHook({"tool": (1, 1.0)})  # type: ignore[dict-item]


def test_rates_are_exposed():
    rates = {"tool": Rate(3, 1.0)}
    assert RateLimitHook(rates).rates == rates
