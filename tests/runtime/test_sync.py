import asyncio
import concurrent.futures
import contextvars
import threading

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime import LoopLocal, run_sync

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


async def add(a: int, b: int) -> int:
    await asyncio.sleep(0)
    return a + b


def test_runs_a_coroutine_without_a_loop():
    assert run_sync(add(1, 2)) == 3


def test_runs_a_zero_argument_callable_inside_the_loop():
    assert run_sync(lambda: add(2, 3)) == 5


def test_exceptions_propagate():
    async def boom():
        raise KeyError("k")

    with pytest.raises(KeyError):
        run_sync(boom())


def test_contextvars_are_visible_without_a_loop():
    async def read():
        return REQUEST_ID.get()

    token = REQUEST_ID.set("req-1")
    try:
        assert run_sync(read()) == "req-1"
    finally:
        REQUEST_ID.reset(token)


async def test_inside_a_running_loop_uses_a_worker_thread_and_copies_contextvars():
    caller = threading.get_ident()

    async def where():
        return threading.get_ident(), REQUEST_ID.get()

    REQUEST_ID.set("req-2")
    thread, value = run_sync(where())
    assert thread != caller and value == "req-2"


def test_timeout_cancels_the_work():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    with pytest.raises(TimeoutError):
        run_sync(slow(), timeout=0.01)
    assert cancelled == [True]


async def test_timeout_inside_a_running_loop():
    with pytest.raises(TimeoutError):
        run_sync(lambda: asyncio.sleep(5), timeout=0.01)


@pytest.mark.parametrize("work", [3, "text", None])
def test_rejects_non_awaitables(work):
    with pytest.raises(ToolUsageError, match="awaitable or a callable"):
        run_sync(work)


def test_rejects_callables_returning_non_awaitables():
    with pytest.raises(ToolUsageError, match="not awaitable"):
        run_sync(lambda: 3)


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_rejects_non_positive_timeouts(timeout):
    with pytest.raises(ToolUsageError, match="timeout must be a positive number"):
        run_sync(add(1, 1), timeout=timeout)


def test_loop_local_values_are_closed_before_the_loop_ends():
    closed = []

    async def close(value):
        closed.append(value)

    local = LoopLocal(object, close=close)

    async def use():
        return local.get()

    first = run_sync(use())
    second = run_sync(use())
    assert first is not second
    assert closed == [first, second]


def test_two_concurrent_calls_from_two_threads_get_their_own_values():
    """Exit gate (B6): both loops are alive at once and each sees only its own value."""
    created: list[object] = []
    closed: list[object] = []
    lock = threading.Lock()

    def factory():
        with lock:
            created.append(object())
            return created[-1]

    async def close(value):
        closed.append(value)

    local = LoopLocal(factory, close=close)
    barrier = threading.Barrier(2)
    results: dict[str, list[object]] = {}
    errors: list[BaseException] = []

    async def work(name: str) -> list[object]:
        seen = [local.get()]
        await asyncio.to_thread(barrier.wait, 5)  # both loops are alive at once
        seen.append(local.get())
        return seen

    def thread_main(name: str) -> None:
        try:
            results[name] = run_sync(work(name), timeout=10)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=thread_main, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []
    a, b = results["a"], results["b"]
    assert a[0] is a[1] and b[0] is b[1] and a[0] is not b[0]
    assert len(created) == 2 and sorted(map(id, closed)) == sorted(map(id, created))


@pytest.mark.parametrize("timeout", ["5", True, float("nan"), [1]])
def test_rejects_non_numeric_timeouts_and_closes_the_coroutine(timeout):
    coro = add(1, 1)
    with pytest.raises(ToolUsageError, match="timeout must be"):
        run_sync(coro, timeout=timeout)
    assert coro.cr_frame is None  # closed: no "never awaited" warning


async def test_a_blocked_worker_loop_is_abandoned_after_the_grace_period(monkeypatch):
    monkeypatch.setattr("agent_adapter.runtime.sync.CALLER_GRACE_S", 0.01)
    release = threading.Event()

    async def blocks_the_loop():
        release.wait(5)  # sync wait: asyncio.timeout inside the loop cannot fire

    try:
        with pytest.raises(TimeoutError, match="did not finish"):
            run_sync(blocks_the_loop(), timeout=0.02)
    finally:
        release.set()


async def test_work_finishing_just_after_the_caller_times_out_is_not_lost(monkeypatch):
    real_pool = concurrent.futures.ThreadPoolExecutor

    class LateFuture:
        """The caller's wait times out, but the work completes right after."""

        def __init__(self, future):
            self.future, self.first = future, True

        def result(self, timeout=None):
            if self.first:
                self.first = False
                self.future.result()  # let the work finish, then report a timeout
                raise concurrent.futures.TimeoutError
            return self.future.result()

        def done(self):
            return self.future.done()

    class Pool(real_pool):
        def submit(self, *args, **kwargs):
            return LateFuture(super().submit(*args, **kwargs))

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Pool)
    assert run_sync(add(2, 2), timeout=1) == 4

    async def boom():
        raise KeyError("late")

    with pytest.raises(KeyError):
        run_sync(boom(), timeout=1)
