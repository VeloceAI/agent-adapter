"""LoopRunner: one warm loop for sync callers; timeouts, close, context and misuse."""

import asyncio
import contextvars
import threading
import time

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime import LoopLocal, LoopRunner
from agent_adapter.runtime import loop_runner as module

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


async def add(a: int, b: int) -> int:
    await asyncio.sleep(0)
    return a + b


@pytest.fixture
def runner():
    found = LoopRunner(name="test-loop")
    yield found
    found.close()


def test_run_returns_results_on_one_loop(runner):
    async def loop_of() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert runner.run(add(1, 2)) == 3
    assert runner.run(lambda: add(2, 3)) == 5
    assert runner.run(loop_of) is runner.run(loop_of)
    assert runner.running


def test_loop_starts_lazily_in_a_daemon_thread():
    runner = LoopRunner()
    assert not runner.running
    try:
        assert runner.run(add(1, 1)) == 2
        assert runner._thread is not None and runner._thread.daemon
    finally:
        runner.close()


def test_loop_local_values_stay_warm_and_close_on_close():
    closed: list[object] = []

    async def close(value: object) -> None:
        closed.append(value)

    local: LoopLocal[object] = LoopLocal(object, close=close)

    async def value() -> object:
        return local.get()

    runner = LoopRunner()
    first, second = runner.run(value), runner.run(value)
    assert first is second and closed == []
    runner.close()
    assert closed == [first]
    assert not runner.running and runner.closed


def test_exceptions_propagate(runner):
    async def boom() -> None:
        raise KeyError("k")

    with pytest.raises(KeyError):
        runner.run(boom())


def test_contextvars_of_the_caller_are_visible(runner):
    async def read() -> str:
        return REQUEST_ID.get()

    token = REQUEST_ID.set("req-7")
    try:
        assert runner.run(read) == "req-7"
    finally:
        REQUEST_ID.reset(token)
    assert runner.run(read) == "-"


def test_timeout_cancels_the_work(runner):
    with pytest.raises(TimeoutError):
        runner.run(asyncio.sleep(5), timeout=0.05)
    assert runner.run(add(1, 1)) == 2  # the loop survives


def test_work_blocking_the_loop_times_out_for_the_caller(runner, monkeypatch):
    monkeypatch.setattr(module, "CALLER_GRACE_S", 0.05)

    async def blocks() -> None:
        time.sleep(0.4)

    with pytest.raises(TimeoutError, match="did not finish"):
        runner.run(blocks, timeout=0.05)


def test_many_threads_share_the_runner(runner):
    results: list[int] = []

    def call(n: int) -> None:
        results.append(runner.run(add(n, 1)))

    threads = [threading.Thread(target=call, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == list(range(1, 9))


@pytest.mark.parametrize("bad", [None, 3, "x"])
def test_bad_work_is_refused(runner, bad):
    with pytest.raises(ToolUsageError, match="awaitable or a callable"):
        runner.run(bad)


def test_callable_returning_non_awaitable_is_refused(runner):
    with pytest.raises(ToolUsageError, match="not awaitable"):
        runner.run(lambda: 3)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), True, "1"])
def test_bad_timeout_is_refused_and_the_coroutine_closed(runner, bad):
    with pytest.raises(ToolUsageError, match="timeout"):
        runner.run(add(1, 2), timeout=bad)


def test_run_after_close_is_refused():
    runner = LoopRunner()
    runner.run(add(1, 1))
    runner.close()
    runner.close()  # idempotent
    with pytest.raises(ToolUsageError, match="is closed"):
        runner.run(add(1, 2))


def test_close_of_an_unstarted_runner_is_a_no_op():
    runner = LoopRunner()
    runner.close()
    assert runner.closed and not runner.running


def test_close_cancels_late_work_and_the_caller_hears_of_it():
    runner = LoopRunner()
    errors: list[BaseException] = []

    def call() -> None:
        try:
            runner.run(asyncio.sleep(30))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    while not runner.running:
        time.sleep(0.01)
    time.sleep(0.05)
    runner.close(timeout=0.05)
    thread.join(timeout=5)
    assert len(errors) == 1 and isinstance(errors[0], ToolUsageError)
    assert "closed while this work ran" in str(errors[0])


def test_close_waits_for_in_flight_work():
    runner = LoopRunner()
    results: list[int] = []

    async def slow() -> int:
        await asyncio.sleep(0.1)
        return 7

    thread = threading.Thread(target=lambda: results.append(runner.run(slow)))
    thread.start()
    while not runner.running:
        time.sleep(0.01)
    time.sleep(0.02)
    runner.close(timeout=5)
    thread.join(timeout=5)
    assert results == [7]


def test_close_timeout_must_be_positive(runner):
    with pytest.raises(ToolUsageError, match="close timeout"):
        runner.close(timeout=0)


def test_run_and_close_inside_the_runner_loop_are_refused(runner):
    async def nested() -> str:
        inner = add(1, 1)
        with pytest.raises(ToolUsageError, match="deadlock"):
            runner.run(inner)
        with pytest.raises(ToolUsageError, match="from its own loop"):
            runner.close()
        return "ok"

    assert runner.run(nested) == "ok"


def test_with_block_closes():
    with LoopRunner() as runner:
        assert runner.run(add(2, 2)) == 4
    assert runner.closed


def test_a_dead_loop_thread_is_replaced(runner, caplog):
    runner.run(add(1, 1))
    old = runner._loop
    assert old is not None and runner._thread is not None
    old.call_soon_threadsafe(old.stop)
    runner._thread.join(timeout=5)
    assert runner.run(add(1, 2)) == 3
    assert runner._loop is not old and old.is_closed()
    assert "thread died" in caplog.text


def test_cancel_after_the_loop_closed_is_logged_not_raised(caplog):
    loop = asyncio.new_event_loop()
    loop.close()
    future: module.concurrent.futures.Future[int] = module.concurrent.futures.Future()
    future.set_running_or_notify_cancel()
    with caplog.at_level("DEBUG", logger=module.__name__):
        module._cancel(loop, future, [])
    assert "loop already closed" in caplog.text


def test_at_exit_closes_every_live_runner(monkeypatch):
    monkeypatch.setattr(module, "_RUNNERS", module.weakref.WeakSet())
    first, second = LoopRunner(), LoopRunner()
    first.run(add(1, 1))
    module._close_at_exit()
    assert first.closed and second.closed


def test_at_exit_logs_a_failing_close(monkeypatch, caplog):
    monkeypatch.setattr(module, "_RUNNERS", module.weakref.WeakSet())
    runner = LoopRunner()

    def fail(timeout: float) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "close", fail)
    module._close_at_exit()
    assert "closing a LoopRunner at exit failed" in caplog.text
    monkeypatch.delattr(runner, "close")
    runner.close()


def test_a_crashed_loop_is_logged_and_replaced(runner, caplog):
    runner.run(add(1, 1))
    old_loop, old_thread = runner._loop, runner._thread
    assert old_loop is not None and old_thread is not None

    def crash() -> None:
        raise SystemExit(3)

    old_loop.call_soon_threadsafe(crash)
    old_thread.join(timeout=5)
    assert "its loop crashed" in caplog.text
    assert runner.run(add(2, 2)) == 4


def test_caller_interrupt_cancels_the_work(runner, monkeypatch):
    started: list[bool] = []

    class Interrupted(module.concurrent.futures.Future):  # type: ignore[type-arg]
        def result(self, timeout: float | None = None) -> object:
            while not started:
                time.sleep(0.01)
            raise KeyboardInterrupt

    async def slow() -> None:
        started.append(True)
        await asyncio.sleep(30)

    monkeypatch.setattr(module.concurrent.futures, "Future", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(slow)
    monkeypatch.undo()
    runner.close(timeout=5)  # returns promptly: the work was cancelled


def test_settle_and_cancel_edge_cases():
    done: module.concurrent.futures.Future[int] = module.concurrent.futures.Future()
    done.set_result(1)
    module._settle(None, done, None)  # type: ignore[arg-type]
    assert done.result() == 1
    pending: module.concurrent.futures.Future[int] = module.concurrent.futures.Future()
    loop = asyncio.new_event_loop()
    try:
        module._cancel(loop, pending, [])
        assert pending.cancelled()  # cancelled before it ever started
    finally:
        loop.close()
