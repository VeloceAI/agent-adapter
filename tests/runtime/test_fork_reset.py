"""After fork(), a child forgets the parent's loops, LoopLocal values and locks."""

import asyncio
import os
import sys

import pytest

from agent_adapter.runtime import LoopLocal, LoopRunner
from agent_adapter.runtime import loop_local as local_module
from agent_adapter.runtime import loop_runner as runner_module


async def add(a: int, b: int) -> int:
    await asyncio.sleep(0)
    return a + b


def test_loop_local_forgets_values_after_fork(monkeypatch):
    monkeypatch.setattr(local_module, "_FORK_ORPHANS", [])
    local: LoopLocal[object] = LoopLocal(object)

    async def value() -> object:
        return local.get()

    with LoopRunner() as runner:
        before = runner.run(value)
        LoopLocal._after_fork_in_child()
        after = runner.run(value)
    assert after is not before  # a fresh client, never the parent's
    assert any(before in values.values() for values in local_module._FORK_ORPHANS)


def test_runner_starts_a_fresh_loop_after_fork(monkeypatch):
    monkeypatch.setattr(runner_module, "_FORK_ORPHANS", [])
    runner = LoopRunner()

    async def loop_of() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    parent_loop = runner.run(loop_of)
    parent_thread = runner._thread
    runner._forget_after_fork()  # what the at-fork hook does in the child
    assert runner._loop is None and not runner.running
    assert [parent_loop] == runner_module._FORK_ORPHANS
    child_loop = runner.run(loop_of)
    assert child_loop is not parent_loop
    runner.close()
    # the "parent" loop of this simulation still runs in this process: stop it by hand
    parent_loop.call_soon_threadsafe(parent_loop.stop)
    assert parent_thread is not None
    parent_thread.join(timeout=5)
    parent_loop.close()


def test_module_hook_resets_every_runner(monkeypatch):
    seen: list[LoopRunner] = []
    runner = LoopRunner()
    monkeypatch.setattr(runner_module, "_RUNNERS", runner_module.weakref.WeakSet([runner]))
    monkeypatch.setattr(LoopRunner, "_forget_after_fork", lambda self: seen.append(self))
    runner_module._after_fork_in_child()
    assert seen == [runner]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() of a threaded process
def test_a_real_forked_child_runs_on_its_own_loop():
    with LoopRunner() as runner:
        runner.run(add(1, 1))
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            code = 0 if runner.run(add(2, 3), timeout=5) == 5 else 1
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert runner.run(add(1, 2)) == 3  # the parent's loop is untouched
    assert sys.platform != "win32"
