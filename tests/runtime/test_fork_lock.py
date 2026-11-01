"""ForkSafeLock: a lock held when the process forks is free again in the child."""

import os
import weakref

import pytest

from agent_adapter.exec import _sugar, defaults
from agent_adapter.runtime import _fork, loop_runner
from agent_adapter.runtime._fork import ForkSafeLock


def test_a_held_lock_is_free_after_the_fork_hook(monkeypatch):
    lock = ForkSafeLock()
    monkeypatch.setattr(_fork, "_LOCKS", weakref.WeakSet([lock]))
    parents = lock._lock
    parents.acquire()  # held by a parent thread that will not exist in the child
    try:
        _fork.renew_locks_after_fork()  # what the at-fork hook does in the child
        assert not lock.locked()
        with lock:
            assert lock.locked()
    finally:
        parents.release()


def test_the_lock_releases_when_the_block_raises():
    lock = ForkSafeLock()
    with pytest.raises(RuntimeError), lock:
        raise RuntimeError("boom")
    assert not lock.locked()


def test_every_module_lock_is_fork_safe():
    for lock in (_sugar._runner_lock, defaults._lock, loop_runner._RUNNERS_LOCK):
        assert isinstance(lock, ForkSafeLock)
        assert lock in _fork._LOCKS


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() of a threaded process
def test_a_real_forked_child_can_take_a_lock_the_parent_holds():
    lock = ForkSafeLock()
    with lock:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            os._exit(0 if not lock.locked() else 1)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
