"""LoopRunner review fixes: bounded close at exit, a warning when sync code blocks a loop."""

import asyncio
import logging
import threading
import time

from agent_adapter.runtime import loop_runner
from agent_adapter.runtime.loop_runner import LoopRunner


async def _one() -> int:
    return 1


def test_close_is_bounded_when_the_loop_blocks_in_sync_code(monkeypatch, caplog):
    monkeypatch.setattr(loop_runner, "CALLER_GRACE_S", 0.2)
    runner = LoopRunner(name="fx-blocked")
    release = threading.Event()

    async def stuck() -> None:
        release.wait(10)  # blocks the loop thread itself

    worker = threading.Thread(target=lambda: _swallow(runner, stuck), daemon=True)
    worker.start()
    time.sleep(0.2)
    started = time.monotonic()
    runner.close(timeout=0.2)
    took = time.monotonic() - started
    release.set()
    worker.join(5)
    thread, loop = runner._thread, runner._loop
    assert thread is not None and loop is not None
    thread.join(5)  # the loop leaves run_forever once unblocked
    loop.run_until_complete(asyncio.sleep(0.05))  # let the cancelled shutdown task finish
    loop.close()
    assert took < 2.0  # was timeout + 30 s for shutdown, then 30 s for the join
    assert "did not stop" in caplog.text or "shutdown did not finish" in caplog.text


def _swallow(runner: LoopRunner, work) -> None:
    try:
        runner.run(work, timeout=5)
    except Exception as exc:  # the runner was closed under it; only timing is tested here
        logging.getLogger(__name__).debug("work ended: %r", exc)


async def test_run_from_a_running_loop_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(loop_runner, "_warned_blocking", False)
    caplog.set_level(logging.WARNING, logger="agent_adapter.runtime.loop_runner")
    with LoopRunner(name="fx-warn") as runner:
        assert runner.run(_one, timeout=5) == 1
        assert runner.run(_one, timeout=5) == 1
    warnings = [r for r in caplog.records if "called from a running event loop" in r.message]
    assert len(warnings) == 1


def test_run_from_plain_sync_code_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(loop_runner, "_warned_blocking", False)
    with LoopRunner(name="fx-quiet") as runner:
        assert runner.run(_one, timeout=5) == 1
    assert "called from a running event loop" not in caplog.text
