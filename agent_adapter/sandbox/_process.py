"""Run a host process with stdin, a hard time limit and capped output; kill it on cancel."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.sandbox._bootstrap import JOIN_S, drain

Kill = Callable[["subprocess.Popen[bytes]"], None]
MAX_STDERR = 16_000


@dataclass(frozen=True)
class Finished:
    """How a host process ended: its exit code, capped output and whether it was killed."""

    exit_code: int
    stdout: bytes
    stdout_over: bool  # more than the cap was printed
    stderr: bytes
    timed_out: bool
    duration_ms: float


def start(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    **popen: Any,
) -> subprocess.Popen[bytes]:
    """Start `argv` with piped stdin, stdout and stderr (no shell)."""
    return subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=cwd,
        **popen,
    )


def _feed(proc: subprocess.Popen[bytes], data: bytes) -> None:
    stdin = proc.stdin
    if stdin is None:  # pragma: no cover - start() always pipes stdin
        return
    try:
        stdin.write(data)
    except OSError:
        pass  # the process exited before reading all of it; its exit code tells why
    finally:
        with contextlib.suppress(OSError):
            stdin.close()


def wait(
    proc: subprocess.Popen[bytes],
    data: bytes,
    *,
    timeout_s: float,
    max_stdout: int,
    kill: Kill,
) -> Finished:
    """Feed `data`, wait up to `timeout_s`, then `kill`; keep at most `max_stdout` bytes."""
    started = time.monotonic()
    out: dict[str, Any] = {}
    err: dict[str, Any] = {}
    threads = [threading.Thread(target=_feed, args=(proc, data), daemon=True)]
    for target, cap, sink in ((proc.stdout, max_stdout, out), (proc.stderr, MAX_STDERR, err)):
        threads.append(threading.Thread(target=drain, args=(target, cap, sink), daemon=True))
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill(proc)
        proc.wait()
        exit_code = -1
    for thread in threads:
        thread.join(JOIN_S)
    return Finished(
        exit_code=exit_code,
        stdout=out.get("data", b""),
        stdout_over=bool(out.get("truncated")),
        stderr=err.get("data", b""),
        timed_out=timed_out,
        duration_ms=round((time.monotonic() - started) * 1000, 3),
    )


async def run_process(
    proc: subprocess.Popen[bytes], data: bytes, *, timeout_s: float, max_stdout: int, kill: Kill
) -> Finished:
    """`wait` in a worker thread; a cancelled caller kills the process before re-raising."""
    try:
        return await asyncio.to_thread(
            wait, proc, data, timeout_s=timeout_s, max_stdout=max_stdout, kill=kill
        )
    except asyncio.CancelledError:
        # Killing may block (Docker removes the container), so it runs off the loop.
        await asyncio.shield(asyncio.to_thread(kill, proc))
        raise
