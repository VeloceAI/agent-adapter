"""Work counts for performance regression tests: lines of code run, never wall-clock seconds.

A line count is deterministic, so it does not flake under machine load, and it still grows
quadratically or exponentially with the bugs these tests guard against. Workloads run in a
child interpreter (`run_child`) so a regression fails at its cap (or the hard timeout)
instead of hanging the suite, and so tracing never displaces the parent's coverage tracer.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# A backstop for work the line count cannot see (C code); normal runs take a few seconds.
CHILD_TIMEOUT_S = 120.0
_TOOL_ID = 4  # a sys.monitoring tool id no debugger, coverage tool or profiler claims


class WorkExceeded(BaseException):
    """The counted block ran more lines than its cap (a BaseException: never swallowed)."""


class Work:
    """Lines run so far inside a `counting` block."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.lines = 0

    def tick(self) -> None:
        self.lines += 1
        if self.lines > self.cap:
            raise WorkExceeded(f"more than {self.cap} lines run")


@contextmanager
def counting(cap: int, packages: tuple[str, ...] = ("agent_adapter",)) -> Iterator[Work]:
    """Count lines run in `packages` inside the block; raise `WorkExceeded` past `cap`."""
    roots = tuple(_package_dir(name) for name in packages)
    work = Work(cap)
    if sys.version_info >= (3, 12):
        with _monitoring(work, roots):
            yield work
    else:  # Python 3.11: no sys.monitoring
        with _tracing(work, roots):
            yield work


def _package_dir(name: str) -> str:
    module = importlib.import_module(name)
    return os.path.dirname(os.path.abspath(module.__file__ or "")) + os.sep


@contextmanager
def _monitoring(work: Work, roots: tuple[str, ...]) -> Iterator[None]:
    monitoring: Any = sys.monitoring  # type: ignore[attr-defined,unused-ignore]  # 3.12+
    line = monitoring.events.LINE

    def on_line(code: Any, _: int) -> Any:
        if not code.co_filename.startswith(roots):
            return monitoring.DISABLE  # this location never fires again: no cost outside
        work.tick()
        return None

    monitoring.use_tool_id(_TOOL_ID, "work-count")
    monitoring.register_callback(_TOOL_ID, line, on_line)
    monitoring.set_events(_TOOL_ID, line)
    try:
        yield
    finally:
        monitoring.set_events(_TOOL_ID, monitoring.events.NO_EVENTS)
        monitoring.register_callback(_TOOL_ID, line, None)
        monitoring.free_tool_id(_TOOL_ID)


@contextmanager
def _tracing(work: Work, roots: tuple[str, ...]) -> Iterator[None]:
    def local(frame: Any, event: str, _: Any) -> Any:
        if event == "line":
            work.tick()
        return local

    def calls(frame: Any, _event: str, _arg: Any) -> Any:
        return local if frame.f_code.co_filename.startswith(roots) else None

    sys.settrace(calls)
    try:
        yield
    finally:
        sys.settrace(None)


def run_child(script: str, *, timeout: float = CHILD_TIMEOUT_S) -> dict[str, Any]:
    """Run `script` in a fresh interpreter and return the JSON object on its last stdout line.

    agent_adapter is imported from `$AA_IMPORT_ROOT` when set (to replay an old source tree
    against today's tests), else from this repository; `tests` always comes from here.
    """
    old = os.environ.get("AA_IMPORT_ROOT")
    paths = [os.path.abspath(old)] if old else []
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*paths, str(ROOT)])}
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=paths[0] if paths else ROOT,  # `-c` puts the working directory first on sys.path
        check=False,
    )
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise AssertionError(f"child exited {done.returncode}: {tail[0]}")
    return json.loads(done.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]
