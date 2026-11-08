"""Small stand-ins for temporalio and dbos: just the API the adapters call."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import types
from typing import Any

WORKFLOW_ID: contextvars.ContextVar[str] = contextvars.ContextVar("workflow_id", default="")


def install_temporal(monkeypatch) -> types.SimpleNamespace:
    """Fake `temporalio` modules; returns the shared registry of activities and calls."""
    seen = types.SimpleNamespace(activities={}, calls=[], workers=[], defn={})

    activity = types.ModuleType("temporalio.activity")

    def activity_defn(*, name: str):
        def wrap(fn):
            seen.activities[name] = fn
            return fn

        return wrap

    activity.defn = activity_defn
    activity.info = lambda: types.SimpleNamespace(workflow_id=WORKFLOW_ID.get())

    workflow = types.ModuleType("temporalio.workflow")
    workflow.logger = logging.getLogger("fake.temporal.workflow")

    def defn(*, name: str, sandboxed: bool):
        seen.defn = {"name": name, "sandboxed": sandboxed}
        return lambda cls: cls

    def named(*, name: str):
        def wrap(fn):
            fn.validator = lambda check: setattr(fn, "checker", check)
            return fn

        return wrap

    async def execute_activity(name: str, arg: Any, **options: Any) -> Any:
        seen.calls.append((name, options))
        return await seen.activities[name](arg)

    async def wait_condition(check, *, timeout=None):
        async def poll():
            while not check():
                await asyncio.sleep(0)

        limit = timeout.total_seconds() if timeout is not None else None
        await asyncio.wait_for(poll(), limit)

    workflow.defn = defn
    workflow.run = lambda fn: fn
    workflow.update = named
    workflow.signal = named
    workflow.query = named
    workflow.execute_activity = execute_activity
    workflow.wait_condition = wait_condition

    common = types.ModuleType("temporalio.common")

    class RetryPolicy:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    common.RetryPolicy = RetryPolicy

    exceptions = types.ModuleType("temporalio.exceptions")

    class ApplicationError(Exception):
        def __init__(self, message: str, *, type: str = "", non_retryable: bool = False):
            super().__init__(message)
            self.type, self.non_retryable = type, non_retryable

    exceptions.ApplicationError = ApplicationError

    worker = types.ModuleType("temporalio.worker")

    class Worker:
        def __init__(self, client: Any, **kwargs: Any) -> None:
            self.client, self.kwargs = client, kwargs
            seen.workers.append(self)

    worker.Worker = Worker

    root = types.ModuleType("temporalio")
    for mod in (activity, workflow, common, exceptions, worker):
        short = mod.__name__.split(".")[1]
        setattr(root, short, mod)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(sys.modules, "temporalio", root)
    seen.ApplicationError = ApplicationError
    return seen


class FakeDbos:
    """The `DBOS` class surface the adapter uses, with messages and events in memory."""

    def __init__(self) -> None:
        self.steps: dict[str, dict[str, Any]] = {}
        self.workflows: dict[str, Any] = {}
        self.events: dict[tuple[str, str], Any] = {}
        self.inbox: dict[str, asyncio.Queue[Any]] = {}
        self.next_id = ""

    @property
    def workflow_id(self) -> str:
        return WORKFLOW_ID.get()

    def step(self, *, name: str, **options: Any):
        def wrap(fn):
            self.steps[name] = options
            return fn

        return wrap

    def workflow(self, *, name: str):
        def wrap(fn):
            self.workflows[name] = fn
            return fn

        return wrap

    def _queue(self, run_id: str) -> asyncio.Queue[Any]:
        return self.inbox.setdefault(run_id, asyncio.Queue())

    async def set_event_async(self, key: str, value: Any) -> None:
        self.events[(self.workflow_id, key)] = value

    async def get_event_async(self, run_id: str, key: str, timeout_seconds: float = 60) -> Any:
        return self.events.get((run_id, key))

    async def send_async(self, run_id: str, message: Any, topic: str | None = None) -> None:
        await self._queue(run_id).put(message)

    async def recv_async(self, topic: str | None = None, timeout_seconds: float = 60) -> Any:
        try:
            return await asyncio.wait_for(self._queue(self.workflow_id).get(), timeout_seconds)
        except TimeoutError:
            return None

    async def start_workflow_async(self, fn: Any, *args: Any) -> Any:
        token = WORKFLOW_ID.set(self.next_id)
        try:
            task = asyncio.ensure_future(fn(*args))
        finally:
            WORKFLOW_ID.reset(token)
        return types.SimpleNamespace(get_result=lambda: task, task=task)


def install_dbos(monkeypatch) -> FakeDbos:
    """A fake `dbos` module whose `DBOS` is one `FakeDbos`."""
    fake = FakeDbos()
    module = types.ModuleType("dbos")
    module.DBOS = fake

    class SetWorkflowID:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

        def __enter__(self) -> None:
            fake.next_id = self.run_id

        def __exit__(self, *exc: object) -> None:
            fake.next_id = ""

    module.SetWorkflowID = SetWorkflowID
    monkeypatch.setitem(sys.modules, "dbos", module)
    return fake
