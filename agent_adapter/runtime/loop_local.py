"""One object per event loop: SDK clients and asyncio primitives shared across loops."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar

from agent_adapter.core.errors import ToolUsageError, log_failure

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Values dropped in a forked child stay referenced: their finalizers would act on sockets
# the parent still uses (a TLS close_notify, a Redis QUIT).
_FORK_ORPHANS: list[object] = []


class LoopLocal(Generic[T]):
    """Lazily creates one `factory()` value per running event loop.

    Loop-bound objects (httpx/redis clients, `asyncio.Lock`, `Semaphore`, `Event`) break when
    reused from another loop, and every `run_sync` call runs a new loop. `run_sync` closes
    the values before its loop ends; hosts that run their own loops should
    `await LoopLocal.aclose_all()` before closing them. Values of loops closed without
    that are dropped (not closed) on the next `get()`.
    """

    _instances: ClassVar[weakref.WeakSet[LoopLocal[Any]]] = weakref.WeakSet()
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self, factory: Callable[[], T], *, close: Callable[[T], Awaitable[None]] | None = None
    ) -> None:
        self._factory = factory
        self._close = close
        self._values: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = (
            weakref.WeakKeyDictionary()
        )
        # Two run_sync calls may run loops in two threads at once.
        self._lock = threading.Lock()
        with LoopLocal._instances_lock:
            LoopLocal._instances.add(self)

    def get(self) -> T:
        """The value for the running loop, created on first use.

        Raises:
            ToolUsageError: called outside a running event loop.
        """
        loop = _running_loop("get")
        with self._lock:
            self._drop_closed()
            if loop not in self._values:
                self._values[loop] = self._factory()
            return self._values[loop]

    def _bound(self) -> bool:
        """True when the running loop already has a value."""
        loop = _running_loop("_bound")
        with self._lock:
            return loop in self._values

    def _drop_closed(self) -> None:
        # A value that references its loop (a contended Lock, a client) keeps the weak key
        # alive forever, so entries of closed loops are removed explicitly. Caller holds _lock.
        closed = [loop for loop in self._values if loop.is_closed()]
        for loop in closed:
            del self._values[loop]
        if closed:
            logger.debug("dropped %d loop-local value(s) of closed loops", len(closed))

    async def aclose(self) -> None:
        """Forget this loop's value and close it with `close` (if given).

        Raises:
            ToolUsageError: called outside a running event loop.
        """
        loop = _running_loop("aclose")
        with self._lock:
            if loop not in self._values:
                return
            value = self._values.pop(loop)
        if self._close is not None:
            await self._close(value)

    @classmethod
    async def aclose_all(cls) -> None:
        """Close every LoopLocal value bound to the running loop; failures are logged."""
        with LoopLocal._instances_lock:
            instances = list(LoopLocal._instances)
        for instance in instances:
            if not instance._bound():
                continue
            try:
                await instance.aclose()
            except Exception as exc:
                # Teardown must reach every instance; one failing close must not leak the rest.
                log_failure(logger, logging.WARNING, exc, "closing a loop-local value failed")

    @classmethod
    def _after_fork_in_child(cls) -> None:
        """Forget every value: clients and locks made before `fork()` belong to the parent."""
        cls._instances_lock = threading.Lock()
        for instance in list(cls._instances):
            _FORK_ORPHANS.append(instance._values)
            instance._values = weakref.WeakKeyDictionary()
            instance._lock = threading.Lock()


if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=LoopLocal._after_fork_in_child)


async def close_client(client: Any) -> None:
    """A `LoopLocal` close for factory-made clients: awaits `client.aclose()` if it has one."""
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()


def _running_loop(method: str) -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        raise ToolUsageError(f"LoopLocal.{method}() needs a running event loop") from None
