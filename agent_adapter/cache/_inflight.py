"""Callers of one cache key at the same moment share one real call.

The first caller (the leader) runs the call; the others await its outcome. If the leader
fails, the waiters get the same error: retries already ran inside the leader, so running
again at once would most likely fail again. If the leader is cancelled, hits control flow
(an approval pause) or has nothing shareable, one waiter takes over and runs the call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from agent_adapter.core.control import control_flow_member
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.runtime.loop_local import LoopLocal

T = TypeVar("T")
NOTHING: Any = object()  # `handout` result: nothing for the waiters, one of them runs the call

_Slot = tuple[int, str]


@dataclass(frozen=True)
class _Outcome:
    payload: Any = NOTHING
    error: BaseException | None = None


class _Flight:
    def __init__(self) -> None:
        self.future: asyncio.Future[_Outcome] = asyncio.get_running_loop().create_future()
        self.task = asyncio.current_task()

    def settle(self, outcome: _Outcome) -> None:
        self.future.set_result(outcome)


# One map per event loop: futures are loop-bound and every run_sync call runs a new loop.
_FLIGHTS: LoopLocal[dict[_Slot, _Flight]] = LoopLocal(dict)


async def share(
    port: object,
    key: str,
    lead: Callable[[], Awaitable[T]],
    *,
    handout: Callable[[T], Any],
    follow: Callable[[Any, float], T],
    hold: Callable[[Callable[[], None]], None] | None = None,
) -> T:
    """`lead()` for the first caller of `(port, key)`; the others get `follow(payload, waited_ms)`.

    `handout(result)` turns the leader's result into the waiters' payload (or `NOTHING`).
    `hold`, when given, receives a release callback instead of the flight being forgotten
    when `lead()` returns: late callers keep getting the payload until the caller releases
    it (after storing the result in the cache, so nobody slips between the two).
    """
    if not key:
        return await lead()
    flights = _FLIGHTS.get()
    slot = (id(port), key)
    started = asyncio.get_running_loop().time()
    while (flight := flights.get(slot)) is not None and flight.task is not asyncio.current_task():
        # A cancelled waiter must not cancel the outcome the other waiters share.
        outcome = await asyncio.shield(flight.future)
        if outcome.error is not None:
            raise outcome.error
        if outcome.payload is not NOTHING:
            waited_ms = (asyncio.get_running_loop().time() - started) * 1000
            return follow(outcome.payload, waited_ms)
    if flight is not None:
        return await lead()  # a nested call of the leader's own task: waiting would deadlock
    flight = _Flight()
    flights[slot] = flight
    try:
        result = await lead()
    except BaseException as exc:
        flight.settle(_Outcome(error=exc if _shared_error(exc) else None))
        _forget(flights, slot, flight)
        raise
    payload: Any = NOTHING
    try:
        payload = handout(result)
    finally:
        flight.settle(_Outcome(payload=payload))
        if hold is None or payload is NOTHING:
            _forget(flights, slot, flight)
    if hold is not None and payload is not NOTHING:
        hold(lambda: _forget(flights, slot, flight))
    return result


def _shared_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, Exception)
        and control_flow_member(exc) is None
        and not isinstance(exc, ApprovalPending)
    )


def _forget(flights: dict[_Slot, _Flight], slot: _Slot, flight: _Flight) -> None:
    if flights.get(slot) is flight:
        del flights[slot]
