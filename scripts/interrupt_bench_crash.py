"""interrupt-bench, crash scenario: the process dies in the middle of a booking, then restarts.

The booking reserves, then charges; the crash lands after the reserve or after the charge.
`agent-adapter` restarts with `recover_tool_loop` from its checkpoint; `restart-turn` runs the
turn again from the start, as a loop without checkpoints must.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Registry, ToolCall
from agent_adapter.errors import RecoveryNeedsDecision
from agent_adapter.exec import recover_tool_loop, run_tool_loop
from agent_adapter.exec.checkpoints import MemoryCheckpointStore

KEY = b"interrupt-bench-checkpoint-key!!"
ASK = [Message(role="user", content="book 4A")]


class Crash(BaseException):
    """The process dies: nothing after it runs, not even `except Exception`."""


class Bookings:
    def __init__(self, crash_after: str) -> None:
        self.crash_after = crash_after  # "reserve", "charge" or "" (no crash)
        self.reserved = 0
        self.charged = 0

    async def book(self, seat: str) -> str:
        self.reserved += 1
        if self.crash_after == "reserve":
            self.crash_after = ""
            raise Crash
        self.charged += 1
        if self.crash_after == "charge":
            self.crash_after = ""
            raise Crash
        return f"booking {seat}"


def _model() -> CallableModel:
    async def answer(request: LLMRequest) -> LLMResponse:
        if request.messages[-1].role == "tool":
            reply = Message(role="assistant", content="Booked.")
            return LLMResponse(provider="bench", model="bench", text="Booked.", message=reply)
        call = ToolCall(name="book", arguments={"seat": "4A"}, id="c1")
        message = Message(role="assistant", tool_calls=(call,))
        return LLMResponse(provider="bench", model="bench", tool_calls=(call,),
                           finish_reason="tool_calls", message=message)  # fmt: skip

    return CallableModel(answer, provider="bench", model="bench")


def _registry(bookings: Bookings) -> Registry:
    registry = Registry()

    @registry.tool(effects="destructive")
    async def book(seat: str) -> str:
        """Book a seat."""
        return await bookings.book(seat)

    return registry


async def ours(bookings: Bookings) -> str:
    store, registry = MemoryCheckpointStore(), _registry(bookings)
    options: dict[str, Any] = {"checkpoints": store, "checkpoint_key": KEY}
    try:
        await run_tool_loop(_model(), registry, ASK, run_id="r1", **options)
        return "finished"
    except Crash:
        pass
    try:
        await recover_tool_loop(_model(), registry, "r1", **options)
    except RecoveryNeedsDecision:
        return "asked a person"
    return "recovered"


async def restart_turn(bookings: Bookings) -> str:
    registry = _registry(bookings)
    try:
        await run_tool_loop(_model(), registry, ASK)
        return "finished"
    except Crash:
        await run_tool_loop(_model(), registry, ASK)  # nothing saved: start the turn again
        return "re-ran"


CONTENDERS = {"agent-adapter": ours, "restart-turn": restart_turn}


async def crash_bench(runs: int, seed: int) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    points = [rng.choice(("reserve", "charge")) for _ in range(runs)]
    results: dict[str, dict[str, float]] = {}
    for name, contender in CONTENDERS.items():
        doubles = repeats = 0
        outcomes: Counter[str] = Counter()
        for point in points:
            bookings = Bookings(point)
            outcomes[await contender(bookings)] += 1
            doubles += bookings.charged > 1
            repeats += bookings.reserved > 1
        results[name] = {
            "double_charge_rate": doubles / runs,
            "booking_run_twice_rate": repeats / runs,
            "asked_a_person_rate": outcomes["asked a person"] / runs,
        }
    return results
