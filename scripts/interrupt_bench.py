"""interrupt-bench: what happens to bookings when callers talk over a voice agent.

Offline and seeded: a fake booking service that commits in two steps (reserve, then charge),
scripted interruptions with random timing, and the same scenarios for every contender.

    python scripts/interrupt_bench.py --runs 200 --seed 7

Contenders: `agent-adapter` (VoiceAgent with its defaults) and `bare-loop` (a plain asyncio
loop that cancels the turn task on interruption, the floor every voice stack starts from).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
)
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.testing import RecordingSpeaker

STEP_S = 0.02  # the gap between reserve and charge


class Service:
    """A booking service: `reserve` then `charge`; half-done means reserved but not charged."""

    def __init__(self) -> None:
        self.reserved: Counter[str] = Counter()
        self.charged: Counter[str] = Counter()

    async def book(self, seat: str) -> str:
        self.reserved[seat] += 1
        await asyncio.sleep(STEP_S)
        self.charged[seat] += 1
        return f"booking {seat}"

    def score(self) -> dict[str, int]:
        seats = set(self.reserved) | set(self.charged)
        return {
            "double": sum(1 for s in seats if self.charged[s] > 1),
            "half_done": sum(1 for s in seats if self.reserved[s] > self.charged[s]),
        }


def _reply(*calls: ToolCall, text: str = "") -> LLMResponse:
    message = Message(role="assistant", content=text, tool_calls=calls)
    finish = "tool_calls" if calls else "stop"
    return LLMResponse(provider="bench", model="bench", text=text, tool_calls=calls,
                       finish_reason=finish, message=message)  # fmt: skip


def _model(script: list[LLMResponse]) -> CallableModel:
    async def answer(request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0)
        return script.pop(0) if script else _reply(text="Done.")

    return CallableModel(answer, provider="bench", model="bench")


def _call(n: int) -> ToolCall:
    return ToolCall(name="book", arguments={"seat": "4A"}, id=f"c{n}")


# Each scenario: (model script, caller actions). The model re-issues the booking with a new id
# after an interruption, as Gemini Live and others were seen to do.
SCENARIOS: dict[str, Callable[[], list[LLMResponse]]] = {
    "cut-mid-booking": lambda: [_reply(_call(1)), _reply(text="Booked.")],
    "reissued-after-cut": lambda: [_reply(_call(1)), _reply(_call(2)), _reply(text="Booked.")],
    "duplicate-burst": lambda: [_reply(_call(1), _call(2)), _reply(text="Booked.")],
}


async def run_ours(script: list[LLMResponse], cut_after: float, service: Service) -> None:
    registry = Registry()
    registry.tool(effects="destructive", parallel_safe=True)(_named(service.book))
    agent = VoiceAgent(_model(script), Executor(registry), RecordingSpeaker())
    await agent.user_said("book 4A")
    await asyncio.sleep(cut_after)
    await agent.user_started_speaking()
    await agent.user_said("did it work?")
    await agent.wait()
    await agent.close()


async def run_bare(script: list[LLMResponse], cut_after: float, service: Service) -> None:
    async def turn() -> None:
        while script:
            reply = script.pop(0)
            if not reply.tool_calls:
                return
            await asyncio.gather(*(service.book(**c.arguments) for c in reply.tool_calls))

    task = asyncio.create_task(turn())
    await asyncio.sleep(cut_after)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await turn()  # the next turn goes on with what the model says next


def _named(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    async def book(seat: str) -> str:
        """Book a seat."""
        return await fn(seat)

    return book


CONTENDERS = {"agent-adapter": run_ours, "bare-loop": run_bare}


async def bench(runs: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    report: dict[str, Any] = {"runs": runs, "seed": seed, "results": {}}
    for scenario, script in SCENARIOS.items():
        cuts = [rng.uniform(0, STEP_S * 3) for _ in range(runs)]  # before, during and after
        for name, contender in CONTENDERS.items():
            totals: Counter[str] = Counter()
            times: list[float] = []
            for cut in cuts:
                service = Service()
                started = time.perf_counter()
                await contender(script(), cut, service)
                times.append((time.perf_counter() - started) * 1000)
                totals.update(service.score())
            report["results"][f"{scenario}/{name}"] = {
                "double_rate": totals["double"] / runs,
                "half_done_rate": totals["half_done"] / runs,
                "turn_p50_ms": round(statistics.median(times), 2),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()
    report = asyncio.run(bench(args.runs, args.seed))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"interrupt-bench: {args.runs} runs per scenario, seed {args.seed}\n")
    print(
        "| scenario / contender | double bookings | half-done | turn p50 ms (incl. the booking) |"
    )
    print("|---|---|---|---|")
    for key, row in report["results"].items():
        rates = f"{row['double_rate']:.1%} | {row['half_done_rate']:.1%}"
        print(f"| {key} | {rates} | {row['turn_p50_ms']} |")


if __name__ == "__main__":
    main()
