"""interrupt-bench on the real frameworks: Pipecat's and LiveKit's own tool-call and
interruption code, with a scripted model and the same booking service as interrupt_bench.py.

    pip install "agent-adapter[livekit,pipecat]"   # their own environment
    python scripts/interrupt_bench_frameworks.py --runs 100

The model is scripted, nothing else: the framework runs the tool, the caller cuts in at a
random moment while the booking is half-way (reserved, not yet charged), then asks "did it
work?". A model that never saw the booking's result books
again (with a new call id), as real models were seen to do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from interrupt_bench import STEP_S, Service

from agent_adapter import Registry
from agent_adapter.exec import Executor

Runner = Callable[["Watched", float, bool], Awaitable[None]]


class Watched(Service):
    """The bench's booking service, telling the caller when a booking has started."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def book(self, seat: str) -> str:
        self.reserved[seat] += 1
        self.started.set()
        await asyncio.sleep(STEP_S)
        self.charged[seat] += 1
        return f"booking {seat}"


async def _cut_mid_booking(service: Watched, cut: float) -> None:
    await asyncio.wait_for(service.started.wait(), timeout=5)
    await asyncio.sleep(cut)


def _registry(service: Watched) -> Registry:
    registry = Registry()

    @registry.tool(effects="destructive")
    async def book(seat: str) -> str:
        """Book a seat."""
        return await service.book(seat)

    return registry


# --- Pipecat ---------------------------------------------------------------------------------


async def pipecat_run(service: Watched, cut: float, ours: bool) -> None:
    from pipecat.frames.frames import FunctionCallFromLLM, InterruptionFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.llm_service import LLMService

    from agent_adapter.integrations.pipecat import pipecat_tools

    class Bench(LLMService):  # a bare service would hold every frame, StartFrame included
        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    seen: list[object] = []  # results that reached the model's context
    llm = Bench()
    if ours:
        pipecat_tools(llm, _registry(service), executor=Executor(_registry(service)))
        handlers = {name: item.handler for name, item in llm._functions.items()}

        def keep(name: str) -> Any:
            async def handler(params: Any) -> None:
                done = params.result_callback

                async def note(result: object, **kw: object) -> None:
                    seen.append(result)
                    await done(result, **kw)

                params.result_callback = note
                await handlers[name](params)

            return handler

        for name, item in llm._functions.items():
            item.handler = keep(name)
    else:

        async def book(params: Any) -> None:
            result = await service.book(**params.arguments)
            seen.append(result)
            await params.result_callback(result)

        llm.register_function("book", book)  # Pipecat's default: cancel_on_interruption=True

    task = PipelineTask(Pipeline([llm]), cancel_on_idle_timeout=False)
    started = asyncio.Event()

    @task.event_handler("on_pipeline_started")
    async def on_started(_task: object, _frame: object) -> None:
        started.set()

    running = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    await started.wait()

    def call(n: int) -> list[FunctionCallFromLLM]:
        return [FunctionCallFromLLM(function_name="book", tool_call_id=f"c{n}",
                                    arguments={"seat": "4A"}, context=LLMContext())]  # fmt: skip

    await llm.run_function_calls(call(1))
    await _cut_mid_booking(service, cut)
    await task.queue_frame(InterruptionFrame())
    await asyncio.sleep(0.005)
    if not seen:  # "did it work?": the model never saw a booking, so it books again
        await llm.run_function_calls(call(2))
    await asyncio.sleep(STEP_S * 3)
    await task.cancel()
    await asyncio.gather(running, return_exceptions=True)


# --- LiveKit ---------------------------------------------------------------------------------


def _livekit_llm() -> Any:
    from livekit.agents import llm

    class Scripted(llm.LLM):
        """Books on "book 4A"; on "did it work?" books again unless it saw a result."""

        def chat(self, *, chat_ctx: Any, tools: Any = None, **_: Any) -> Any:
            return Stream(self, chat_ctx=chat_ctx, tools=tools or [],
                          conn_options=llm.DEFAULT_API_CONNECT_OPTIONS if hasattr(
                              llm, "DEFAULT_API_CONNECT_OPTIONS") else _options())  # fmt: skip

    class Stream(llm.LLMStream):
        async def _run(self) -> None:
            items = self._chat_ctx.items
            last = items[-1]
            saw_result = any(i.type == "function_call_output" for i in items)
            asked = [i.text_content for i in items if getattr(i, "role", None) == "user"]
            if last.type == "function_call_output" or (saw_result and asked[-1] != "book 4A"):
                delta = llm.ChoiceDelta(role="assistant", content="Booked.")
            else:
                n = sum(1 for i in items if i.type == "function_call") + 1
                call = llm.FunctionToolCall(name="book", arguments='{"seat": "4A"}',
                                            call_id=f"c{n}")  # fmt: skip
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
            self._event_ch.send_nowait(llm.ChatChunk(id="x", delta=delta))

    return Scripted()


def _options() -> Any:
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    return DEFAULT_API_CONNECT_OPTIONS


async def livekit_run(service: Watched, cut: float, ours: bool) -> None:
    from livekit.agents import Agent, AgentSession, function_tool

    from agent_adapter.integrations.livekit import livekit_tools

    if ours:
        registry = _registry(service)
        tools = livekit_tools(registry, executor=Executor(registry))
    else:

        @function_tool
        async def book(seat: str) -> str:
            """Book a seat."""
            return await service.book(seat)

        tools = [book]
    session = AgentSession(llm=_livekit_llm())
    await session.start(Agent(instructions="Book seats.", tools=tools))
    first = session.run(user_input="book 4A")
    await _cut_mid_booking(service, cut)
    await session.interrupt(force=True)
    await asyncio.gather(first, return_exceptions=True)
    await asyncio.gather(session.run(user_input="did it work?"), return_exceptions=True)
    await asyncio.sleep(STEP_S * 3)
    await session.aclose()


FRAMEWORKS: dict[str, Runner] = {"pipecat": pipecat_run, "livekit": livekit_run}


async def bench(runs: int, seed: int, only: list[str]) -> dict[str, Any]:
    rng = random.Random(seed)
    cuts = [rng.uniform(0, STEP_S * 0.9) for _ in range(runs)]
    report: dict[str, Any] = {"runs": runs, "seed": seed, "results": {}}
    for framework in only:
        runner = FRAMEWORKS[framework]
        for label, ours in (
            (f"{framework} default", False),
            (f"{framework} + agent-adapter", True),
        ):
            totals: Counter[str] = Counter()
            for cut in cuts:
                service = Watched()
                await runner(service, cut, ours)
                totals.update(service.score())
            report["results"][label] = {
                "double_rate": totals["double"] / runs,
                "half_done_rate": totals["half_done"] / runs,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="*", default=list(FRAMEWORKS), choices=list(FRAMEWORKS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore", DeprecationWarning)
    from loguru import logger  # Pipecat logs through loguru

    logger.remove()
    report = asyncio.run(bench(args.runs, args.seed, args.only))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"interrupt-bench on the real frameworks: {args.runs} runs, seed {args.seed}")
    print("| framework | double bookings | half-done |")
    print("|---|---|---|")
    for key, row in report["results"].items():
        print(f"| {key} | {row['double_rate']:.1%} | {row['half_done_rate']:.1%} |")


if __name__ == "__main__":
    main()
