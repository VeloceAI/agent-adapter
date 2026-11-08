"""Per-call overhead of the executor over a direct call: a Markdown table (median, p95, us).

Run from the repo root: `python scripts/bench_overhead.py` (`--iterations 20000` for steadier
numbers). Every scenario calls a tool or a model that does nothing, so the figure is the
adapter's own work; a real tool or model call usually dwarfs it. Logging is on at INFO,
formatted and written to the null device, as a service would log.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import platform
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from agent_adapter import (
    AutoApprove,
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    LoggingHook,
    MemoryCache,
    Message,
    Prices,
    Registry,
    __version__,
)
from agent_adapter.hooks import PIIGuard
from agent_adapter.policy import load_policy

Step = Callable[[], Awaitable[Any]]
REPLY = LLMResponse(
    provider="local", model="noop", text="ok", message=Message(role="assistant", content="ok")
)
REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))
POLICY = {
    "default": "deny",
    "rules": [{"tools": ["echo"], "allow": True}],
    "guards": [{"guard": "pii", "tools": ["echo"], "on": ["args"]}],
}
ROUND = 100  # calls per scenario before moving to the next
PRICES = Prices.load(
    {"version": "bench", "models": {"noop": {"provider": "local", "input": 0.0, "output": 0.0}}}
)


@dataclass(frozen=True)
class Row:
    """One scenario's timings, in microseconds (us) per call."""

    scenario: str
    median_us: float
    p95_us: float


async def echo(text: str) -> str:
    """Return the text (the tool under test)."""
    return text


def echo_sync(text: str) -> str:
    """Return the text, as a sync tool (the executor runs it in a worker thread)."""
    return text


async def noop_model(request: LLMRequest) -> LLMResponse:
    """A model port that answers at once."""
    return REPLY


def _registry() -> Registry:
    registry = Registry()
    registry.tool(echo)
    registry.tool(echo_sync)
    registry.tool(name="echo_confirm", confirm=True)(echo)
    registry.tool(name="echo_cached", cache_ttl=3600)(echo)
    return registry


def scenarios() -> list[tuple[str, Step]]:
    """Each scenario's name and one awaitable call of it."""
    registry = _registry()
    hooks = [LoggingHook(), PIIGuard(on=("args", "result"))]
    bare = Executor(registry)
    hooked = Executor(registry, hooks=hooks)
    approved = Executor(registry, approval=AutoApprove())
    cached = Executor(registry, cache=MemoryCache())
    ledgered = Executor(registry, ledger=Ledger(prices=PRICES))
    everything = Executor(
        registry,
        hooks=hooks,
        approval=AutoApprove(),
        cache=MemoryCache(),
        ledger=Ledger(prices=PRICES),
    )
    args = {"text": "hello"}
    model = CallableModel(noop_model, provider="local", model="noop")
    bare_model = bare.model(model)
    full_model = Executor(hooks=hooks, ledger=Ledger(prices=PRICES)).model(model)
    cached_model = cached.model(model, cache_ttl=3600)
    governed = load_policy(POLICY).apply(Executor(registry))

    return [
        ("tool: direct `await echo()` (baseline)", lambda: echo(**args)),
        ("tool: `executor.run`, nothing on", lambda: bare.run("echo", args)),
        ("tool: + hooks (LoggingHook, PIIGuard)", lambda: hooked.run("echo", args)),
        (
            "tool: + approval (AutoApprove, confirm tool)",
            lambda: approved.run("echo_confirm", args),
        ),
        ("tool: + cache (every call a hit)", lambda: cached.run("echo_cached", args)),
        ("tool: + policy (allow rule, PII guard)", lambda: governed.run("echo", args)),
        ("tool: + ledger", lambda: ledgered.run("echo", args)),
        ("tool: hooks, approval, cache, ledger", lambda: everything.run("echo_confirm", args)),
        ("tool: a sync tool, nothing on (worker thread)", lambda: bare.run("echo_sync", args)),
        ("model: direct `generate` (baseline)", lambda: model.generate(REQUEST)),
        ("model: `executor.model(...)`, nothing on", lambda: bare_model.generate(REQUEST)),
        ("model: + hooks and ledger", lambda: full_model.generate(REQUEST)),
        ("model: + response cache (every call a hit)", lambda: cached_model.generate(REQUEST)),
    ]


async def sample(step: Step, calls: int) -> list[float]:
    """Microseconds of each of `calls` calls."""
    samples = []
    for _ in range(calls):
        start = time.perf_counter_ns()
        await step()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return samples


def summarise(name: str, samples: list[float]) -> Row:
    """Median and p95 of the samples."""
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return Row(name, statistics.median(ordered), p95)


async def measure(iterations: int, warmup: int) -> list[Row]:
    """Time every scenario on one event loop, in interleaved rounds so load spikes hit all."""
    steps = scenarios()
    for _, step in steps:
        await sample(step, warmup)
    samples: dict[str, list[float]] = {name: [] for name, _ in steps}
    done = 0
    while done < iterations:
        batch = min(ROUND, iterations - done)
        for name, step in steps:
            samples[name] += await sample(step, batch)
        done += batch
    return [summarise(name, samples[name]) for name, _ in steps]


def render(rows: Sequence[Row], iterations: int) -> str:
    """The Markdown table, with the machine it ran on."""
    machine = (
        f"{platform.system()} {platform.release()} {platform.machine()}, {os.cpu_count()} CPUs"
    )
    lines = [
        f"agent_adapter {__version__}, Python {platform.python_version()}, {machine}; "
        f"{iterations} calls each",
        "",
        "| Scenario | Median (us) | p95 (us) |",
        "|---|---:|---:|",
    ]
    lines += [f"| {r.scenario} | {r.median_us:.1f} | {r.p95_us:.1f} |" for r in rows]
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def production_logging() -> Iterator[None]:
    """INFO records formatted and written (to the null device), as a service logs them."""
    logger = logging.getLogger("agent_adapter")
    level = logger.level
    with open(os.devnull, "w", encoding="utf-8") as sink:
        handler = logging.StreamHandler(sink)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Measure and print the table; 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=_positive, default=5000, help="calls per scenario")
    parser.add_argument("--warmup", type=int, default=200, help="untimed calls first")
    args = parser.parse_args(argv)
    with production_logging():
        rows = asyncio.run(measure(args.iterations, max(0, args.warmup)))
    (out or sys.stdout).write(render(rows, args.iterations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
