"""Warm-up at call start: connections, rendered lines and read-only prefetches, all at once."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.voice.speculate import can_speculate

if TYPE_CHECKING:
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)


class Warmup:
    """What to get ready when a call starts, so its first turn is as fast as the rest.

    `connections` are objects with an async `warm()` (a provider's connection pool) or plain
    functions, sync or async (open a socket, send a HEAD request). The call's STT, TTS and
    model are warmed too when they have `warm()`. `lines` are rendered to audio now (a
    greeting); acknowledgements and cached answers are always rendered. `tools` are read-only
    calls, `(name, arguments)`, run now through the executor: with `cache_ttl` on the tool,
    the turn that asks for one gets the cached result. Each step has `timeout_s`; a failure
    is logged and never stops the call.
    """

    def __init__(
        self,
        *,
        connections: Iterable[Any] = (),
        lines: Iterable[str] = (),
        tools: Iterable[tuple[str, Mapping[str, Any]]] = (),
        timeout_s: float = 5.0,
    ) -> None:
        self.connections = tuple(connections)
        self.lines = tuple(lines)
        self.tools = tuple((name, dict(args)) for name, args in tools)
        self.timeout_s = timeout_s

    def check(self, registry: Registry) -> None:
        """Raise `ToolUsageError` if a prefetch tool is missing or may change something."""
        for name, _ in self.tools:
            if name not in registry or not can_speculate(registry.get(name)):
                raise ToolUsageError(
                    f"prefetch tool {name!r} must be registered, effects='read_only', no confirm"
                )


@dataclass
class WarmReport:
    """How long each warm-up step took, in ms, and the steps that failed (error type)."""

    ms: dict[str, float] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


async def warm_up(agent: VoiceAgent, providers: Iterable[Any] = ()) -> WarmReport:
    """Run every warm-up step of `agent.speed` concurrently; never raises."""
    speed = agent.speed
    report = WarmReport()
    if speed is None:
        return report
    warmup = speed.warmup or Warmup()
    steps: dict[str, Callable[[], Awaitable[Any]]] = {}
    for n, thing in enumerate((*providers, agent.model, *warmup.connections)):
        step = _connect(thing)
        if step is not None:
            steps[f"{type(thing).__name__}#{n}"] = step
    if speed.acks is not None:  # first: the speaker renders one line at a time
        steps["acks"] = lambda: _render(agent, list(speed.acks.lines) if speed.acks else [])
    answers = speed.answers.lines() if speed.answers is not None else []
    lines = list(dict.fromkeys([*warmup.lines, *answers]))
    if lines:
        steps["lines"] = lambda: _render(agent, lines)
    for name, args in warmup.tools:
        spec = agent.registry.get(name) if name in agent.registry else None
        if spec is not None and can_speculate(spec):
            steps[f"tool:{name}"] = _prefetch(agent, name, args)
    results = await asyncio.gather(*(_timed(fn, warmup.timeout_s) for fn in steps.values()))
    for name, (ms, error) in zip(steps, results, strict=True):
        report.ms[name] = ms
        if error is not None:
            report.failed[name] = type(error).__name__
            log_failure(logger, logging.WARNING, error, "voice warm-up step %s failed", name)
    return report


def _connect(thing: Any) -> Callable[[], Awaitable[Any]] | None:
    warm = getattr(thing, "warm", None)
    target = warm if callable(warm) else thing if inspect.isfunction(thing) else None
    if target is None:
        return None

    async def step() -> None:
        value = target()
        if inspect.isawaitable(value):
            await value

    return step


async def _render(agent: VoiceAgent, lines: list[str]) -> None:
    warm = getattr(agent.speaker, "warm", None)
    if callable(warm):
        await warm(lines)
    acks = agent.speed.acks if agent.speed else None
    if acks is not None:  # a speaker that cannot pre-render speaks its own way
        acks.rendered.update(line for line in acks.lines if line in lines)


def _prefetch(agent: VoiceAgent, name: str, args: dict[str, Any]) -> Callable[[], Awaitable[Any]]:
    return lambda: agent.executor.run(name, args, agent.ctx)


async def _timed(
    fn: Callable[[], Awaitable[Any]], timeout_s: float
) -> tuple[float, BaseException | None]:
    start = time.perf_counter()
    error: BaseException | None = None
    try:
        await asyncio.wait_for(fn(), timeout_s)
    except Exception as exc:
        error = exc
    return round((time.perf_counter() - start) * 1000, 1), error


__all__ = ["WarmReport", "Warmup", "warm_up"]
