"""Sink plumbing for runs: merge sinks without duplicates, and hand flow graphs to takers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import EventSink
from agent_adapter.tracing.sinks import FanoutSink, SinkHub, members

logger = logging.getLogger(__name__)

__all__ = ["GraphTaker", "SinkHub", "members", "merge_sinks", "offer_flow_graph"]


@runtime_checkable
class GraphTaker(Protocol):
    """Internal to the run viewer: a sink that also keeps flow graphs (not a public API)."""

    def take_flow_graph(self, run_id: str, flow: str, graph: dict[str, Any]) -> None: ...


def merge_sinks(primary: EventSink | None, extra: EventSink | None) -> EventSink | None:
    """`primary` plus every sink under `extra` not already under it, so each gets events once."""
    if extra is None:
        return primary
    if primary is None:
        return extra
    have = {id(sink) for sink in members(primary)}
    if id(extra) in have:
        return primary
    leaves = [
        sink
        for sink in members(extra)
        if not isinstance(sink, SinkHub | FanoutSink) and id(sink) not in have
    ]
    return FanoutSink(primary, *leaves) if leaves else primary


def offer_flow_graph(
    sink: EventSink | None, run_id: str, flow: str, build: Callable[[], dict[str, Any]]
) -> None:
    """Hand a flow graph to each graph-taking sink under `sink`; `build` runs only if one exists."""
    takers = [s for s in members(sink) if isinstance(s, GraphTaker)]
    if not takers:
        return
    graph = build()
    for taker in takers:
        try:
            taker.take_flow_graph(run_id, flow, graph)
        except Exception as exc:
            # Like emit: a sink never breaks the traced run.
            log_failure(logger, logging.WARNING, exc, "graph sink failed for flow %s", flow)
