"""Ledger summaries: cost and tokens per stage and kind, cache hit rates, parallelism."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from agent_adapter.core.types import LedgerEntry

_FROZEN = ConfigDict(frozen=True)


class StageSummary(BaseModel):
    """Totals over one group of entries (a stage or a kind); unknown costs count as 0."""

    model_config = _FROZEN

    calls: int
    cost_usd: float
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int
    latency_ms: float
    errors: int


class LedgerSummary(BaseModel):
    """A run's (or every run's) cost, cache effectiveness and achieved parallelism."""

    model_config = _FROZEN

    run_id: str | None
    total_usd: float
    unknown_cost_entries: int
    by_stage: dict[str, StageSummary]
    by_kind: dict[str, StageSummary]
    prompt_cache_hit_rate: float  # cached input tokens / all prompt tokens, llm entries
    response_cache_hit_rate: float  # llm entries served from the response cache
    tool_cache_hit_rate: float  # tool entries served from the tool-result cache
    wall_ms: float  # first start to last end, over entries with both times
    busy_ms: float  # sum of tool + llm latencies
    parallelism: float  # busy_ms / wall_ms


def _group(entries: list[LedgerEntry]) -> StageSummary:
    return StageSummary(
        calls=len(entries),
        cost_usd=sum(e.cost_usd or 0.0 for e in entries),
        input_tokens=sum(e.input_tokens for e in entries),
        cached_input_tokens=sum(e.cached_input_tokens for e in entries),
        cache_write_input_tokens=sum(e.cache_write_input_tokens for e in entries),
        output_tokens=sum(e.output_tokens for e in entries),
        latency_ms=sum(e.latency_ms for e in entries),
        errors=sum(not e.ok for e in entries),
    )


def _by(entries: list[LedgerEntry], attr: str) -> dict[str, StageSummary]:
    groups: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        groups.setdefault(getattr(entry, attr), []).append(entry)
    return {key: _group(group) for key, group in groups.items()}


def _rate(hits: float, total: float) -> float:
    return hits / total if total else 0.0


def summarise(entries: Iterable[LedgerEntry], run_id: str | None = None) -> LedgerSummary:
    """Summarise `entries` (already filtered to `run_id` by the caller)."""
    items = list(entries)
    llm = [e for e in items if e.kind == "llm"]
    tools = [e for e in items if e.kind == "tool"]
    prompt = sum(e.input_tokens + e.cached_input_tokens + e.cache_write_input_tokens for e in llm)
    timed = [e for e in items if e.started_at > 0 and e.ended_at > 0]
    wall_s = max(e.ended_at for e in timed) - min(e.started_at for e in timed) if timed else 0.0
    wall_ms = max(wall_s, 0.0) * 1000.0
    busy_ms = sum(e.latency_ms for e in llm + tools)
    return LedgerSummary(
        run_id=run_id,
        total_usd=sum(e.cost_usd or 0.0 for e in items),
        unknown_cost_entries=sum(e.cost_usd is None for e in items),
        by_stage=_by(items, "stage"),
        by_kind=_by(items, "kind"),
        prompt_cache_hit_rate=_rate(sum(e.cached_input_tokens for e in llm), prompt),
        response_cache_hit_rate=_rate(sum(e.cache_hit for e in llm), len(llm)),
        tool_cache_hit_rate=_rate(sum(e.cache_hit for e in tools), len(tools)),
        wall_ms=wall_ms,
        busy_ms=busy_ms,
        parallelism=_rate(busy_ms, wall_ms),
    )
