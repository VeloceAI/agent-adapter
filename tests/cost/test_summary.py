import pytest
from pydantic import ValidationError

from agent_adapter.core.types import LedgerEntry
from agent_adapter.cost import LedgerSummary, StageSummary
from agent_adapter.cost.summary import summarise


def e(call_id, kind="llm", **fields):
    return LedgerEntry(run_id=fields.pop("run_id", "r1"), call_id=call_id, kind=kind, **fields)


ENTRIES = [
    e(
        "l1",
        stage="plan",
        input_tokens=600,
        cached_input_tokens=300,
        cache_write_input_tokens=100,
        output_tokens=50,
        cost_usd=0.5,
        latency_ms=1000,
        started_at=100.0,
        ended_at=101.0,
    ),
    e(
        "l2",
        stage="plan",
        input_tokens=100,
        output_tokens=10,
        cost_usd=None,
        cache_hit=True,
        latency_ms=0,
        started_at=101.0,
        ended_at=101.0,
    ),
    e(
        "l3",
        stage="answer",
        input_tokens=500,
        cached_input_tokens=400,
        cost_usd=0.25,
        ok=False,
        error_kind="model-timeout",
        latency_ms=500,
        started_at=100.5,
        ended_at=101.0,
    ),
    e(
        "t1",
        kind="tool",
        stage="search",
        cost_usd=0.0,
        latency_ms=800,
        started_at=100.2,
        ended_at=101.0,
    ),
    e("t2", kind="tool", stage="search", cost_usd=0.0, cache_hit=True, latency_ms=0),
    e(
        "a1",
        kind="api",
        stage="rerank",
        units=2.0,
        cost_usd=0.1,
        latency_ms=300,
        started_at=100.0,
        ended_at=102.0,
    ),
]


def test_summary_totals_and_unknown_costs():
    summary = summarise(ENTRIES, "r1")
    assert isinstance(summary, LedgerSummary)
    assert summary.run_id == "r1"
    assert summary.total_usd == pytest.approx(0.85)
    assert summary.unknown_cost_entries == 1


def test_summary_by_stage():
    plan = summarise(ENTRIES).by_stage["plan"]
    assert plan == StageSummary(
        calls=2,
        cost_usd=0.5,
        input_tokens=700,
        cached_input_tokens=300,
        cache_write_input_tokens=100,
        output_tokens=60,
        latency_ms=1000.0,
        errors=0,
    )
    assert summarise(ENTRIES).by_stage["answer"].errors == 1
    assert set(summarise(ENTRIES).by_stage) == {"plan", "answer", "search", "rerank"}


def test_summary_by_kind():
    by_kind = summarise(ENTRIES).by_kind
    assert by_kind["llm"].calls == 3
    assert by_kind["tool"].calls == 2
    assert by_kind["api"].cost_usd == pytest.approx(0.1)


def test_summary_hit_rates():
    summary = summarise(ENTRIES)
    # cached 300 + 400 over prompt tokens (600+300+100) + 100 + (500+400)
    assert summary.prompt_cache_hit_rate == pytest.approx(700 / 2000)
    assert summary.response_cache_hit_rate == pytest.approx(1 / 3)
    assert summary.tool_cache_hit_rate == pytest.approx(1 / 2)


def test_summary_wall_busy_parallelism():
    summary = summarise(ENTRIES)
    assert summary.wall_ms == pytest.approx(2000.0)  # 100.0 -> 102.0, entries without times ignored
    assert summary.busy_ms == pytest.approx(1000 + 0 + 500 + 800 + 0)  # tool + llm only
    assert summary.parallelism == pytest.approx(2300 / 2000)


def test_summary_of_nothing_is_zero():
    summary = summarise([])
    assert summary.run_id is None
    assert summary.total_usd == 0.0
    assert summary.by_stage == {} and summary.by_kind == {}
    assert summary.prompt_cache_hit_rate == 0.0
    assert summary.response_cache_hit_rate == 0.0
    assert summary.tool_cache_hit_rate == 0.0
    assert summary.wall_ms == 0.0 and summary.parallelism == 0.0


def test_summary_is_frozen_and_serialisable():
    summary = summarise(ENTRIES, "r1")
    assert LedgerSummary.model_validate_json(summary.model_dump_json()) == summary
    with pytest.raises(ValidationError):
        summary.total_usd = 1.0  # type: ignore[misc]
