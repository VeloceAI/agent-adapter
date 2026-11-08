"""Price LLM and paid-API calls with a ledger; override prices; summarise a run."""

import tempfile
from datetime import date
from pathlib import Path

from agent_adapter import Ledger, LedgerEntry, Prices, Usage
from agent_adapter.cost import JsonlSink, MemorySink, read_jsonl


def prices_demo() -> None:
    # --8<-- [start:prices]
    prices = Prices.default()  # the packaged table, version "2026-11-05"
    usage = Usage(input_tokens=150_000, cached_input_tokens=50_000, output_tokens=2_000)
    print(prices.resolve("gpt-4.1-2025-04-14"))  # alias -> "gpt-4.1"
    cost = prices.cost("gpt-4.1", usage)  # None for an unknown or caps-only model
    assert cost is not None
    print(round(cost, 4))
    # Gemini 2.5 Pro: the tier is chosen by all prompt tokens (here 200k: the low tier)
    tier = prices.price("gemini-2.5-pro", prompt_tokens=usage.prompt_tokens)
    assert tier is not None
    print(tier.input)
    # dated promotional periods
    promo = prices.price("gemini-4-argon", at=date(2026, 12, 1))
    assert promo is not None
    print(promo.input)
    # caps-only entry: capability flags, no price
    print(prices.caps("claude-sonnet-4-5").cache_control, prices.price("claude-sonnet-4-5"))
    # --8<-- [end:prices]


def overrides_demo() -> Prices:
    # --8<-- [start:override]
    house = Prices.default().merged(  # your own prices; the numbers are illustrative
        {
            "version": "acme-2026-11",
            "models": {
                "claude-sonnet-4-5": {  # a priced row per model; "claude-" stays caps-only
                    "provider": "anthropic",
                    "input": 3.00,
                    "output": 15.00,
                    "cache_write_by_ttl": {"5m": 3.75, "1h": 6.00},
                    "cached_input": 0.30,
                    "caps": {"cache_control": True},
                },
            },
            "units": {"voyage:rerank-2.5": {"provider": "voyage", "per_unit": 0.05}},
        }
    )
    print(house.version)  # "2026-11-05+acme-2026-11"
    # --8<-- [end:override]
    return house


def ledger_demo(prices: Prices, folder: Path) -> None:
    # --8<-- [start:ledger]
    memory = MemorySink()
    ledger = Ledger(prices=prices, sinks=[memory, JsonlSink(folder / "ledger.jsonl")])
    usage = Usage(
        input_tokens=1_000,
        cache_write_input_tokens=4_000,
        cache_write_by_ttl={"5m": 4_000},
        output_tokens=300,
    )
    ledger.record(
        LedgerEntry.from_usage(
            usage,
            run_id="run_1",
            call_id="call_1",
            kind="llm",
            stage="plan",
            provider="anthropic",
            model="claude-sonnet-4-5",
            latency_ms=900.0,
            started_at=1_790_000_000.0,
            ended_at=1_790_000_000.9,
        )
    )
    rerank = LedgerEntry(
        run_id="run_1",
        call_id="call_2",
        kind="api",
        stage="rerank",
        provider="voyage",
        model="rerank-2.5",
        units=0.2,  # 0.2M tokens reranked, priced per unit
        latency_ms=120.0,
    )
    ledger.record(rerank)
    ledger.record(rerank)  # same (run_id, entry_id): replaced, never double-counted
    print(round(ledger.total_usd("run_1"), 6))
    summary = ledger.summary("run_1")
    print(sorted(summary.by_stage), summary.unknown_cost_entries)
    # --8<-- [end:ledger]
    print(len(memory.entries), len(read_jsonl(folder / "ledger.jsonl")))


def main() -> None:
    prices_demo()
    house = overrides_demo()
    with tempfile.TemporaryDirectory() as folder:
        ledger_demo(house, Path(folder))


if __name__ == "__main__":
    main()
