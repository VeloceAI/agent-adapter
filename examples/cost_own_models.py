"""Price your own models: pattern keys, free local models, per-call and per-second prices."""

from agent_adapter import Ledger, LedgerEntry, Prices, Usage


def own_table_demo() -> Prices:
    # --8<-- [start:own]
    prices = Prices.load(  # only your models; the numbers are illustrative
        {
            "version": "acme-2026-11",
            "models": {
                "llama*": {"provider": "ollama", "free": True},  # local: always 0, no warning
                "acme-gpu-*": {"provider": "runpod", "per_second": 0.0011},  # a rented GPU
                "acme-api": {"provider": "acme", "input": 0.2, "output": 0.6, "per_call": 0.001},
            },
        }
    )
    usage = Usage(input_tokens=2_000, output_tokens=500)
    print(prices.resolve("llama3.1:8b"), prices.cost("llama3.1:8b", usage))
    print(f"{prices.cost('acme-gpu-a100', usage, seconds=4.0):.4f}")  # 4 s x 0.0011
    print(f"{prices.cost('acme-api', usage):.4f}")  # tokens + one call
    # --8<-- [end:own]
    return prices


def starting_points_demo() -> None:
    # --8<-- [start:start]
    # Keep the packaged prices, or start from an empty table.
    free = {"provider": "local", "free": True}
    # Every unrowed model is free. Careful: an unknown paid model then costs 0, not unknown.
    house = Prices.default().merged({"version": "house", "models": {"*": free}})
    print(house.resolve("gpt-4.1"), house.resolve("my-finetune"))  # a real row beats "*"
    mine = Prices.empty().merged({"version": "mine", "models": {"my-finetune": free}})
    print(mine.version, mine.resolve("gpt-4.1"))  # nothing packaged: gpt-4.1 is unknown
    # --8<-- [end:start]


def ledger_demo(prices: Prices) -> None:
    ledger = Ledger(prices=prices)  # pass it as Executor(ledger=...) to price every model call
    ledger.record(
        LedgerEntry(
            run_id="run_1",
            call_id="call_1",
            kind="llm",
            provider="runpod",
            model="acme-gpu-a100",
            output_tokens=800,
            latency_ms=2_500.0,  # per-second prices use the entry's latency
        )
    )
    print(round(ledger.total_usd("run_1"), 6))  # 2.5 s x 0.0011


def main() -> None:
    prices = own_table_demo()
    starting_points_demo()
    ledger_demo(prices)


if __name__ == "__main__":
    main()
