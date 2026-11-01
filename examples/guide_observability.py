"""Observability guide: events, the ledger, OTel spans (when installed) and log hygiene."""

import asyncio
import logging
import tempfile
from pathlib import Path

from agent_adapter import (
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LoggingHook,
    Message,
    MissingDependency,
    Prices,
    Registry,
    Usage,
)
from agent_adapter.cost import JsonlSink, read_jsonl
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.tracing import EventLog, EventSink, FanoutSink, LoggingSink, OTelSink

registry = Registry()
PRICES = Prices.load(
    {
        "version": "demo",
        "models": {"house": {"provider": "host", "input": 1.0, "output": 4.0}},
        "units": {"local:vec-1": {"provider": "local", "per_unit": 0.02, "unit": "1M tokens"}},
    }
)


@registry.tool
def lookup(topic: str) -> str:
    """Look up a help-center topic."""
    return f"{topic}: help article"


def vectors(texts: list[str]) -> EmbeddingResult:
    """A fake embedding model: two numbers per text."""
    usage = Usage(input_tokens=100_000 * len(texts))  # 100k tokens per text
    return EmbeddingResult(vectors=[[len(t), 1.0] for t in texts], model="vec-1", usage=usage)


async def answer(request: LLMRequest) -> dict[str, object]:
    """A fake model: one reply, priced by its usage."""
    return {
        "text": "Returns are free for 30 days",
        "usage": Usage(input_tokens=1_000, output_tokens=50),
    }


# --8<-- [start:otel]
def otel_or_none() -> OTelSink | None:
    """OTel spans when the `otel` extra is installed; tracing stays optional."""
    try:
        return OTelSink()  # global providers; capture_content=False by default
    except MissingDependency:
        return None


# --8<-- [end:otel]


async def main(folder: Path) -> None:
    # --8<-- [start:wire]
    events = EventLog(max_events=10_000)  # bounded: newest kept
    sinks: list[EventSink] = [
        events,
        LoggingSink(logging.getLogger("app.trace")),
    ]  # DEBUG lines, no content
    otel = otel_or_none()
    sink = FanoutSink(*sinks, *([otel] if otel else []))
    ledger = Ledger(prices=PRICES, sinks=[JsonlSink(folder / "ledger.jsonl")])
    executor = Executor(registry, sink=sink, ledger=ledger, hooks=[LoggingHook()])
    embedder = CallableEmbedder(vectors, model="vec-1", provider="local")
    llm = executor.model(CallableModel(answer, provider="host", model="house"))
    # --8<-- [end:wire]
    # --8<-- [start:run]
    async with executor.run_scope(run_id="run_obs") as ctx:
        await executor.run("lookup", {"topic": "returns"}, ctx)
        await embedder.embed(["returns", "shipping"], ctx=ctx)  # uses the context's ledger and sink
        ask = LLMRequest(messages=(Message(role="user", content="Returns?"),), stage="answer")
        await llm.bind(ctx).generate(ask)
    print([e.type for e in events.events if e.run_id == "run_obs"])
    for entry in read_jsonl(folder / "ledger.jsonl"):  # the same entries, from disk
        print(entry.kind, entry.stage, f"${entry.cost_usd or 0:.6f}", entry.run_id)
    print("otel spans:", otel is not None)
    # --8<-- [end:run]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(main(Path(tmp)))
