"""Embeddings: wrap your own embedding function, then batch, cache and bill it."""

import asyncio

from agent_adapter import Ledger, MemoryCache, Prices, Usage
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.errors import ModelOutputInvalid


# --8<-- [start:fn]
def letters(texts: list[str]) -> EmbeddingResult:
    """A local fake model: 3 numbers per text, one token per word."""
    vectors = [[len(t), t.count(" "), sum(c in "aeiou" for c in t)] for t in texts]
    tokens = sum(len(t.split()) for t in texts)
    return EmbeddingResult(vectors=vectors, model="letters-v1", usage=Usage(input_tokens=tokens))


# Real models plug in the same way; returning plain vectors (or a numpy array) is enough.
#
#   from sentence_transformers import SentenceTransformer
#   st = SentenceTransformer("all-MiniLM-L6-v2")
#   embedder = CallableEmbedder(st.encode, model="all-MiniLM-L6-v2", provider="local")
#
#   import ollama
#   client = ollama.AsyncClient()  # http://localhost:11434
#
#   async def nomic(texts: list[str]) -> EmbeddingResult:
#       reply = await client.embed(model="nomic-embed-text", input=texts)
#       usage = Usage(input_tokens=reply.prompt_eval_count or 0)
#       return EmbeddingResult(vectors=reply.embeddings, usage=usage)
#
#   embedder = CallableEmbedder(nomic, model="nomic-embed-text", provider="ollama")
# --8<-- [end:fn]


async def main() -> None:
    # --8<-- [start:wire]
    unit_price = {"provider": "local", "per_unit": 0.02, "unit": "1M tokens"}
    ledger = Ledger(prices=Prices.default().merged({"units": {"local:letters-v1": unit_price}}))
    embedder = CallableEmbedder(
        letters,  # sync: runs in a worker thread
        model="letters-v1",
        provider="local",
        batch_size=2,
        cache=MemoryCache(),
        ledger=ledger,
    )
    first = await embedder.embed(["fix my bill", "my lost box", "fix my bill"])
    print(len(first.vectors), first.vectors[0], first.usage.input_tokens, first.cache_hits)
    again = await embedder.embed(["my lost box", "return label"])
    print(again.cache_hits, again.usage.input_tokens)
    for entry in ledger.entries():
        cost = f"{entry.cost_usd:.8f}"
        print(entry.kind, entry.stage, entry.model, entry.input_tokens, cost, entry.cache_hit)
    # --8<-- [end:wire]

    # --8<-- [start:errors]
    broken = CallableEmbedder(lambda texts: [[1.0]], model="broken")
    try:
        await broken.embed(["a", "b"])
    except ModelOutputInvalid as exc:
        print(exc.kind, exc)
    # --8<-- [end:errors]


if __name__ == "__main__":
    asyncio.run(main())
