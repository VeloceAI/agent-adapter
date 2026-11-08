"""Retrieval (RAG): a model searches a museum help desk's documents and answers with citations."""

import asyncio
import dataclasses
import json

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    MemoryCache,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.embeddings import CallableEmbedder
from agent_adapter.retrieval import (
    CallableRetriever,
    Document,
    EmbeddingRetriever,
    Source,
    retriever_tool,
)
from agent_adapter.tracing import EventLog

WORDS = ("ticket", "refund", "opening", "hours", "parking", "audio", "guide")


def embed(texts: list[str]) -> list[list[float]]:
    """Stands in for a real embedding model: one count per known word."""
    return [[float(text.lower().count(word)) for word in WORDS] for text in texts]


# --8<-- [start:corpus]
def page(id: str, text: str, museum: str) -> Document:
    """A help page of one museum; `museum` is the scope a search is limited to."""
    source = Source(uri=f"https://help.example.com/{museum}/{id}", title=id.replace("-", " "))
    return Document(id=f"{museum}-{id}", text=text, source=source, metadata={"museum": museum})


PAGES = [
    page("refunds", "A ticket refund is possible up to 24 hours before the visit.", "city"),
    page("opening-hours", "Opening hours are 9 to 17, closed on Mondays.", "city"),
    page("audio-guide", "The audio guide is free with every ticket.", "city"),
    page("refunds", "Tickets bought online get no refund.", "harbour"),
]
retriever = EmbeddingRetriever(CallableEmbedder(embed), PAGES, scope_key="museum")
# --8<-- [end:corpus]

# --8<-- [start:tool]
registry = Registry()
registry.add(
    retriever_tool(
        retriever,
        k=2,
        filters=lambda ctx: {"museum": ctx.user_id},  # the host's scope, never the model's
        cache_ttl=300,
    )
)
# --8<-- [end:tool]


async def scripted_llm(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: searches once, then answers from the passages it got."""
    last = request.messages[-1]
    if last.role == "tool":
        documents = json.loads(last.content)["documents"]
        best = documents[0]
        text = f"{best['text']} [source: {best['source']['uri']}]"
        reply = Message(role="assistant", content=text)
        return LLMResponse(provider="host", model="help-desk-v1", text=text, message=reply)
    call = ToolCall(name="search_docs", arguments={"query": last.content}, id="call_search")
    return LLMResponse(
        provider="host",
        model="help-desk-v1",
        tool_calls=(call,),
        finish_reason="tool_calls",
        message=Message(role="assistant", tool_calls=(call,)),
    )


async def main() -> None:
    # --8<-- [start:loop]
    sink = EventLog()
    executor = Executor(registry, sink=sink, cache=MemoryCache())
    model = CallableModel(scripted_llm, provider="host", model="help-desk-v1")
    question = (Message(role="user", content="Can I get a ticket refund?"),)
    for museum in ("city", "harbour"):
        ctx = dataclasses.replace(executor.context(), user_id=museum)
        result = await run_tool_loop(model, registry, question, executor=executor, ctx=ctx)
        print(museum, "->", result.final.text)
    [first, second] = sink.of_type("retrieval.searched")
    print(first.data["ids"], second.data["ids"])
    # --8<-- [end:loop]

    # --8<-- [start:callable]
    def search_service(query: str, k: int, filters: dict[str, object]) -> list[dict[str, object]]:
        """Stands in for a vector database or search API client."""
        hits = [{"id": "faq-7", "text": "Parking is free after 17:00.", "metadata": filters}]
        return hits[:k]

    faq = CallableRetriever(search_service, name="faq")
    [hit] = await faq.retrieve("parking", k=3, filters={"museum": "city"})
    print(hit.id, hit.text)
    # --8<-- [end:callable]


if __name__ == "__main__":
    asyncio.run(main())
