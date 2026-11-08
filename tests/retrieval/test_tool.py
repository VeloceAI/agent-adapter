import dataclasses

import pytest

from agent_adapter import Executor, MemoryCache, Registry
from agent_adapter.core.context import ToolContext
from agent_adapter.errors import ConfigError, ToolArgsInvalid, ToolUsageError
from agent_adapter.retrieval import (
    RETRIEVAL_KIND,
    CallableRetriever,
    Document,
    EmbeddingRetriever,
    retriever_tool,
)
from agent_adapter.retrieval.documents import FilterValue
from agent_adapter.retrieval.tool import DEFAULT_DESCRIPTION
from agent_adapter.tracing import EventLog
from tests.retrieval.support import CORPUS, embedder


def setup(**options: object) -> tuple[Executor, EventLog, list[str]]:
    calls: list[str] = []

    def fn(query: str, k: int, filters: dict[str, FilterValue]) -> list[Document]:
        calls.append(query)
        return [d for d in CORPUS if all(d.metadata.get(a) == b for a, b in filters.items())][:k]

    registry, sink = Registry(), EventLog()
    registry.add(retriever_tool(CallableRetriever(fn, name="help"), **options))  # type: ignore[arg-type]
    return Executor(registry, sink=sink, cache=MemoryCache()), sink, calls


def test_spec_shows_only_the_query_to_the_model() -> None:
    spec = retriever_tool(CallableRetriever(lambda q, k, f: []))
    assert (spec.name, spec.kind, spec.description) == (
        "search_docs",
        RETRIEVAL_KIND,
        DEFAULT_DESCRIPTION,
    )
    schema = spec.json_schema()
    assert list(schema["properties"]) == ["query"]
    assert schema["required"] == ["query"]


async def test_tool_returns_documents_with_sources_through_the_executor() -> None:
    executor, sink, _ = setup(k=2, filters={"scope": "north"})
    async with executor.run_scope() as ctx:
        value = await executor.run("search_docs", {"query": "refund"}, ctx)
    assert value == {
        "documents": [
            {
                "id": "refunds",
                "text": CORPUS[0].text,
                "score": None,
                "source": {"uri": "https://help.example.com/refunds", "title": "Refunds"},
            },
            {
                "id": "hours",
                "text": CORPUS[1].text,
                "score": None,
                "source": {"uri": "https://help.example.com/hours", "title": "Hours"},
            },
        ]
    }
    types = [e.type for e in sink.events]
    assert types == [
        "run.started",
        "tool.started",
        "retrieval.searched",
        "tool.finished",
        "run.finished",
    ]
    tool_started, searched = sink.events[1], sink.events[2]
    assert searched.call_id == tool_started.call_id
    assert searched.data["ids"] == ["refunds", "hours"]


async def test_per_call_filters_come_from_the_context_not_the_model() -> None:
    executor, _, _ = setup(k=5, filters=lambda ctx: {"scope": ctx.user_id})
    ctx = dataclasses.replace(executor.context(), user_id="south")
    value = await executor.run("search_docs", {"query": "parking"}, ctx)
    assert [d["id"] for d in value["documents"]] == ["parking", "shop"]
    with pytest.raises(ToolArgsInvalid):  # the model cannot add arguments
        await executor.run("search_docs", {"query": "q", "filters": {}}, ctx)


async def test_bad_per_call_filters_fail_the_call() -> None:
    executor, _, calls = setup(filters=lambda ctx: {"scope": None})
    with pytest.raises(ToolUsageError):
        await executor.run("search_docs", {"query": "q"}, executor.context())
    assert calls == []


async def test_cache_ttl_serves_repeats_from_the_executor_cache() -> None:
    executor, sink, calls = setup(cache_ttl=60)
    ctx = executor.context()
    first = await executor.run("search_docs", {"query": "ticket"}, ctx)
    again = await executor.run("search_docs", {"query": "ticket"}, ctx)
    assert first == again
    assert calls == ["ticket"]
    assert len(sink.of_type("tool.cached")) == 1


async def test_metadata_and_max_chars_options() -> None:
    executor, _, _ = setup(k=1, include_metadata=True, max_chars=5)
    value = await executor.run("search_docs", {"query": "q"}, executor.context())
    [shown] = value["documents"]
    assert shown["text"] == CORPUS[0].text[:5]
    assert shown["metadata"] == {"scope": "north"}


@pytest.mark.parametrize(
    "options",
    [
        {"k": 0},
        {"filters": {"scope": None}},
        {"max_chars": 0},
        {"max_chars": True},
        {"filters": lambda ctx: {}, "cache_scope": "global"},
    ],
)
def test_bad_options_are_config_errors(options: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        retriever_tool(CallableRetriever(lambda q, k, f: []), **options)  # type: ignore[arg-type]


def test_a_non_retriever_is_refused() -> None:
    with pytest.raises(ToolUsageError):
        retriever_tool(object())  # type: ignore[arg-type]


class ForeignPort:
    """A RetrieverPort that is not a GuardedRetriever and ignores filters."""

    def __init__(self, result: object = None) -> None:
        self.result = result

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: object = None,
        ctx: ToolContext | None = None,
    ) -> object:
        return list(CORPUS) if self.result is None else self.result


async def test_a_foreign_port_gets_the_same_checks_and_events() -> None:
    registry, sink = Registry(), EventLog()
    registry.add(retriever_tool(ForeignPort(), k=1, filters={"scope": "south"}))  # type: ignore[arg-type]
    executor = Executor(registry, sink=sink)
    value = await executor.run("search_docs", {"query": "q"}, executor.context())
    assert [d["id"] for d in value["documents"]] == ["parking"]
    [searched] = sink.of_type("retrieval.")
    assert (searched.data["retriever"], searched.data["backend"]) == ("ForeignPort", "port")


async def test_a_foreign_port_returning_a_non_list_fails() -> None:
    registry = Registry()
    registry.add(retriever_tool(ForeignPort(result="text")))  # type: ignore[arg-type]
    executor = Executor(registry)
    with pytest.raises(ToolUsageError):
        await executor.run("search_docs", {"query": "q"}, executor.context())


async def test_embedding_retriever_tool_keeps_scopes_apart() -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS, scope_key="scope")
    registry = Registry()
    registry.add(retriever_tool(retriever, k=10, filters=lambda ctx: {"scope": ctx.user_id}))
    executor = Executor(registry)
    north = dataclasses.replace(executor.context(), user_id="north")
    value = await executor.run("search_docs", {"query": "parking ticket"}, north)
    assert {d["id"] for d in value["documents"]} == {"refunds", "hours"}
    nobody = executor.context()  # no user: the scope filter would be "", so the call fails
    with pytest.raises(ToolUsageError, match="blank"):
        await executor.run("search_docs", {"query": "parking"}, nobody)


@pytest.mark.parametrize("user", ["", "   "])
async def test_blank_per_call_filter_values_fail_closed_before_the_backend(user: str) -> None:
    executor, _, calls = setup(filters=lambda ctx: {"scope": ctx.user_id, "kind": "faq"})
    ctx = dataclasses.replace(executor.context(), user_id=user)
    with pytest.raises(ToolUsageError, match=r"\['scope'\]"):
        await executor.run("search_docs", {"query": "q"}, ctx)
    assert calls == []


async def test_fixed_filters_may_still_hold_an_empty_string() -> None:
    executor, _, calls = setup(filters={"scope": ""})
    value = await executor.run("search_docs", {"query": "q"}, executor.context())
    assert value == {"documents": []}
    assert calls == ["q"]
