import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_adapter.errors import IntegrationError, MissingDependency, ToolUsageError
from agent_adapter.retrieval import LangChainRetriever, LlamaIndexRetriever
from agent_adapter.retrieval.documents import document_id
from agent_adapter.retrieval.llamaindex import metadata_filters

# --- LlamaIndex fakes: TextNode, NodeWithScore, a retriever and an index ----------------------


@dataclass
class FakeNode:
    node_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text


@dataclass
class FakeScored:
    node: Any
    score: float | None


NODES = [
    FakeScored(FakeNode("n1", "Opening hours", {"file_path": "hours.md", "scope": "north"}), 0.9),
    FakeScored(
        FakeNode("n2", "Parking", {"url": "https://x/p", "title": "P", "scope": "south"}), 0.4
    ),
]


class FakeLlamaRetriever:
    def __init__(self, nodes: list[Any]) -> None:
        self.nodes = nodes
        self.queries: list[str] = []

    async def aretrieve(self, query: str) -> list[Any]:
        self.queries.append(query)
        return self.nodes


class SyncLlamaRetriever:
    def retrieve(self, query: str) -> list[Any]:
        return NODES


class FakeIndex:
    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []

    def as_retriever(self, **options: Any) -> FakeLlamaRetriever:
        self.options.append(options)
        return FakeLlamaRetriever(NODES[: options["similarity_top_k"]])


async def test_llamaindex_retriever_maps_nodes_to_documents() -> None:
    inner = FakeLlamaRetriever(NODES)
    found = await LlamaIndexRetriever(inner).retrieve("hours", k=5)
    assert inner.queries == ["hours"]
    assert [(d.id, d.score, d.source.uri, d.source.title) for d in found] == [
        ("n1", 0.9, "hours.md", ""),
        ("n2", 0.4, "https://x/p", "P"),
    ]


async def test_llamaindex_sync_retriever_runs_in_a_thread_and_is_cut_to_k() -> None:
    found = await LlamaIndexRetriever(SyncLlamaRetriever()).retrieve("hours", k=1)
    assert [d.id for d in found] == ["n1"]


async def test_llamaindex_bare_retriever_results_are_rechecked_against_filters() -> None:
    found = await LlamaIndexRetriever(FakeLlamaRetriever(NODES)).retrieve(
        "q", filters={"scope": "south"}
    )
    assert [d.id for d in found] == ["n2"]


async def test_llamaindex_index_gets_k_and_native_filters() -> None:
    index = FakeIndex()
    retriever = LlamaIndexRetriever(index, native_filters=lambda f: ("native", f))
    await retriever.retrieve("q", k=1)
    found = await retriever.retrieve("q", k=2, filters={"scope": "north"})
    assert index.options == [
        {"similarity_top_k": 1},
        {"similarity_top_k": 2, "filters": ("native", {"scope": "north"})},
    ]
    assert [d.id for d in found] == ["n1"]


async def test_llamaindex_default_filters_use_metadata_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("llama_index.core.vector_stores")

    @dataclass
    class MetadataFilter:
        key: str
        value: Any

    @dataclass
    class MetadataFilters:
        filters: list[Any]

    fake.MetadataFilter = MetadataFilter  # type: ignore[attr-defined]
    fake.MetadataFilters = MetadataFilters  # type: ignore[attr-defined]
    for name in ("llama_index", "llama_index.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores", fake)
    index = FakeIndex()
    await LlamaIndexRetriever(index).retrieve("q", filters={"scope": "north"})
    assert index.options[0]["filters"] == MetadataFilters([MetadataFilter("scope", "north")])


def test_llamaindex_filters_without_the_library_raise_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[llamaindex\]"):
        metadata_filters({"scope": "north"})


async def test_llamaindex_nodes_without_ids_or_scores_and_bare_nodes() -> None:
    bare = types.SimpleNamespace(text="Plain text", metadata=None)
    nan = FakeScored(FakeNode("", "Shop", {}), float("nan"))
    found = await LlamaIndexRetriever(FakeLlamaRetriever([bare, nan])).retrieve("q")
    assert [d.id for d in found] == [document_id("Plain text"), document_id("Shop")]
    assert [d.score for d in found] == [None, None]


async def test_llamaindex_bad_results() -> None:
    no_text = FakeLlamaRetriever([types.SimpleNamespace(text=None)])
    with pytest.raises(ToolUsageError, match="no text"):
        await LlamaIndexRetriever(no_text).retrieve("q")
    with pytest.raises(ToolUsageError):
        LlamaIndexRetriever(object())


async def test_llamaindex_backend_failure_is_an_integration_error() -> None:
    class Broken:
        async def aretrieve(self, query: str) -> list[Any]:
            raise ConnectionError("vector store down")

    with pytest.raises(IntegrationError) as info:
        await LlamaIndexRetriever(Broken()).retrieve("q")
    assert info.value.integration == "llamaindex"


# --- LangChain fakes: Document and a retriever --------------------------------------------------


@dataclass
class FakeLCDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


LC_DOCS = [
    FakeLCDocument("Refunds up to 24 hours", {"source": "refunds.md", "scope": "north"}, "r1"),
    FakeLCDocument("Shop bundle", {"source": "shop.md", "scope": "south", "score": 0.7}),
    FakeLCDocument("Parking", {"scope": "south", "id": "p1", "relevance_score": 0.2}),
]


class FakeLCRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def ainvoke(self, query: str, **kwargs: Any) -> list[Any]:
        self.calls.append((query, kwargs))
        return LC_DOCS


class SyncLCRetriever:
    def invoke(self, query: str, **kwargs: Any) -> list[Any]:
        return LC_DOCS


async def test_langchain_retriever_maps_documents_and_sends_no_kwargs_by_default() -> None:
    inner = FakeLCRetriever()
    found = await LangChainRetriever(inner).retrieve("refund", k=2)
    assert inner.calls == [("refund", {})]
    assert [(d.id, d.score, d.source.uri) for d in found] == [
        ("r1", None, "refunds.md"),
        (document_id("Shop bundle", "shop.md"), 0.7, "shop.md"),
    ]


async def test_langchain_search_kwargs_carry_k_and_filters() -> None:
    inner = FakeLCRetriever()
    retriever = LangChainRetriever(inner, search_kwargs=lambda k, f: {"k": k, "filter": f})
    found = await retriever.retrieve("q", k=3, filters={"scope": "south"})
    assert inner.calls == [("q", {"k": 3, "filter": {"scope": "south"}})]
    assert [(d.id, d.score) for d in found] == [
        (document_id("Shop bundle", "shop.md"), 0.7),
        ("p1", 0.2),
    ]


async def test_langchain_sync_invoke_runs_in_a_thread() -> None:
    found = await LangChainRetriever(SyncLCRetriever()).retrieve("q", k=1)
    assert [d.id for d in found] == ["r1"]


async def test_langchain_bad_inputs_and_results() -> None:
    with pytest.raises(ToolUsageError):
        LangChainRetriever(object())

    class NoContent:
        async def ainvoke(self, query: str) -> list[Any]:
            return [types.SimpleNamespace(page_content=None)]

    with pytest.raises(ToolUsageError, match="page_content"):
        await LangChainRetriever(NoContent()).retrieve("q")


# --- real libraries (own nox sessions) ----------------------------------------------------------


@pytest.mark.contract
async def test_real_langchain_retriever() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever

    class Keyword(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[LCDocument]:
            return [LCDocument(page_content=f"about {query}", metadata={"source": "a.md"}, id="a")]

    [found] = await LangChainRetriever(Keyword()).retrieve("tickets")
    assert (found.id, found.text, found.source.uri) == ("a", "about tickets", "a.md")


@pytest.mark.contract
async def test_real_llamaindex_filters_and_nodes() -> None:
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeWithScore, TextNode

    native = metadata_filters({"scope": "north"})
    assert native.filters[0].key == "scope"

    class Fixed:
        async def aretrieve(self, query: str) -> list[Any]:
            node = TextNode(id_="t1", text="Opening hours", metadata={"title": "Hours"})
            return [NodeWithScore(node=node, score=0.8)]

    [found] = await LlamaIndexRetriever(Fixed()).retrieve("hours")
    assert (found.id, found.score, found.source.title) == ("t1", 0.8, "Hours")


@pytest.mark.contract
async def test_real_llamaindex_index_filters_natively() -> None:
    pytest.importorskip("llama_index.core")
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode

    nodes = [
        TextNode(id_="n", text="North hours", metadata={"scope": "north"}),
        TextNode(id_="s", text="South hours", metadata={"scope": "south"}),
    ]
    index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=8))
    found = await LlamaIndexRetriever(index).retrieve("hours", k=2, filters={"scope": "south"})
    assert [d.id for d in found] == ["s"]
