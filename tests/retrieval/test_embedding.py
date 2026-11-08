from collections.abc import Sequence

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.errors import ConfigError, ModelError, ToolUsageError
from agent_adapter.retrieval import Document, EmbeddingRetriever
from agent_adapter.retrieval import embedding as embedding_module
from tests.retrieval.support import CORPUS, doc, embed, embedder


async def test_ranks_by_cosine_similarity_with_scores() -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS)
    found = await retriever.retrieve("museum opening hours", k=2)
    assert [d.id for d in found] == ["hours", "parking"]
    assert found[0].score is not None and found[1].score is not None
    assert 0 < found[1].score < found[0].score <= 1
    assert found[0].source.uri == "https://help.example.com/hours"


async def test_ties_keep_corpus_order_and_unrelated_queries_score_zero() -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS)
    found = await retriever.retrieve("weather", k=4)
    assert [d.id for d in found] == [d.id for d in CORPUS]
    assert {d.score for d in found} == {0.0}


async def test_filters_are_applied_before_ranking() -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS)
    found = await retriever.retrieve("ticket", k=5, filters={"scope": "south"})
    assert [d.id for d in found] == ["shop", "parking"]


async def test_scope_key_isolates_scopes() -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS, scope_key="scope")
    north = await retriever.retrieve("ticket parking", k=10, filters={"scope": "north"})
    south = await retriever.retrieve("ticket parking", k=10, filters={"scope": "south"})
    assert {d.metadata["scope"] for d in north} == {"north"}
    assert {d.metadata["scope"] for d in south} == {"south"}
    assert await retriever.retrieve("ticket", k=10, filters={"scope": "west"}) == []


@pytest.mark.parametrize(
    "filters", [None, {}, {"other": "x"}, {"scope": 1}, {"scope": ""}, {"scope": "  "}]
)
async def test_scope_key_refuses_an_unscoped_search(filters: dict[str, object] | None) -> None:
    retriever = EmbeddingRetriever(embedder(), CORPUS, scope_key="scope")
    with pytest.raises(ToolUsageError, match="scope"):
        await retriever.retrieve("ticket", filters=filters)  # type: ignore[arg-type]


@pytest.mark.parametrize("scope", [None, "", " ", 3])
def test_scope_key_refuses_an_unscoped_document(scope: object) -> None:
    metadata = {} if scope is None else {"scope": scope}
    with pytest.raises(ToolUsageError, match="non-empty str metadata"):
        EmbeddingRetriever(embedder(), [doc("x", "no scope", **metadata)], scope_key="scope")
    retriever = EmbeddingRetriever(embedder(), scope_key="scope")
    with pytest.raises(ToolUsageError):
        retriever.add_sync([doc("y", "late", **metadata)])
    assert len(retriever) == 0


def test_scope_key_unset_ignores_empty_scopes() -> None:
    retriever = EmbeddingRetriever(embedder(), [doc("x", "ticket", scope="")])
    assert [d.id for d in retriever.retrieve_sync("ticket", filters={"scope": ""})] == ["x"]


def test_scope_key_must_be_a_non_empty_str() -> None:
    with pytest.raises(ConfigError):
        EmbeddingRetriever(embedder(), scope_key="")


def test_rejects_a_non_embedder_and_repeated_ids() -> None:
    with pytest.raises(ToolUsageError):
        EmbeddingRetriever(object(), CORPUS)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="unique"):
        EmbeddingRetriever(embedder(), [CORPUS[0], CORPUS[0]])
    retriever = EmbeddingRetriever(embedder(), CORPUS[:1])
    with pytest.raises(ToolUsageError, match="unique"):
        retriever.add_sync([CORPUS[0]])


def test_caps_the_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module, "MAX_DOCUMENTS", 2)
    with pytest.raises(ToolUsageError, match="at most 2"):
        EmbeddingRetriever(embedder(), CORPUS)


async def test_empty_corpus_returns_nothing_and_add_embeds_now() -> None:
    texts: list[int] = []

    def counting(batch: list[str]) -> list[list[float]]:
        texts.append(len(batch))
        return embed(batch)

    retriever = EmbeddingRetriever(CallableEmbedder(counting))
    assert await retriever.retrieve("ticket") == []
    assert texts == []
    assert await retriever.add(["A ticket refund", "Parking info"]) == 2
    assert await retriever.add([]) == 0
    assert texts == [2]
    assert len(retriever) == 2
    found = await retriever.retrieve("refund", k=1)
    assert found[0].text == "A ticket refund"
    assert texts == [2, 1]  # only the query


def test_add_sync_and_retrieve_sync() -> None:
    retriever = EmbeddingRetriever(embedder())
    assert retriever.add_sync(CORPUS) == 4
    assert retriever.retrieve_sync("parking", k=1)[0].id == "parking"


async def test_a_failed_embedding_keeps_documents_queued() -> None:
    fail = [True]

    def flaky(batch: list[str]) -> list[list[float]]:
        if fail[0]:
            raise RuntimeError("down")
        return embed(batch)

    retriever = EmbeddingRetriever(CallableEmbedder(flaky), CORPUS)
    with pytest.raises(ModelError):
        await retriever.retrieve("ticket")
    fail[0] = False
    assert len(await retriever.retrieve("ticket", k=10)) == 4


class OddEmbedder:
    """An EmbeddingPort that returns what it is told to."""

    provider = "odd"
    model = "odd-1"

    def __init__(self, rows: list[list[list[float]]]) -> None:
        self.rows = rows

    async def embed(
        self, texts: Sequence[str], *, ctx: ToolContext | None = None
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=self.rows.pop(0))


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ([[[1.0, 0.0]]], "1 vectors for 2 texts"),
        ([[[1.0, 0.0], [1.0, 0.0, 0.0]]], "different sizes"),
    ],
)
async def test_bad_embedder_output_is_a_config_error(
    rows: list[list[list[float]]], problem: str
) -> None:
    retriever = EmbeddingRetriever(OddEmbedder(rows), [doc("a", "x"), doc("b", "y")])
    with pytest.raises(ConfigError, match=problem):
        await retriever.retrieve("q")
    assert len(retriever) == 2  # still queued


async def test_a_changed_vector_size_is_caught() -> None:
    odd = OddEmbedder([[[1.0, 0.0]]])
    retriever = EmbeddingRetriever(odd, [doc("a", "x")])
    assert await retriever.add([]) == 1  # embeds the queued document
    odd.rows = [[[1.0, 0.0, 0.0]]]
    with pytest.raises(ConfigError, match="query vector has 3 dimensions"):
        await retriever.retrieve("q")
    odd.rows = [[[1.0, 0.0, 0.0]]]
    with pytest.raises(ConfigError, match="different sizes"):
        await retriever.add([doc("b", "y")])


async def test_zero_vectors_score_zero() -> None:
    odd = OddEmbedder([[[0.0, 0.0]], [[1.0, 0.0]]])
    retriever = EmbeddingRetriever(odd, [Document(id="z", text="zero")])
    [found] = await retriever.retrieve("q")
    assert found.score == 0.0
