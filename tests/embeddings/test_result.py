import math

import pytest
from pydantic import ValidationError

import agent_adapter.embeddings as embeddings
from agent_adapter.core.types import Usage
from agent_adapter.embeddings import CallableEmbedder, EmbeddingPort, EmbeddingResult


class Array:
    """Stands in for a numpy array or torch tensor."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


def test_package_exports_the_port_result_embedder_and_cache():
    assert set(embeddings.__all__) == {
        "CallableEmbedder",
        "EmbeddingCache",
        "EmbeddingPort",
        "EmbeddingResult",
        "embed_key",
    }
    assert all(hasattr(embeddings, name) for name in embeddings.__all__)


def test_result_holds_float_tuples_and_defaults():
    result = EmbeddingResult(vectors=[[1, 2.5], (3, 4)])
    assert result.vectors == ((1.0, 2.5), (3.0, 4.0))
    assert all(isinstance(x, float) for row in result.vectors for x in row)
    assert (result.model, result.usage, result.cache_hits) == ("", Usage(), 0)


def test_result_accepts_array_like_values():
    assert EmbeddingResult(vectors=Array([[0.5, 1.5]])).vectors == ((0.5, 1.5),)
    assert EmbeddingResult(vectors=[Array([1.0]), Array([2.0])]).vectors == ((1.0,), (2.0,))


def test_result_is_frozen_and_strict_on_fields():
    result = EmbeddingResult(vectors=[[1.0]])
    with pytest.raises(ValidationError):
        result.model = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        EmbeddingResult(vectors=[[1.0]], extra_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "vectors",
    [
        [[math.nan]],
        [[math.inf, 1.0]],
        [[]],
        ["abc"],
        "abc",
        [["a", "b"]],
        [["1.5"]],
        [[True, 1.0]],
        5,
        None,
        {"a": [1.0]},
    ],
)
def test_result_rejects_non_finite_empty_or_non_numeric_vectors(vectors):
    with pytest.raises(ValidationError):
        EmbeddingResult(vectors=vectors)


def test_result_rejects_negative_cache_hits():
    with pytest.raises(ValidationError):
        EmbeddingResult(vectors=(), cache_hits=-1)


def test_callable_embedder_is_an_embedding_port():
    async def fn(texts):
        return [[1.0] for _ in texts]

    embedder = CallableEmbedder(fn, model="m")
    assert isinstance(embedder, EmbeddingPort)
    assert (embedder.provider, embedder.model) == ("callable", "m")
