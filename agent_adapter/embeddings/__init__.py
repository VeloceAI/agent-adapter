"""Embeddings: the `EmbeddingPort`, `CallableEmbedder` for any host function, and its cache."""

from agent_adapter.embeddings.cache import EmbeddingCache, embed_key
from agent_adapter.embeddings.callable import CallableEmbedder
from agent_adapter.embeddings.port import EmbeddingPort, EmbeddingResult

__all__ = [
    "CallableEmbedder",
    "EmbeddingCache",
    "EmbeddingPort",
    "EmbeddingResult",
    "embed_key",
]
