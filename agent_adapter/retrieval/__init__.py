"""Retrieval (RAG): the `RetrieverPort`, `Document`, backends and a cited `retriever_tool`."""

from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.callable import CallableRetriever, RetrieveFn
from agent_adapter.retrieval.documents import (
    MAX_K,
    Document,
    Filters,
    FilterValue,
    Source,
)
from agent_adapter.retrieval.embedding import EmbeddingRetriever
from agent_adapter.retrieval.langchain import LangChainRetriever
from agent_adapter.retrieval.llamaindex import LlamaIndexRetriever
from agent_adapter.retrieval.port import RetrieverPort
from agent_adapter.retrieval.tool import RETRIEVAL_KIND, ToolFilters, retriever_tool

__all__ = [
    "MAX_K",
    "RETRIEVAL_KIND",
    "CallableRetriever",
    "Document",
    "EmbeddingRetriever",
    "FilterValue",
    "Filters",
    "GuardedRetriever",
    "LangChainRetriever",
    "LlamaIndexRetriever",
    "RetrieveFn",
    "RetrieverPort",
    "Source",
    "ToolFilters",
    "retriever_tool",
]
