"""`retriever_tool`: a retriever as a registry tool that returns passages with their sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import CacheScope
from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.documents import Filters, FilterValue, check_filters, check_k
from agent_adapter.retrieval.port import RetrieverPort
from agent_adapter.runtime.current import current_context

RETRIEVAL_KIND = "retrieval"
DEFAULT_DESCRIPTION = (
    "Search the knowledge base. Returns passages, each with an id and a source; "
    "answer only from them and cite the source of every fact you use."
)

# Host-set filters: fixed, or computed per call from the context (e.g. the signed-in user).
ToolFilters = Filters | Callable[[ToolContext], Filters]


def retriever_tool(
    retriever: RetrieverPort,
    *,
    name: str = "search_docs",
    description: str = DEFAULT_DESCRIPTION,
    k: int = 5,
    filters: ToolFilters | None = None,
    include_metadata: bool = False,
    max_chars: int | None = None,
    cache_ttl: float | None = None,
    cache_scope: CacheScope = "run",
    timeout: float | None = None,
    tags: Iterable[str] = (),
) -> ToolSpec:
    """A tool (kind "retrieval") that searches `retriever` for the model's `query`.

    The model chooses only the query. `k` and `filters` are the host's, so a prompt can never
    widen a search to another scope. Calls run through the executor like any tool.

    Raises:
        ConfigError: a bad `k`, `max_chars` or `filters`, or per-call `filters` with
            `cache_scope="global"` (a cached result would cross scopes).
        ToolUsageError: `retriever` has no async `retrieve`.
    """
    if not isinstance(retriever, RetrieverPort):
        raise ToolUsageError("retriever_tool needs a RetrieverPort (async retrieve(query, k=))")
    try:
        k = check_k(k)
        fixed = None if filters is None or callable(filters) else check_filters(filters)
    except ToolUsageError as exc:
        raise ConfigError(f"retriever tool {name!r}: {exc}") from exc
    if max_chars is not None and (
        isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1
    ):
        raise ConfigError(f"retriever tool {name!r}: max_chars must be an int >= 1 or None")
    if callable(filters) and cache_scope == "global":
        raise ConfigError(
            f"retriever tool {name!r}: per-call filters need cache_scope 'run' or 'thread'"
        )
    guarded = retriever if isinstance(retriever, GuardedRetriever) else PortRetriever(retriever)
    metadata = bool(include_metadata)

    async def search(query: str, ctx: ToolContext) -> dict[str, Any]:
        """Search the documents.

        Args:
            query: What to look for, in plain words.
        """
        scope = _per_call(filters, ctx) if callable(filters) else fixed
        found = await guarded.retrieve(query, k=k, filters=scope, ctx=ctx)
        shown = [doc.cited(include_metadata=metadata, max_chars=max_chars) for doc in found]
        return {"documents": shown}

    return build_spec(
        search,
        name=name,
        description=description,
        kind=RETRIEVAL_KIND,
        tags=frozenset(tags),
        timeout=timeout,
        cache_ttl=cache_ttl,
        cache_scope=cache_scope,
    )


def _per_call(filters: Callable[[ToolContext], Filters], ctx: ToolContext) -> Filters:
    """The filters `fn(ctx)` returns; a blank str value (an unset user id) fails closed.

    Raises:
        ToolUsageError: invalid filters, or a blank str value.
    """
    scope = check_filters(filters(ctx))
    blank = sorted(
        key for key, value in scope.items() if isinstance(value, str) and not value.strip()
    )
    if blank:
        raise ToolUsageError(f"per-call filters {blank} are blank; refusing an unscoped search")
    return scope


class PortRetriever(GuardedRetriever):
    """Any other `RetrieverPort` behind the shared checks, filter enforcement and events."""

    backend = "port"

    def __init__(self, port: RetrieverPort, *, name: str = "") -> None:
        super().__init__(name=name or type(port).__name__)
        self.port = port

    async def _retrieve(self, query: str, k: int, filters: dict[str, FilterValue]) -> Sequence[Any]:
        found = await self.port.retrieve(query, k=k, filters=filters, ctx=current_context())
        if not isinstance(found, list):
            raise ToolUsageError(f"retriever {self.name!r} must return a list of Document")
        return found
