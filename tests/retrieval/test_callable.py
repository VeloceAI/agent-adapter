import asyncio
import logging
import time

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.errors import ConfigError, IntegrationError, ToolUsageError
from agent_adapter.retrieval import CallableRetriever, Document, RetrieverPort
from agent_adapter.runtime.current import current_context
from agent_adapter.tracing import EventLog
from tests.retrieval.support import CORPUS, doc


def corpus_fn(query: str, k: int, filters: dict[str, object]) -> list[Document]:
    return [d for d in CORPUS if all(d.metadata.get(key) == v for key, v in filters.items())][:k]


def test_callable_retriever_is_a_retriever_port() -> None:
    assert isinstance(CallableRetriever(corpus_fn), RetrieverPort)


def test_bad_fn_and_timeout_are_rejected() -> None:
    with pytest.raises(ToolUsageError):
        CallableRetriever("nope")  # type: ignore[arg-type]
    for bad in (0, -1, float("inf")):
        with pytest.raises(ConfigError):
            CallableRetriever(corpus_fn, timeout_s=bad)


async def test_sync_fn_gets_query_k_and_a_copy_of_filters() -> None:
    seen: list[tuple[str, int, dict[str, object]]] = []

    def fn(query: str, k: int, filters: dict[str, object]) -> list[str]:
        seen.append((query, k, filters))
        filters["scope"] = "changed"
        return ["one", "two", "three"]

    retriever = CallableRetriever(fn)
    found = await retriever.retrieve("  hours ", k=2, filters={"scope": "north"})
    assert seen == [("hours", 2, {"scope": "changed"})]
    # str results have no metadata, so a scope filter drops them: fail closed.
    assert found == []
    found = await retriever.retrieve("hours", k=2)
    assert [d.text for d in found] == ["one", "two"]


async def test_async_fn_and_awaitable_results_work() -> None:
    async def fn(query: str, k: int, filters: dict[str, object]) -> list[Document]:
        return list(CORPUS)

    def lazy(query: str, k: int, filters: dict[str, object]) -> object:
        return fn(query, k, filters)

    assert len(await CallableRetriever(fn).retrieve("q", k=10)) == 4
    assert len(await CallableRetriever(lazy).retrieve("q", k=3)) == 3


async def test_documents_outside_filters_and_repeats_are_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def leaky(query: str, k: int, filters: dict[str, object]) -> list[Document]:
        return [*CORPUS, CORPUS[0]]  # ignores filters, repeats one

    with caplog.at_level(logging.WARNING):
        found = await CallableRetriever(leaky).retrieve("q", k=5, filters={"scope": "north"})
    assert [d.id for d in found] == ["refunds", "hours"]
    assert "2 documents outside the filters" in caplog.text
    assert "north" not in caplog.text


@pytest.mark.parametrize("raw", ["text", {"text": "x"}, 42, [42]])
async def test_bad_results_raise_usage_errors(raw: object) -> None:
    retriever = CallableRetriever(lambda q, k, f: raw)
    with pytest.raises(ToolUsageError):
        await retriever.retrieve("q")


async def test_none_result_is_empty() -> None:
    assert await CallableRetriever(lambda q, k, f: None).retrieve("q") == []


async def test_bad_inputs_raise_before_fn_runs() -> None:
    calls: list[str] = []
    retriever = CallableRetriever(lambda q, k, f: calls.append(q))
    for kwargs in ({"k": 0}, {"filters": {"a": None}}):
        with pytest.raises(ToolUsageError):
            await retriever.retrieve("q", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError):
        await retriever.retrieve("")
    assert calls == []


async def test_backend_errors_become_integration_errors_without_their_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(query: str, k: int, filters: dict[str, object]) -> list[str]:
        raise RuntimeError("secret token abc")

    with caplog.at_level(logging.ERROR), pytest.raises(IntegrationError) as info:
        await CallableRetriever(broken, name="faq").retrieve("q")
    assert info.value.integration == "callable"
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "secret" not in str(info.value)
    assert "RuntimeError" in caplog.text
    assert "secret" not in "\n".join(r.getMessage() for r in caplog.records if r.levelno >= 30)


async def test_adapter_errors_pass_through() -> None:
    def denied(query: str, k: int, filters: dict[str, object]) -> list[str]:
        raise ConfigError("bad index name")

    with pytest.raises(ConfigError):
        await CallableRetriever(denied).retrieve("q")


async def test_timeout_raises_integration_error() -> None:
    async def slow(query: str, k: int, filters: dict[str, object]) -> list[str]:
        await asyncio.sleep(5)
        return []

    with pytest.raises(IntegrationError, match="timed out"):
        await CallableRetriever(slow, timeout_s=0.01).retrieve("q")


async def test_fn_raising_its_own_timeout_is_a_failure_not_ours() -> None:
    async def flaky(query: str, k: int, filters: dict[str, object]) -> list[str]:
        raise TimeoutError

    with pytest.raises(IntegrationError, match="retrieval failed"):
        await CallableRetriever(flaky, timeout_s=5).retrieve("q")


async def test_fn_sees_the_context_as_current() -> None:
    seen: list[ToolContext | None] = []

    def fn(query: str, k: int, filters: dict[str, object]) -> list[str]:
        seen.append(current_context())
        return []

    ctx = ToolContext(run_id="run_1", user_id="ann")
    await CallableRetriever(fn).retrieve("q", ctx=ctx)
    assert seen[0] is not None and seen[0].user_id == "ann"


async def test_searched_event_has_counts_and_ids_but_no_text() -> None:
    sink = EventLog()
    ctx = ToolContext(run_id="run_1", call_id="call_1", sink=sink)
    retriever = CallableRetriever(corpus_fn, name="help")
    await retriever.retrieve("refund", k=1, filters={"scope": "north"}, ctx=ctx)
    [event] = sink.events
    assert event.type == "retrieval.searched"
    assert (event.run_id, event.call_id) == ("run_1", "call_1")
    data = dict(event.data)
    assert data.pop("latency_ms") >= 0
    assert data == {
        "retriever": "help",
        "backend": "callable",
        "k": 1,
        "count": 1,
        "ids": ["refunds"],
        "dropped": 0,
    }


async def test_capture_content_adds_query_and_texts() -> None:
    sink = EventLog()
    retriever = CallableRetriever(corpus_fn, capture_content=True)
    await retriever.retrieve("refund", k=1, ctx=ToolContext(sink=sink))
    [event] = sink.events
    assert event.data["arguments"] == {"query": "refund"}
    assert event.data["result"] == [CORPUS[0].text]


async def test_failed_event_has_the_error_kind() -> None:
    sink = EventLog()
    retriever = CallableRetriever(lambda q, k, f: 1 / 0)
    with pytest.raises(IntegrationError):
        await retriever.retrieve("q", ctx=ToolContext(sink=sink))
    [event] = sink.events
    assert event.type == "retrieval.failed"
    assert event.data["error_kind"] == "integration-error"
    retriever = CallableRetriever(lambda q, k, f: [3])
    with pytest.raises(ToolUsageError):
        await retriever.retrieve("q", ctx=ToolContext(sink=sink))
    assert sink.events[-1].data["error_kind"] == "tool-usage-error"


async def test_no_sink_no_event_and_current_context_is_used() -> None:
    from agent_adapter.runtime.current import use_context

    sink = EventLog()
    with use_context(ToolContext(run_id="run_cur", sink=sink)):
        await CallableRetriever(corpus_fn).retrieve("q")
    assert sink.events[0].run_id == "run_cur"
    await CallableRetriever(corpus_fn).retrieve("q", ctx=ToolContext())  # no sink: no error


def test_retrieve_sync_blocks_until_done() -> None:
    retriever = CallableRetriever(corpus_fn)
    found = retriever.retrieve_sync("parking", k=2, filters={"scope": "south"})
    assert [d.id for d in found] == ["parking", "shop"]


def test_retrieve_sync_timeout() -> None:
    def slow(query: str, k: int, filters: dict[str, object]) -> list[str]:
        time.sleep(0.5)
        return []

    with pytest.raises(TimeoutError):
        CallableRetriever(slow).retrieve_sync("q", timeout=0.05)


def test_doc_helper_is_consistent() -> None:
    assert doc("a", "t", scope="x").metadata == {"scope": "x"}
