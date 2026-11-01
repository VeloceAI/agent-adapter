"""Unit tests for the server's private parts: SSE framing, replay cache, owner table."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_adapter.core.errors import Forbidden
from agent_adapter.core.types import Event
from agent_adapter.server import _sse
from agent_adapter.server._idempotency import CacheIdempotency, Replays, fingerprint
from agent_adapter.server._runs import RunOwners
from agent_adapter.server.auth import Principal, RequestInfo
from agent_adapter.server.idempotency import Stored
from tests.fakes.cache import BrokenCache, DictCache
from tests.server.helpers import call, invoke, make_service


async def collect(stream: AsyncIterator[bytes]) -> list[bytes]:
    return [chunk async for chunk in stream]


async def events(*items: tuple[int, Event]) -> AsyncIterator[tuple[int, Event]]:
    for item in items:
        yield item


async def forever() -> AsyncIterator[tuple[int, Event]]:
    await asyncio.Event().wait()
    yield 1, Event("never")  # pragma: no cover


def test_frames_name_the_event_and_show_the_client_run_key():
    raw = _sse.frame(3, Event("custom.x\r\nevent: forged", run_id="run_internal"), "r1")
    assert raw is not None
    head, data = raw.decode().split("\ndata: ")
    assert head == "id: 3\nevent: custom.xevent: forged"  # no line break: no injected frame
    assert json.loads(data)["run_id"] == "r1"


def test_unencodable_events_are_skipped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_sse, "to_jsonable", lambda value: {"x": float("nan")})
    assert _sse.frame(1, Event("tool.started"), "r") is None


async def test_the_stream_ends_when_the_source_ends_or_run_finishes():
    out = await collect(
        _sse.stream(
            events((1, Event("tool.started")), (2, Event("run.finished")), (3, Event("late"))),
            run_key="r",
            close_on=(),
            keep_alive_s=1,
            idle_s=1,
        )
    )
    assert [chunk.split(b"\n")[1] for chunk in out[:2]] == [
        b"event: tool.started",
        b"event: run.finished",
    ]
    assert out[-1] == _sse.END and len(out) == 3
    ended = await collect(_sse.stream(events(), run_key="r", close_on=(), keep_alive_s=1, idle_s=1))
    assert ended == [_sse.END]


async def test_keep_alives_then_an_idle_close():
    out = await collect(
        _sse.stream(forever(), run_key="r", close_on=(), keep_alive_s=0.005, idle_s=0.03)
    )
    assert out[-1] == _sse.END
    assert _sse.KEEP_ALIVE in out


async def test_an_expired_idle_window_closes_at_once():
    ticks = iter([0.0, 10.0])
    out = await collect(
        _sse.stream(
            forever(),
            run_key="r",
            close_on=(),
            keep_alive_s=1,
            idle_s=5,
            clock=lambda: next(ticks),
        )
    )
    assert out == [_sse.END]


async def test_a_keep_alive_is_not_sent_once_idle_time_ran_out():
    ticks = iter([0.0, 0.0, 10.0, 10.0])
    out = await collect(
        _sse.stream(
            forever(),
            run_key="r",
            close_on=(),
            keep_alive_s=0.001,
            idle_s=5,
            clock=lambda: next(ticks),
        )
    )
    assert out == [_sse.END]


async def test_a_failing_source_ends_the_stream(caplog: pytest.LogCaptureFixture):
    async def broken() -> AsyncIterator[tuple[int, Event]]:
        raise RuntimeError("down")
        yield  # pragma: no cover

    out = await collect(_sse.stream(broken(), run_key="r", close_on=(), keep_alive_s=1, idle_s=1))
    assert out == [_sse.END]
    assert "subscription failed: RuntimeError" in caplog.text


async def test_a_source_without_aclose_is_fine():
    class Plain:
        def __init__(self) -> None:
            self.items = [(1, Event("tool.finished"))]

        def __aiter__(self) -> "Plain":
            return self

        async def __anext__(self) -> tuple[int, Event]:
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop()

    out = await collect(
        _sse.stream(Plain(), run_key="r", close_on={"tool.finished"}, keep_alive_s=1, idle_s=1)
    )
    assert out[-1] == _sse.END and len(out) == 2


async def test_cache_entries_round_trip_and_bad_entries_are_misses():
    cache = DictCache()
    store = CacheIdempotency(cache)
    replays = Replays(store, ttl=60, lease=30)
    key = replays.key("alice", "k")
    assert key.startswith("aa:http:idem:v1:") and key != replays.key("bob", "k")
    claim = await store.claim(key, "fp", 30)
    stored = Stored("fp", 202, {"a": 1}, headers={"Location": "/x"})
    assert await store.complete(key, claim.token, stored, 60)
    assert (await store.claim(key, "fp", 30)).stored == stored
    for raw in (b"not json", b'{"v": 2}', b'{"v": 1}', b"[1]"):
        await cache.set(key, raw)
        assert (await store.claim(key, "fp", 30)).state == "new"
        await store.release(key, "not-mine")
        store._busy.clear()
    assert not await store.update("missing", stored, 60)


async def test_a_broken_cache_degrades_to_no_replay(caplog: pytest.LogCaptureFixture):
    store = CacheIdempotency(BrokenCache())
    claim = await store.claim("k", "fp", 30)
    assert claim.state == "new"
    await store.complete("k", claim.token, Stored("fp", 200, {}), 60)
    assert "idempotency cache write failed" in caplog.text
    assert "idempotency cache read failed" in caplog.text


async def test_cache_wrapper_lets_cancellation_through():
    class Cancelling(DictCache):
        async def get(self, key: str) -> bytes | None:
            raise asyncio.CancelledError

        async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
            raise asyncio.CancelledError

    store = CacheIdempotency(Cancelling())
    with pytest.raises(asyncio.CancelledError):
        await store.claim("k", "fp", 30)
    with pytest.raises(asyncio.CancelledError):
        await store.complete("k2", "t", Stored("fp", 200, {}), 60)


async def test_replays_let_cancellation_through_every_store_call():
    class Cancelled:
        async def claim(self, *args: Any) -> Any:
            raise asyncio.CancelledError

        complete = release = update = claim

    replays = Replays(Cancelled(), ttl=60, lease=30)  # type: ignore[arg-type]
    for call_ in (
        replays.claim("k", "fp"),
        replays.complete("k", "t", Stored("fp", 200, {})),
        replays.release("k", "t"),
        replays.update("k", Stored("fp", 200, {})),
    ):
        with pytest.raises(asyncio.CancelledError):
            await call_


def test_fingerprints_cover_tool_arguments_run_and_thread():
    base = fingerprint("t", {"a": 1}, "r", "th")
    assert base == fingerprint("t", {"a": 1}, "r", "th")
    variants = [
        fingerprint("u", {"a": 1}, "r", "th"),
        fingerprint("t", {"a": 2}, "r", "th"),
        fingerprint("t", {"a": 1}, "", "th"),
        fingerprint("t", {"a": 1}, "r", ""),
    ]
    assert base not in variants


def test_the_owner_table_is_bounded_with_a_per_principal_quota():
    owners = RunOwners(max_runs=3, per_principal=2)
    assert owners.claim("a1", "alice") and owners.claim("a2", "alice")
    assert owners.claim("a3", "alice")  # alice's own oldest ("a1") goes
    assert owners.owner("a1") is None and owners.owner("a3") == "alice"
    assert owners.claim("b1", "bob") and owners.claim("b2", "bob")  # global cap: oldest goes
    assert owners.owner("a2") is None
    assert not owners.claim("b1", "alice")
    assert owners.claim("b1", "bob")


async def test_a_forbidden_from_authorize_is_a_refusal():
    async def no(principal: Principal, spec: Any, args: Any) -> bool:
        raise Forbidden("no")

    assert (await invoke(make_service(authorize=no), "add", {"a": 1, "b": 1})).status == 403

    async def no_decide(principal: Principal, record: Any) -> bool:
        raise Forbidden("no")

    service = make_service(authorize_decision=no_decide)
    parked = await invoke(service, "send_email", {"to": "x", "body": "y"})
    path = f"/approvals/{parked.body['approval_id']}/decide"
    assert (await call(service, "POST", path, body={"approved": True})).status == 403


@pytest.mark.parametrize("where", ["authorize", "authenticate"])
async def test_cancellation_in_hooks_propagates(where: str):
    async def cancelled(*args: Any) -> Any:
        raise asyncio.CancelledError

    service = make_service(**{where: cancelled})
    with pytest.raises(asyncio.CancelledError):
        await service.handle(
            RequestInfo(
                "POST",
                "/tools/add/invoke",
                {"Authorization": "Bearer t-alice", "Content-Type": "application/json"},
                {},
                None,
            ),
            b'{"arguments": {"a": 1, "b": 1}}',
        )


async def test_cancellation_in_the_decision_rule_propagates():
    async def cancelled(*args: Any) -> Any:
        raise asyncio.CancelledError

    service = make_service(authorize_decision=cancelled)
    parked = await invoke(service, "send_email", {"to": "x", "body": "y"})
    with pytest.raises(asyncio.CancelledError):
        await call(
            service,
            "POST",
            f"/approvals/{parked.body['approval_id']}/decide",
            body={"approved": True},
        )


async def test_unexpected_handler_errors_are_500_with_the_class_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    service = make_service()

    async def explode(*args: Any) -> Any:
        raise KeyError("secret-key")

    monkeypatch.setattr(service, "_catalogue", explode)
    reply = await call(service, "GET", "/tools")
    assert reply.status == 500 and reply.body["detail"] == "KeyError"
    assert "GET tools failed (KeyError)" in caplog.text
    assert "secret-key" not in caplog.text  # the exception text stays at DEBUG


async def test_cache_wrapper_updates_only_its_own_entry_and_checks_tokens():
    store = CacheIdempotency(DictCache())
    claim = await store.claim("k", "fp", 30)
    assert not await store.complete("k", "not-mine", Stored("fp", 200, {}), 60)
    assert await store.update("k", Stored("fp", 201, {}), 60)
    assert not await store.update("k", Stored("other", 201, {}), 60)
    await store.release("k", claim.token)
    assert (await store.claim("k", "fp", 30)).stored == Stored("fp", 201, {})


async def test_a_cancelled_call_frees_its_idempotency_key():
    gate = asyncio.Event()
    from agent_adapter.core.registry import Registry

    reg = Registry()

    @reg.tool
    async def wait() -> str:
        """Wait for the gate."""
        await gate.wait()
        return "done"

    service = make_service(reg)
    key = {"Idempotency-Key": "k-cancel"}
    task = asyncio.ensure_future(invoke(service, "wait", {}, headers=key))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    assert (await invoke(service, "wait", {}, headers=key)).status == 200
