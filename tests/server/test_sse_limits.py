"""Event-stream subscriptions are capped per principal and in total (fix round 1)."""

import gc

from tests.server.helpers import call, drain, make_service


async def open_stream(service, token: str = "t-alice", run: str = "r1"):  # type: ignore[no-untyped-def]
    return await call(service, "GET", f"/runs/{run}/events", token=token)


async def test_streams_are_capped_per_principal_with_retry_after():
    service = make_service()
    service.max_streams_per_principal = 2
    first, second = await open_stream(service), await open_stream(service, run="r2")
    assert (first.status, second.status) == (200, 200)
    third = await open_stream(service, run="r3")
    assert third.status == 429 and third.headers["Retry-After"]
    assert third.body["type"].endswith(":too-many-streams")
    assert (await open_stream(service, token="t-bob")).status == 200
    assert first.raw.stream is not None
    await first.raw.stream.aclose()  # a closed stream frees its slot
    assert (await open_stream(service, run="r4")).status == 200


async def test_streams_are_capped_in_total():
    service = make_service()
    service.max_streams = 2
    held = [await open_stream(service), await open_stream(service, token="t-bob")]
    assert (await open_stream(service, token="t-reviewer")).status == 429
    assert all(reply.status == 200 for reply in held)


async def test_a_finished_or_dropped_stream_frees_its_slot():
    service = make_service()
    service.max_streams_per_principal = 1
    service.sse_idle_s = 0.01
    finished = await open_stream(service)
    assert [f["event"] for f in await drain(finished.raw)] == ["end"]
    dropped = await open_stream(service)
    assert dropped.status == 200
    del dropped
    gc.collect()  # never iterated, never closed: released when collected
    assert (await open_stream(service)).status == 200
