"""M5: what a failing idempotency store does after the tool ran (Security model row 4)."""

from __future__ import annotations

import pytest

from agent_adapter.server.idempotency import MemoryIdempotencyStore, Stored
from tests.server.helpers import call, invoke, make_registry, make_service

KEY = {"Idempotency-Key": "k-m5"}
EMAIL = {"to": "a@example.com", "body": "hi"}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FailingComplete(MemoryIdempotencyStore):
    async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
        raise ConnectionError("store down")


class FailingUpdate(MemoryIdempotencyStore):
    async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
        raise ConnectionError("store down")


async def test_a_failed_complete_keeps_the_response_and_a_retry_after_the_lease_reruns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[str] = []
    clock = Clock()
    service = make_service(make_registry(log), idempotency=FailingComplete(clock=clock))
    first = await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)
    assert first.status == 200 and first.body["result"] == 2
    assert "idempotency store complete failed: ConnectionError" in caplog.text
    assert "store down" not in caplog.text
    held = await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)
    assert held.status == 409  # the lease still holds the key
    clock.now += service.idempotency_lease_s + 1
    again = await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)
    assert again.status == 200 and "Idempotent-Replayed" not in again.headers
    assert log == ["add 1 1", "add 1 1"]  # ran twice: the documented residual


async def test_a_failed_update_after_a_decision_replays_the_pending_response() -> None:
    log: list[str] = []
    service = make_service(make_registry(log), idempotency=FailingUpdate())
    parked = await invoke(service, "send_email", EMAIL, headers=KEY)
    assert parked.status == 202
    path = f"/approvals/{parked.body['approval_id']}/decide"
    assert (await call(service, "POST", path, body={"approved": True})).status == 200
    replay = await invoke(service, "send_email", EMAIL, headers=KEY)
    assert (replay.status, replay.body) == (202, parked.body)
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert log == ["send a@example.com"]  # stale 202, never a second run
