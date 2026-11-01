"""Review fixes: pinned outbound connects, bounded task store, per-owner parked-call cap."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

import httpcore
from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.types import Message, Role, Task, TaskState, TaskStatus
from google.protobuf.json_format import MessageToDict

from agent_adapter.core.errors import IntegrationError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.exec import Executor
from agent_adapter.integrations.a2a import (
    HostRefused,
    _guard,
    a2a_app,
    agent_card,
    bounded_task_store,
    connect_a2a,
    static_bearer,
)
from agent_adapter.integrations.a2a._pending import PendingCall, PendingCalls
from tests.integrations.a2a.support import RawClient, data, serving, state, status_data
from tests.integrations.a2a.test_a2a_approval import ALICE, BOB, TOKENS, Tools, _deferred

PUBLIC = "93.184.216.34"


class RecordingBackend(httpcore.AsyncMockBackend):
    """A mock network that records where each socket would go."""

    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__(buffer)
        self.connects: list[str] = []

    async def connect_tcp(self, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
        self.connects.append(host)
        return await super().connect_tcp(host, port, *args, **kwargs)


def _http(status: int, body: bytes) -> list[bytes]:
    head = f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
    return [(head + f"Content-Length: {len(body)}\r\n\r\n").encode(), body]


def _card_bytes(url: str) -> bytes:
    registry = Registry()
    registry.add(build_spec(lambda: 1, name="one", description="One."))
    return json.dumps(MessageToDict(agent_card(registry, url=url))).encode()


def _network(monkeypatch: pytest.MonkeyPatch, answers: list[list[str]], buffer: list[bytes]):
    backend = RecordingBackend(buffer)
    lookups: list[str] = []

    async def resolver(host: str, port: int) -> list[str]:
        lookups.append(host)
        return answers[min(len(lookups), len(answers)) - 1]

    monkeypatch.setattr(_guard, "resolve", resolver)
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    return backend, lookups


async def test_dns_rebinding_after_the_check_never_connects_to_the_private_answer(monkeypatch):
    backend, _ = _network(monkeypatch, [[PUBLIC], ["127.0.0.1"]], [])
    with pytest.raises(HostRefused, match="private"):
        await connect_a2a("http://rebind.test/")
    assert backend.connects == []


async def test_the_socket_goes_to_the_address_the_guard_checked(monkeypatch):
    backend, lookups = _network(
        monkeypatch, [[PUBLIC]], _http(200, _card_bytes("http://agent.example/"))
    )
    connection = await connect_a2a("http://agent.example/")
    await connection.aclose()
    assert backend.connects == [PUBLIC]
    assert lookups == ["agent.example", "agent.example"]  # the up-front check, then the connect


async def test_card_errors_show_only_the_origin(monkeypatch, caplog):
    _network(monkeypatch, [[PUBLIC]], _http(500, b"{}"))
    caplog.set_level(logging.WARNING)
    with pytest.raises(IntegrationError) as raised:
        await connect_a2a("http://user:hunter2@agent.example:8080/x?token=s3cret")
    shown = str(raised.value) + caplog.text
    assert "http://agent.example:8080" in shown
    assert "hunter2" not in shown and "s3cret" not in shown and "user" not in shown


def test_one_owner_cannot_evict_another_owners_parked_calls():
    table = PendingCalls(max_entries=10, max_per_owner=1)
    call = PendingCall("t", {}, "run", "apr_1", "port")
    assert table.park("alice", "task_1", call)
    assert not table.park("alice", "task_2", call)
    assert table.park("bob", "task_3", call)
    assert table.take("alice", "task_1") == call and table.take("bob", "task_3") == call
    assert table.park("alice", "task_2", call)  # room again once taken


async def test_a_caller_over_its_parked_call_cap_is_refused():
    tools, (executor, _) = Tools(), _deferred()
    app = a2a_app(
        tools.registry(),
        url="http://agent.test/",
        executor=executor,
        authenticate=static_bearer(TOKENS),
        allowed_hosts=["agent.test"],
    )
    app.state.a2a_agent.pending = PendingCalls(max_per_owner=1)
    async with serving(app), RawClient(app, ALICE) as alice, RawClient(app, BOB) as bob:
        assert state(await alice.call("remove", {"item": "a"})) == "input_required"
        refused = await alice.call("remove", {"item": "b"})
        assert state(refused) == "rejected"
        assert status_data(refused)["error"]["kind"] == "too-many-pending"
        assert state(await bob.call("remove", {"item": "c"})) == "input_required"
    assert tools.removed == []


class _Owner(User):  # type: ignore[misc]
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.name


def _ctx(owner: str) -> ServerCallContext:
    return ServerCallContext(user=_Owner(owner))


def _task(task_id: str, task_state: Any = TaskState.TASK_STATE_COMPLETED) -> Task:
    message = Message(role=Role.ROLE_USER, message_id="m", context_id="c")
    return Task(id=task_id, context_id="c", status=TaskStatus(state=task_state), history=[message])


async def test_task_store_expires_tasks_after_the_ttl():
    clock = [0.0]
    store = bounded_task_store(ttl=10, clock=lambda: clock[0])
    await store.save(_task("t1"), _ctx("alice"))
    assert (await store.get("t1", _ctx("alice"))) is not None
    assert (await store.get("t1", _ctx("bob"))) is None
    clock[0] = 11.0
    assert (await store.get("t1", _ctx("alice"))) is None
    assert len(store) == 0 and store.inner._impl.tasks == {}


async def test_task_store_caps_drop_finished_tasks_first_and_never_another_owners():
    store = bounded_task_store(max_tasks=3, max_per_owner=2)
    working = TaskState.TASK_STATE_INPUT_REQUIRED
    await store.save(_task("b1", working), _ctx("bob"))
    await store.save(_task("a1", working), _ctx("alice"))
    await store.save(_task("a2"), _ctx("alice"))
    await store.save(_task("a3"), _ctx("alice"))  # alice's cap: drops her finished a2
    assert (await store.get("a2", _ctx("alice"))) is None
    assert (await store.get("a1", _ctx("alice"))) is not None
    await store.save(_task("c1"), _ctx("carol"))  # total cap: the oldest finished (a3) goes
    assert (await store.get("a3", _ctx("alice"))) is None
    assert (await store.get("b1", _ctx("bob"))) is not None and len(store) == 3


async def test_rejected_messages_are_not_kept_with_their_history():
    tools = Tools()
    app = a2a_app(
        tools.registry(),
        url="http://agent.test/",
        executor=Executor(),
        authenticate=static_bearer(TOKENS),
        allowed_hosts=["agent.test"],
    )
    async with serving(app), RawClient(app, ALICE) as alice:
        rejected = await alice.send([data({"tool": "nope", "padding": "x" * 50_000})])
        assert state(rejected) == "rejected"
        kept = await app.state.a2a_task_store.get(rejected.id, _ctx("alice"))
    assert kept is not None and len(kept.history) == 0
    assert kept.status.state == TaskState.TASK_STATE_REJECTED


def test_bounds_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        bounded_task_store(max_tasks=0)


class FlakyBackend(RecordingBackend):
    """Refuses connects to the addresses in `down`."""

    def __init__(self, buffer: list[bytes], down: set[str]) -> None:
        super().__init__(buffer)
        self.down = down

    async def connect_tcp(self, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
        if host in self.down:
            self.connects.append(host)
            raise httpcore.ConnectError("refused")
        return await super().connect_tcp(host, port, *args, **kwargs)


async def test_a_refused_address_falls_back_to_the_next_checked_one(monkeypatch):
    other = "8.8.8.8"
    backend = FlakyBackend(_http(200, _card_bytes("http://agent.example/")), {PUBLIC})
    _network(monkeypatch, [[PUBLIC, other]], [])
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    connection = await connect_a2a("http://agent.example/")
    await connection.aclose()
    assert backend.connects == [PUBLIC, other]


async def test_no_reachable_address_is_an_integration_error(monkeypatch):
    backend = FlakyBackend([], {PUBLIC})
    _network(monkeypatch, [[PUBLIC]], [])
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    with pytest.raises(IntegrationError, match="ConnectError"):
        await connect_a2a("http://agent.example/")
