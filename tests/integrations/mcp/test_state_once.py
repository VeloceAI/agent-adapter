"""Client-approval states are single use everywhere; resumes are never re-elicited."""

from __future__ import annotations

import secrets
import time
from typing import Any

import pytest

pytest.importorskip("mcp")

import mcp.types as t
from mcp import Client, MCPError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import (
    APPROVAL_ID_META,
    CALL_ID_META,
    RUN_ID_META,
    _approval,
    mcp_server,
    streamable_http_app,
)
from agent_adapter.integrations.mcp._approval import APPROVAL_INPUT_KEY
from agent_adapter.integrations.mcp._resume import STATE_TTL_S, SpentStates

pytestmark = pytest.mark.contract

YES = {APPROVAL_INPUT_KEY: t.ElicitResult(action="accept", content={"approved": True})}
INVALID = "Invalid or expired requestState"


def make_registry(log: list[str]) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True, once_per_run=True)
    async def wire(amount: int) -> str:
        """Wire money once per run (needs approval)."""
        log.append(f"wire {amount}")
        return f"wired {amount}"

    return reg


class Elicitor:
    def __init__(self) -> None:
        self.asked = 0

    async def __call__(self, ctx: Any, params: Any) -> t.ElicitResult:
        self.asked += 1
        return t.ElicitResult(action="accept", content={"approved": True})


async def first_round(client: Client, args: dict[str, Any]) -> str:
    first = await client.session.call_tool("wire", args, allow_input_required=True)
    assert isinstance(first, t.InputRequiredResult) and first.request_state is not None
    return first.request_state


async def replay(client: Client, args: dict[str, Any], state: str) -> Any:
    return await client.session.call_tool("wire", args, input_responses=YES, request_state=state)


async def test_a_second_worker_with_a_shared_approval_store_refuses_the_replay():
    log: list[str] = []
    store, keys = MemoryApprovalStore(), [secrets.token_hex(32)]
    workers = [
        mcp_server(
            make_registry(log),
            executor=Executor(),
            name="payments",
            client_approval=True,
            request_state_keys=keys,
            approval_store=store,
        )
        for _ in range(2)
    ]
    async with Client(workers[0], elicitation_callback=Elicitor()) as one:
        state = await first_round(one, {"amount": 5})
        done = await replay(one, {"amount": 5}, state)
    assert isinstance(done, t.CallToolResult) and not done.is_error
    async with Client(workers[1], elicitation_callback=Elicitor()) as two:
        with pytest.raises(MCPError, match=INVALID):
            await replay(two, {"amount": 5}, state)
    assert log == ["wire 5"]
    (record,) = await store.list()
    assert record.consumed


def test_request_state_keys_need_a_shared_approval_store():
    keys = [secrets.token_hex(32)]
    for build in (mcp_server, streamable_http_app):
        with pytest.raises(ConfigError, match="shared ApprovalStore"):
            build(
                Registry(),
                executor=Executor(),
                name="p",
                client_approval=True,
                request_state_keys=keys,
            )
    store = MemoryApprovalStore()
    assert mcp_server(
        Registry(),
        executor=Executor(approval=DeferredApproval(store)),
        name="p",
        client_approval=True,
        request_state_keys=keys,
    )


def test_a_full_spent_set_refuses_rather_than_forgets():
    spent = SpentStates(max_entries=1)
    deadline = time.time() + STATE_TTL_S
    assert spent.use(("r", "a"), deadline)
    with pytest.raises(MCPError, match="approval states busy"):
        spent.use(("r", "b"), deadline)
    assert not spent.use(("r", "a"), deadline)  # the old state is still spent
    assert not spent.use(("r", "c"), time.time() - 1)  # already past its deadline


async def test_a_state_expires_on_the_clock_its_spent_entry_uses(monkeypatch: pytest.MonkeyPatch):
    now = [time.time()]
    monkeypatch.setattr(_approval, "SpentStates", lambda: SpentStates(clock=lambda: now[0]))
    log: list[str] = []
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=Elicitor()) as client:
        state = await first_round(client, {"amount": 1})
        assert not (await replay(client, {"amount": 1}, state)).is_error
        now[0] += STATE_TTL_S + 1  # the spent entry's clock moved on; the seal is still valid
        with pytest.raises(MCPError, match=INVALID):
            await replay(client, {"amount": 1}, state)
    assert log == ["wire 1"]


async def test_an_eliciting_client_resumes_a_decided_server_side_approval():
    log: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(approval=DeferredApproval(store))
    server = mcp_server(make_registry(log), executor=executor, client_approval=True)
    async with Client(server) as plain:  # cannot elicit: the executor's port parks the call
        pending = await plain.call_tool("wire", {"amount": 3}, meta={RUN_ID_META: "r"})
    assert pending.meta is not None
    await store.decide(pending.meta[APPROVAL_ID_META], ApprovalDecision(approved=True))
    keys = {RUN_ID_META: "r", CALL_ID_META: pending.meta[CALL_ID_META]}
    elicitor = Elicitor()
    async with Client(server, elicitation_callback=elicitor) as eliciting:
        done = await eliciting.call_tool("wire", {"amount": 3}, meta=keys)
    assert not done.is_error and log == ["wire 3"]
    assert elicitor.asked == 0


def test_one_principal_at_its_quota_does_not_block_another():
    spent = SpentStates(per_principal=2)
    deadline = time.time() + STATE_TTL_S
    assert spent.use(("r", "a1"), deadline, "A") and spent.use(("r", "a2"), deadline, "A")
    with pytest.raises(MCPError, match="approval states busy"):
        spent.use(("r", "a3"), deadline, "A")
    assert spent.use(("r", "b1"), deadline, "B")
    assert not spent.use(("r", "a1"), deadline, "A")  # A's replay is still refused


def test_anonymous_callers_share_one_quota_bucket():
    spent = SpentStates(per_principal=1)
    deadline = time.time() + STATE_TTL_S
    assert spent.use(("r", "x"), deadline, None)
    with pytest.raises(MCPError, match="approval states busy"):
        spent.use(("r", "y"), deadline, None)


def test_the_quota_frees_up_as_a_principals_entries_expire():
    now = [0.0]
    spent = SpentStates(per_principal=1, clock=lambda: now[0])
    assert spent.use(("r", "a"), 10.0, "A")
    now[0] = 11.0  # A's entry expired: the sweep gives the quota back
    assert spent.use(("r", "b"), 20.0, "A")
    with pytest.raises(MCPError, match="approval states busy"):
        spent.use(("r", "c"), 20.0, "A")


async def test_the_quota_is_per_principal_over_mcp(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(_approval, "SpentStates", lambda: SpentStates(per_principal=1))
    who = ["alice"]
    log: list[str] = []
    server = mcp_server(
        make_registry(log),
        executor=Executor(),
        client_approval=True,
        principal=lambda ctx: who[0],
    )
    async with Client(server, elicitation_callback=Elicitor()) as client:
        used = await first_round(client, {"amount": 1})
        assert not (await replay(client, {"amount": 1}, used)).is_error
        fresh = await first_round(client, {"amount": 2})
        with caplog.at_level("WARNING"), pytest.raises(MCPError, match="approval states busy"):
            await replay(client, {"amount": 2}, fresh)
        assert "alice" not in caplog.text
        with pytest.raises(MCPError, match=INVALID):
            await replay(client, {"amount": 1}, used)
        who[0] = "bob"
        bobs = await first_round(client, {"amount": 3})
        assert not (await replay(client, {"amount": 3}, bobs)).is_error
    assert log == ["wire 1", "wire 3"]


class NotAStore:
    async def create(self, request: Any) -> Any: ...


class SharedStore:
    """Stands in for a store shared by every worker (not a `MemoryApprovalStore`)."""

    def __init__(self) -> None:
        self.inner = MemoryApprovalStore()

    async def create(self, request: Any) -> Any:
        return await self.inner.create(request)

    async def get(self, approval_id: str) -> Any:
        return await self.inner.get(approval_id)

    async def find(self, run_id: str, call_id: str) -> Any:
        return await self.inner.find(run_id, call_id)

    async def decide(self, approval_id: str, decision: Any) -> Any:
        return await self.inner.decide(approval_id, decision)

    async def consume(self, approval_id: str) -> bool:
        return await self.inner.consume(approval_id)

    async def list(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.list(*args, **kwargs)


class UncallableConsume(SharedStore):
    consume = None  # type: ignore[assignment]


@pytest.mark.parametrize("build", [mcp_server, streamable_http_app])
def test_approval_store_without_client_approval_is_a_config_error(build: Any):
    with pytest.raises(ConfigError, match="approval_store only applies with client_approval=True"):
        build(Registry(), approval_store=MemoryApprovalStore())


@pytest.mark.parametrize("build", [mcp_server, streamable_http_app])
@pytest.mark.parametrize("bad", [object(), NotAStore(), "store"])
def test_approval_store_must_be_an_approval_store(build: Any, bad: Any):
    with pytest.raises(ConfigError, match="approval_store must be an ApprovalStore"):
        build(Registry(), client_approval=True, approval_store=bad)


def test_approval_store_methods_must_be_callable():
    assert mcp_server(Registry(), client_approval=True, approval_store=SharedStore())
    with pytest.raises(ConfigError, match="approval_store must be an ApprovalStore"):
        mcp_server(Registry(), client_approval=True, approval_store=UncallableConsume())


@pytest.mark.parametrize("build", [mcp_server, streamable_http_app])
def test_keys_with_a_memory_store_warn_once_at_build(build: Any, caplog: pytest.LogCaptureFixture):
    keys = [secrets.token_hex(32)]
    with caplog.at_level("WARNING"):
        build(
            Registry(),
            name="p",
            client_approval=True,
            request_state_keys=keys,
            approval_store=MemoryApprovalStore(),
        )
    warned = [r for r in caplog.records if "MemoryApprovalStore" in r.getMessage()]
    assert len(warned) == 1 and warned[0].levelname == "WARNING"
    assert keys[0] not in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        build(
            Registry(),
            name="p",
            client_approval=True,
            request_state_keys=keys,
            approval_store=SharedStore(),
        )
    assert "MemoryApprovalStore" not in caplog.text
