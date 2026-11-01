"""Single-use resumes: replays, concurrent resumes, swapped arguments, several workers."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
import mcp.types as t
from mcp import Client, MCPError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, ConfigError, ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import (
    APPROVAL_ID_META,
    CALL_ID_META,
    ERROR_KIND_META,
    RUN_ID_META,
    mcp_client,
    mcp_server,
    streamable_http_app,
)
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp._approval import APPROVAL_INPUT_KEY
from agent_adapter.integrations.mcp._resume import ResumeKeys, SpentStates, store_resumable
from agent_adapter.integrations.mcp._scope import scoped_id
from tests.fakes.approval import ScriptedApproval

pytestmark = pytest.mark.contract

URL = "http://127.0.0.1:8000/mcp"
YES = {APPROVAL_INPUT_KEY: t.ElicitResult(action="accept", content={"approved": True})}
INVALID = "Invalid or expired requestState"


def make_registry(log: list[str], gate: asyncio.Event | None = None) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True, once_per_run=True)
    async def pay(amount: int) -> str:
        """Pay once per run (needs approval)."""
        if gate is not None:
            await gate.wait()
        log.append(f"pay {amount}")
        return f"paid {amount}"

    @reg.tool
    async def leak(ctx: ToolContext) -> str:
        """Fail with the ids in the message."""
        raise ToolError(f"failed in {ctx.run_id} / {ctx.call_id}")

    @reg.tool
    async def whoami(ctx: ToolContext) -> str:
        """The run id."""
        return ctx.run_id

    return reg


async def approve(ctx: Any, params: Any) -> t.ElicitResult:
    return t.ElicitResult(action="accept", content={"approved": True})


async def first_round(client: Client, args: dict[str, Any]) -> str:
    first = await client.session.call_tool("pay", args, allow_input_required=True)
    assert isinstance(first, t.InputRequiredResult) and first.request_state is not None
    return first.request_state


async def replay(client: Client, args: dict[str, Any], state: str) -> Any:
    return await client.session.call_tool("pay", args, input_responses=YES, request_state=state)


async def test_a_client_approval_state_runs_once_in_process():
    log: list[str] = []
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=approve) as client:
        state = await first_round(client, {"amount": 5})
        done = await replay(client, {"amount": 5}, state)
        assert isinstance(done, t.CallToolResult) and not done.is_error
        for _ in range(5):
            with pytest.raises(MCPError, match=INVALID):
                await replay(client, {"amount": 5}, state)
    assert log == ["pay 5"]


@asynccontextmanager
async def serving(*apps: Any) -> AsyncIterator[None]:
    """Run the apps' lifespans; URLs with host `w<i>` reach app `i` (no sockets)."""
    routes = {f"w{i}": app for i, app in enumerate(apps)}

    async def route(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            host = dict(scope["headers"]).get(b"x-worker", b"w0").decode()
            await routes[host](scope, receive, send)

    async with contextlib_stack(apps):
        yield route  # type: ignore[misc]


@asynccontextmanager
async def contextlib_stack(apps: tuple[Any, ...]) -> AsyncIterator[None]:
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        for app in apps:
            await stack.enter_async_context(app.router.lifespan_context(app))
        yield


def worker_client(router: Any, worker: str, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> Client:
    def asgi_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=router), headers={**headers, "x-worker": worker}
        )

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    return Client(mcp_client(URL).server, **kw)


async def test_a_replayed_state_is_refused_on_another_worker_sharing_the_store(
    monkeypatch: pytest.MonkeyPatch,
):
    log: list[str] = []
    store = MemoryApprovalStore()
    keys = [secrets.token_hex(32)]
    apps = [
        streamable_http_app(
            make_registry(log),
            executor=Executor(approval=DeferredApproval(store)),
            name="tax",
            client_approval=True,
            request_state_keys=keys,
            allow_anonymous=True,
        )
        for _ in range(2)
    ]
    async with serving(*apps) as router:
        async with worker_client(router, "w0", monkeypatch, elicitation_callback=approve) as one:
            state = await first_round(one, {"amount": 7})
            done = await replay(one, {"amount": 7}, state)
        assert isinstance(done, t.CallToolResult) and not done.is_error
        async with worker_client(router, "w1", monkeypatch, elicitation_callback=approve) as two:
            with pytest.raises(MCPError, match=INVALID):
                await replay(two, {"amount": 7}, state)
    assert log == ["pay 7"]
    (record,) = await store.list()
    assert record.consumed and record.status == "approved"
    assert record.request.hint == "decided by the MCP client"


class OtherOpenRecordStore(MemoryApprovalStore):
    """`create` answers with another open record, as a racing worker's would be."""

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        other = request.model_copy(update={"approval_id": "apr_racer"})
        return ApprovalRecord(request=other, status="pending")


class UsedStore(MemoryApprovalStore):
    async def consume(self, approval_id: str) -> bool:
        return False


class BrokenStore(MemoryApprovalStore):
    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        raise OSError("database down")

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        raise OSError("database down")


@pytest.mark.parametrize(
    ("store", "message"),
    [
        (OtherOpenRecordStore(), INVALID),
        (UsedStore(), INVALID),
        (BrokenStore(), "approval store unavailable"),
    ],
)
async def test_the_shared_store_record_fails_closed(store: MemoryApprovalStore, message: str):
    log: list[str] = []
    executor = Executor(approval=DeferredApproval(store))
    server = mcp_server(make_registry(log), executor=executor, client_approval=True)
    async with Client(server, elicitation_callback=approve) as client:
        state = await first_round(client, {"amount": 1})
        with pytest.raises(MCPError, match=message):
            await replay(client, {"amount": 1}, state)
    assert log == []


def pending_server(log: list[str], store: MemoryApprovalStore, gate: Any = None) -> Any:
    return mcp_server(make_registry(log, gate), executor=Executor(approval=DeferredApproval(store)))


async def call(client: Client, args: dict[str, Any], **meta: str) -> t.CallToolResult:
    return await client.call_tool("pay", args, meta=meta or None)


async def pending_keys(server: Any, args: dict[str, Any]) -> tuple[dict[str, str], str]:
    async with Client(server) as client:
        pending = await call(client, args, **{RUN_ID_META: "r"})
    assert pending.meta is not None
    keys = {RUN_ID_META: "r", CALL_ID_META: pending.meta[CALL_ID_META]}
    return keys, pending.meta[APPROVAL_ID_META]


async def test_a_resume_works_on_any_worker_sharing_the_store():
    log: list[str] = []
    store = MemoryApprovalStore()
    one, two = pending_server(log, store), pending_server(log, store)
    keys, approval_id = await pending_keys(one, {"amount": 3})
    await store.decide(approval_id, ApprovalDecision(approved=True))
    async with Client(two) as client:
        done = await call(client, {"amount": 3}, **keys)
        again = await call(client, {"amount": 3}, **keys)  # spent: a new call in run "r"
    assert not done.is_error and log == ["pay 3"]
    assert again.meta is not None and again.meta[ERROR_KIND_META] == "run-limit-exceeded"


async def test_a_store_failure_during_the_resume_check_fails_closed():
    log: list[str] = []
    server = pending_server(log, BrokenStore())
    async with Client(server) as client:
        with pytest.raises(MCPError, match="approval store unavailable"):
            await call(client, {"amount": 1}, **{RUN_ID_META: "r", CALL_ID_META: "k"})
    assert log == []


async def test_eight_concurrent_resumes_run_once_and_orphan_nothing():
    log: list[str] = []
    store = MemoryApprovalStore()
    gate = asyncio.Event()
    server = pending_server(log, store, gate)
    keys, approval_id = await pending_keys(server, {"amount": 9})
    await store.decide(approval_id, ApprovalDecision(approved=True))

    async def resume() -> t.CallToolResult | MCPError:
        async with Client(server) as client:
            try:
                return await call(client, {"amount": 9}, **keys)
            except MCPError as exc:  # inside the block: the client's task group would wrap it
                return exc

    tasks = [asyncio.ensure_future(resume()) for _ in range(8)]
    async with asyncio.timeout(10):
        while sum(task.done() for task in tasks) < 7:
            await asyncio.sleep(0.01)
    gate.set()
    outcomes = await asyncio.gather(*tasks)
    refused = [o for o in outcomes if isinstance(o, MCPError)]
    done = [o for o in outcomes if isinstance(o, t.CallToolResult)]
    assert len(refused) == 7 and all("already being resumed" in str(o) for o in refused)
    assert len(done) == 1 and not done[0].is_error
    assert log == ["pay 9"]
    assert [r for r in await store.list() if not r.consumed] == []


async def test_a_resume_with_other_arguments_is_refused_and_keeps_the_approval():
    log: list[str] = []
    store = MemoryApprovalStore()
    server = pending_server(log, store)
    keys, approval_id = await pending_keys(server, {"amount": 2})
    await store.decide(approval_id, ApprovalDecision(approved=True))
    async with Client(server) as client:
        with pytest.raises(MCPError, match="resume arguments differ"):
            await call(client, {"amount": 2000}, **keys)
        done = await call(client, {"amount": 2}, **keys)
    assert not done.is_error and log == ["pay 2"]


async def test_the_fallback_keys_refuse_other_arguments_without_spending_the_key():
    log: list[str] = []
    port = ScriptedApproval([ApprovalPending("apr_x"), ApprovalDecision(approved=True)])
    server = mcp_server(make_registry(log), executor=Executor(approval=port))
    async with Client(server) as client:
        pending = await call(client, {"amount": 4}, **{RUN_ID_META: "r"})
        assert pending.meta is not None
        keys = {RUN_ID_META: "r", CALL_ID_META: pending.meta[CALL_ID_META]}
        with pytest.raises(MCPError, match="resume arguments differ"):
            await call(client, {"amount": 40}, **keys)
        done = await call(client, {"amount": 4}, **keys)
    assert not done.is_error and log == ["pay 4"]


async def test_derived_ids_never_reach_the_error_text():
    async with Client(mcp_server(make_registry([]))) as client:
        result = await client.call_tool("leak", {}, meta={RUN_ID_META: "my-run"})
    text = result.content[0].text
    assert "my-run / call_" in text
    assert scoped_id("run_", None, "my-run") not in text


async def test_sdk_direct_http_without_a_principal_ignores_client_keys():
    for allow, honoured in ((None, False), (True, True)):
        app = mcp_server(make_registry([]), allow_anonymous=allow).streamable_http_app(
            json_response=True, stateless_http=True
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as http,
        ):
            from mcp.client.streamable_http import streamable_http_client

            @asynccontextmanager
            async def transport() -> AsyncIterator[Any]:
                async with streamable_http_client(URL, http_client=http) as streams:
                    yield streams

            async with Client(transport()) as client:
                seen = await client.call_tool("whoami", {}, meta={RUN_ID_META: "r"})
        assert (seen.structured_content["result"] == scoped_id("run_", None, "r")) is honoured


async def test_in_process_anonymous_keys_stay_honoured_by_default():
    async with Client(mcp_server(make_registry([]))) as client:
        seen = await client.call_tool("whoami", {}, meta={RUN_ID_META: "r"})
    assert seen.structured_content == {"result": scoped_id("run_", None, "r")}


def test_request_state_keys_need_a_unique_name():
    keys = [secrets.token_hex(32)]
    with pytest.raises(ConfigError, match="unique name"):
        mcp_server(Registry(), client_approval=True, request_state_keys=keys)
    with pytest.raises(ConfigError, match="unique name"):
        streamable_http_app(Registry(), client_approval=True, request_state_keys=keys)
    assert mcp_server(
        Registry(),
        name="tax",
        client_approval=True,
        request_state_keys=keys,
        approval_store=MemoryApprovalStore(),
    )


def test_spent_states_expire_on_their_deadline_and_never_evict(
    caplog: pytest.LogCaptureFixture,
):
    now = [0.0]
    spent = SpentStates(max_entries=2, clock=lambda: now[0])
    assert spent.use(("r", "a"), 10.0) and not spent.use(("r", "a"), 10.0)
    now[0] = 11.0
    assert spent.use(("r", "a"), 21.0)  # expired: a new state may use the pair again
    assert spent.use(("r", "b"), 21.0)
    with caplog.at_level(logging.WARNING), pytest.raises(MCPError, match="busy"):
        spent.use(("r", "c"), 21.0)
    assert "approval-state set full" in caplog.text
    now[0] = 22.0  # both expired: the full set sweeps them and takes the new state
    assert spent.use(("r", "c"), 30.0)


def test_error_kind_constant_is_unchanged():
    assert ERROR_KIND_META == "agent-adapter/error_kind"


class GraphBubbleUp(Exception):
    """Control flow by name."""


class InterruptingStore(MemoryApprovalStore):
    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        raise GraphBubbleUp("pause")


async def test_control_flow_from_the_store_propagates():
    with pytest.raises(GraphBubbleUp):
        await store_resumable(InterruptingStore(), "r", "c", "pay")


def test_claiming_one_key_keeps_the_principals_others():
    keys = ResumeKeys()
    keys.issue("p", "r", "c1", "t", "d")
    keys.issue("p", "r", "c2", "t", "d")
    assert keys.claim("p", "r", "c1", "t", "d") == "claimed"
    assert keys.claim("p", "r", "c2", "t", "d") == "claimed"
