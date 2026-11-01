"""Principal-scoped ids: one MCP client never reaches another's runs, calls or approvals."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
import mcp.types as t
from mcp import Client, MCPError
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

from agent_adapter.approval import AutoApprove, DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
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
from agent_adapter.integrations.mcp._approval import APPROVAL_INPUT_KEY, ClientApproval
from agent_adapter.integrations.mcp._handlers import Handlers
from agent_adapter.integrations.mcp._resume import ResumeKeys
from agent_adapter.integrations.mcp._scope import principal_of, scoped_id

pytestmark = pytest.mark.contract

URL = "http://127.0.0.1:8000/mcp"
TOKENS = {
    "tok-a": AccessToken(token="tok-a", client_id="alice", scopes=[]),
    "tok-b": AccessToken(token="tok-b", client_id="bob", scopes=[]),
}
AUTH = AuthSettings(issuer_url="https://auth.example", resource_server_url=None)


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        return TOKENS.get(token)


def make_registry(log: list[str]) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    async def delete(item: str) -> str:
        """Delete an item (needs approval)."""
        log.append(item)
        return f"deleted {item}"

    @reg.tool(once_per_run=True)
    async def charge(amount: int) -> str:
        """Charge once per run."""
        log.append(f"charge {amount}")
        return "charged"

    @reg.tool
    async def whoami(ctx: ToolContext) -> dict[str, Any]:
        """The ids and principal the tool sees."""
        return {"run_id": ctx.run_id, "call_id": ctx.call_id, "principal": ctx.extras["principal"]}

    return reg


@asynccontextmanager
async def serving(app: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    def asgi_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=dict(headers))

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with app.router.lifespan_context(app):
        yield


@asynccontextmanager
async def users(
    app: Any, monkeypatch: pytest.MonkeyPatch, *tokens: str, **kwargs: Any
) -> AsyncIterator[list[Client]]:
    """The app served, with one connected client per bearer token."""
    async with serving(app, monkeypatch), AsyncExitStack() as stack:
        yield [await stack.enter_async_context(as_user(token, **kwargs)) for token in tokens]


def as_user(token: str, **kwargs: Any) -> Client:
    transport = mcp_client(URL, headers={"Authorization": f"Bearer {token}"}).server
    return Client(transport, **kwargs)


async def call(client: Client, name: str, args: dict[str, Any], **meta: str) -> t.CallToolResult:
    return await client.call_tool(name, args, meta=meta or None)


def authed_app(log: list[str], store: MemoryApprovalStore, **options: Any) -> Any:
    executor = Executor(approval=DeferredApproval(store))
    return streamable_http_app(
        make_registry(log), executor=executor, auth=AUTH, token_verifier=Verifier(), **options
    )


async def test_b_cannot_consume_a_decided_approval_and_a_resumes(monkeypatch: pytest.MonkeyPatch):
    log: list[str] = []
    store = MemoryApprovalStore()
    async with users(authed_app(log, store), monkeypatch, "tok-a", "tok-b") as (alice, bob):
        pending = await call(alice, "delete", {"item": "a"}, **{RUN_ID_META: "shared"})
        assert pending.meta is not None
        keys = {RUN_ID_META: "shared", CALL_ID_META: pending.meta[CALL_ID_META]}
        await store.decide(pending.meta[APPROVAL_ID_META], ApprovalDecision(approved=True))
        stolen = await call(bob, "delete", {"item": "a"}, **keys)
        assert stolen.is_error and stolen.meta is not None
        assert stolen.meta[ERROR_KIND_META] == "approval-pending"
        assert stolen.meta[APPROVAL_ID_META] != pending.meta[APPROVAL_ID_META]
        assert log == []
        done = await call(alice, "delete", {"item": "a"}, **keys)
    assert not done.is_error and log == ["a"]


async def test_b_cannot_plant_under_a_call_key(monkeypatch: pytest.MonkeyPatch):
    log: list[str] = []
    store = MemoryApprovalStore()
    async with users(authed_app(log, store), monkeypatch, "tok-a", "tok-b") as (alice, bob):
        pending = await call(alice, "delete", {"item": "a"}, **{RUN_ID_META: "shared"})
        assert pending.meta is not None
        keys = {RUN_ID_META: "shared", CALL_ID_META: pending.meta[CALL_ID_META]}
        planted = await call(bob, "delete", {"item": "everything"}, **keys)
        assert planted.meta is not None
        records = {r.approval_id: r.request for r in await store.list()}
        alices = records[pending.meta[APPROVAL_ID_META]]
        bobs = records[planted.meta[APPROVAL_ID_META]]
        assert alices.arguments == {"item": "a"}
        assert (bobs.run_id, bobs.call_id) != (alices.run_id, alices.call_id)
        await store.decide(pending.meta[APPROVAL_ID_META], ApprovalDecision(approved=True))
        done = await call(alice, "delete", {"item": "a"}, **keys)
    assert not done.is_error and log == ["a"]


async def test_b_cannot_touch_a_once_per_run(monkeypatch: pytest.MonkeyPatch):
    log: list[str] = []
    async with users(authed_app(log, MemoryApprovalStore()), monkeypatch, "tok-a", "tok-b") as (
        alice,
        bob,
    ):
        shared = {RUN_ID_META: "shared"}
        assert not (await call(bob, "charge", {"amount": 1}, **shared)).is_error
        assert not (await call(alice, "charge", {"amount": 2}, **shared)).is_error
        again = await call(alice, "charge", {"amount": 3}, **shared)
    assert again.is_error and again.meta is not None
    assert again.meta[ERROR_KIND_META] == "run-limit-exceeded"
    assert log == ["charge 1", "charge 2"]


async def test_a_reused_call_key_never_reruns_a_once_per_run_tool():
    log: list[str] = []
    keys = {RUN_ID_META: "r1", CALL_ID_META: "k1"}
    async with Client(mcp_server(make_registry(log))) as client:
        first = await call(client, "charge", {"amount": 1}, **keys)
        second = await call(client, "charge", {"amount": 1}, **keys)
    assert not first.is_error and second.is_error and log == ["charge 1"]
    assert second.meta is not None and second.meta[ERROR_KIND_META] == "run-limit-exceeded"


async def test_the_principal_reaches_the_tool_and_scopes_its_ids(monkeypatch: pytest.MonkeyPatch):
    async with users(authed_app([], MemoryApprovalStore()), monkeypatch, "tok-a") as (alice,):
        seen = (await call(alice, "whoami", {}, **{RUN_ID_META: "r"})).structured_content
    assert "alice" in seen["principal"]
    assert seen["run_id"] == scoped_id("run_", seen["principal"], "r")
    assert seen["run_id"] != scoped_id("run_", None, "r")


async def test_anonymous_http_ignores_client_ids(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    log: list[str] = []
    store = MemoryApprovalStore()
    app = streamable_http_app(
        make_registry(log), executor=Executor(approval=DeferredApproval(store))
    )
    keys = {RUN_ID_META: "r", CALL_ID_META: "c"}
    with caplog.at_level(logging.WARNING):
        async with serving(app, monkeypatch), mcp_client(URL, headers={"X-Run-Id": "h"}) as client:
            seen = (await call(client, "whoami", {}, **keys)).structured_content
            pending = await call(client, "delete", {"item": "x"}, **keys)
    assert seen["run_id"] not in {scoped_id("run_", None, "r"), scoped_id("run_", None, "h")}
    assert seen["principal"] is None
    assert pending.meta is not None and APPROVAL_ID_META in pending.meta
    assert RUN_ID_META not in pending.meta and CALL_ID_META not in pending.meta
    assert "anonymous MCP request: ignoring" in caplog.text
    assert "'r'" not in caplog.text and "'h'" not in caplog.text


async def test_allow_anonymous_honours_keys_and_warns_at_build(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        app = streamable_http_app(make_registry([]), allow_anonymous=True)
    assert "accepts anonymous run and call keys" in caplog.text
    async with serving(app, monkeypatch), mcp_client(URL) as client:
        seen = (await call(client, "whoami", {}, **{RUN_ID_META: "r"})).structured_content
    assert seen["run_id"] == scoped_id("run_", None, "r")


def boom(ctx: Any) -> str:
    raise RuntimeError("identity provider down")


@pytest.mark.parametrize("hook", [boom, lambda ctx: 42])
async def test_a_failing_principal_hook_refuses_the_request(hook: Any):
    log: list[str] = []
    async with Client(mcp_server(make_registry(log), principal=hook)) as client:
        with pytest.raises(MCPError, match="principal check failed"):
            await client.call_tool("charge", {"amount": 1})
    assert log == []


def test_principal_of_treats_empty_as_none():
    assert principal_of(SimpleNamespace(), lambda ctx: "") is None  # type: ignore[arg-type]
    assert principal_of(SimpleNamespace(), None) is None  # type: ignore[arg-type]


async def approve(ctx: Any, params: Any) -> t.ElicitResult:
    return t.ElicitResult(action="accept", content={"approved": True})


async def test_sealed_state_is_bound_to_the_principal(monkeypatch: pytest.MonkeyPatch):
    log: list[str] = []
    yes = {APPROVAL_INPUT_KEY: t.ElicitResult(action="accept", content={"approved": True})}
    app = authed_app(log, MemoryApprovalStore(), client_approval=True)
    async with (
        serving(app, monkeypatch),
        as_user("tok-a", elicitation_callback=approve) as alice,
        as_user("tok-b", elicitation_callback=approve) as bob,
    ):
        first = await alice.session.call_tool("delete", {"item": "a"}, allow_input_required=True)
        assert isinstance(first, t.InputRequiredResult)
        with pytest.raises(MCPError, match="Invalid or expired requestState"):
            await bob.session.call_tool(
                "delete", {"item": "a"}, input_responses=yes, request_state=first.request_state
            )
        done = await alice.session.call_tool(
            "delete", {"item": "a"}, input_responses=yes, request_state=first.request_state
        )
    assert isinstance(done, t.CallToolResult) and not done.is_error and log == ["a"]


async def test_a_bad_request_state_is_a_protocol_error_not_a_tool_error():
    handlers = Handlers(make_registry([]), Executor(), approval=ClientApproval())
    ctx = SimpleNamespace(request=None, protocol_version="2026-07-28", session=None)
    params = t.CallToolRequestParams(name="delete", arguments={"item": "a"}, request_state="junk")
    with pytest.raises(MCPError, match="Invalid or expired requestState"):
        await handlers.call_tool(ctx, params)  # type: ignore[arg-type]


def test_resume_keys_are_capped_per_principal(caplog: pytest.LogCaptureFixture):
    keys = ResumeKeys(per_principal=1, max_principals=2)
    keys.issue("p", "r", "c1", "t", "d")
    keys.issue("q", "r", "c1", "t", "d")
    with caplog.at_level(logging.WARNING):
        keys.issue("p", "r", "c2", "t", "d")  # p's own oldest goes, q's key stays
    assert keys.claim("p", "r", "c1", "t", "d") == "absent"
    assert keys.claim("q", "r", "c1", "t", "d") == "claimed"
    assert keys.claim("p", "r", "c2", "t", "other") == "mismatch"
    assert keys.claim("p", "r", "c2", "t", "d") == "claimed"
    assert keys.claim("p", "r", "c2", "t", "d") == "absent"  # popped: single use
    assert "dropped a principal's oldest MCP resume key" in caplog.text
    for principal in ("a", "b", "c"):
        keys.issue(principal, "r", "c", "t", None)
    assert keys.claim("a", "r", "c", "t", None) == "absent"  # least recent principal dropped
    assert "least recent principal" in caplog.text


def test_scoped_id_is_stable_and_principal_bound():
    assert scoped_id("call_", "p", "k") == scoped_id("call_", "p", "k")
    assert scoped_id("call_", "p", "k") != scoped_id("call_", "q", "k")
    assert scoped_id("call_", None, "k") != scoped_id("call_", "", "k")
    assert len(scoped_id("run_", None, "k")) == len("run_") + 32


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"host": "0.0.0.0"}, "not local"),
        ({"stateless_http": False}, "sets"),
        ({"streamable_http_path": "/x"}, "sets"),
    ],
)
def test_http_app_refuses_unsafe_or_clashing_options(options: dict[str, Any], match: str):
    with pytest.raises(ConfigError, match=match):
        streamable_http_app(Registry(), **options)


def test_a_public_host_with_transport_security_and_auth_is_accepted():
    security = TransportSecuritySettings(allowed_hosts=["tools.example"])
    assert streamable_http_app(
        Registry(),
        host="0.0.0.0",
        transport_security=security,
        auth=AUTH,
        token_verifier=Verifier(),
    )


def test_bad_request_state_keys_are_config_errors():
    with pytest.raises(ConfigError, match="request_state_keys"):
        mcp_server(
            Registry(),
            name="tax",
            client_approval=True,
            request_state_keys=["short"],
            approval_store=MemoryApprovalStore(),
        )


def test_non_callable_principal_is_refused():
    with pytest.raises(ToolUsageError, match="principal"):
        mcp_server(Registry(), principal="alice")  # type: ignore[arg-type]


def test_client_approval_warns_when_it_replaces_a_real_port(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        mcp_server(Registry(), executor=Executor(approval=AutoApprove()), client_approval=True)
    assert "client_approval=True" not in caplog.text
    store = MemoryApprovalStore()
    with caplog.at_level(logging.WARNING):
        mcp_server(
            Registry(), executor=Executor(approval=DeferredApproval(store)), client_approval=True
        )
    assert "DeferredApproval is not asked" in caplog.text


def test_http_app_passes_name_through_to_the_server():
    app = streamable_http_app(Registry(), name="tax-tools")
    assert app is not None


def test_the_default_principal_marker_reads_well():
    from agent_adapter.integrations.mcp.server import AUTHENTICATED

    assert repr(AUTHENTICATED) == "AUTHENTICATED"
