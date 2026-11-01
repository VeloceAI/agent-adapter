"""import_mcp and mcp_client: remote tools become local ToolSpecs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
import mcp.types as t
from mcp import Client, StdioServerParameters
from mcp.server.lowlevel import Server
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    DuplicateTool,
    IntegrationError,
    ToolArgsInvalid,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp import import_mcp, mcp_client, mcp_server
from agent_adapter.integrations.mcp._scope import scoped_id
from agent_adapter.integrations.mcp.client import _HttpTransport

pytestmark = pytest.mark.contract


class Address(BaseModel):
    city: str
    pin: int | None = None


class Person(BaseModel):
    name: str
    addresses: list[Address]


def remote_registry() -> Registry:
    reg = Registry()

    @reg.tool(effects="read_only", idempotent=True)
    async def add(a: int, b: int = 1) -> int:
        """Add two integers."""
        return a + b

    @reg.tool(effects="destructive")
    async def register(person: Person, note: str = "") -> Person:
        """Register a person with nested addresses."""
        return person

    @reg.tool
    async def whoami(ctx: ToolContext) -> dict[str, str]:
        """Report the ids the server saw."""
        return {"run_id": ctx.run_id, "call_id": ctx.call_id}

    @reg.tool
    async def refuse(why: str) -> str:
        """Always refuses."""
        raise ToolError(f"no: {why}")

    @reg.tool(confirm=True)
    async def delete(item: str) -> str:
        """Delete (needs approval)."""
        return item

    return reg


async def test_import_keeps_remote_schemas_verbatim_and_maps_hints():
    async with Client(mcp_server(remote_registry())) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        local = Registry()
        specs = await import_mcp(local, client, prefix="r_")
    assert [s.name for s in specs] == ["r_add", "r_register", "r_whoami", "r_refuse", "r_delete"]
    register = local.get("r_register")
    assert register.json_schema() == listed["register"].input_schema
    nested = register.json_schema()["properties"]["person"]["properties"]["addresses"]
    assert nested["items"]["properties"]["city"] == {"type": "string"}
    assert register.result_schema() == listed["register"].output_schema
    assert (register.kind, register.tags, register.effects) == (
        "mcp",
        frozenset({"mcp"}),
        "destructive",
    )
    add = local.get("r_add")
    assert (add.effects, add.idempotent) == ("read_only", True)
    assert add.result_schema() == {"type": "integer"}  # the remote wrapper is undone
    assert local.get("r_whoami").effects is None
    assert local.get("r_delete").effects == "destructive"  # destructive_hint from confirm


async def test_imported_tools_call_the_remote_and_unwrap_results():
    async with Client(mcp_server(remote_registry())) as client:
        local = Registry()
        await import_mcp(local, client)
        assert await local.get("add").invoke({"a": 4}) == 5
        person = {"name": "Asha", "addresses": [{"city": "Pune"}]}
        assert await local.get("register").invoke({"person": person}) == {
            "name": "Asha",
            "addresses": [{"city": "Pune", "pin": None}],
        }
        with pytest.raises(ToolArgsInvalid):
            await local.get("add").invoke({"a": "four"})  # rejected before the wire


async def test_remote_errors_become_tool_errors_with_details():
    store = MemoryApprovalStore()
    remote = mcp_server(remote_registry(), executor=Executor(approval=DeferredApproval(store)))
    async with Client(remote) as client:
        local = Registry()
        await import_mcp(local, client)
        with pytest.raises(ToolError, match="no: rules") as refused:
            await local.get("refuse").invoke({"why": "rules"})
        assert refused.value.tool == "refuse"
        assert refused.value.details == {"remote_kind": "tool-error"}
        with pytest.raises(ToolError, match="approval required: apr_") as pending:
            await local.get("delete").invoke({"item": "x"})
    assert pending.value.details["approval_id"].startswith("apr_")
    assert pending.value.details["remote_kind"] == "approval-pending"


async def test_local_ids_are_sent_only_when_opted_in():
    async with Client(mcp_server(remote_registry())) as client:
        quiet, chatty = Registry(), Registry()
        await import_mcp(quiet, client, names=["whoami"])
        await import_mcp(chatty, client, names=["whoami"], send_ids=True)
        ctx = ToolContext(run_id="run_local")
        unsent = await Executor(quiet).run("whoami", {}, ctx, call_id="call_local")
        sent = await Executor(chatty).run("whoami", {}, ctx, call_id="call_local")
        direct = await chatty.get("whoami").invoke({})
    assert unsent["run_id"] != scoped_id("run_", None, "run_local")
    assert sent["run_id"] == scoped_id("run_", None, "run_local")
    assert sent["call_id"] != scoped_id("call_", None, "call_local")  # never a client choice
    assert direct["run_id"].startswith("run_") and direct["call_id"].startswith("call_")


async def test_a_remote_pending_approval_resumes_with_the_keys_it_named():
    store = MemoryApprovalStore()
    remote = mcp_server(remote_registry(), executor=Executor(approval=DeferredApproval(store)))
    async with Client(remote) as client:
        local = Registry()
        await import_mcp(local, client, names=["delete"])
        executor = Executor(local)
        ctx = executor.context(run_id="run_l")
        with pytest.raises(ToolError) as pending:
            await executor.run("delete", {"item": "x"}, ctx, call_id="call_l")
        approval_id = pending.value.details["approval_id"]
        await store.decide(approval_id, ApprovalDecision(approved=True))
        assert await executor.run("delete", {"item": "x"}, ctx, call_id="call_l") == "x"
        with pytest.raises(ToolError) as again:  # done: the next run asks again
            await executor.run("delete", {"item": "x"}, ctx, call_id="call_l")
    assert again.value.details["approval_id"] != approval_id


async def test_names_filter_and_missing_names():
    async with Client(mcp_server(remote_registry())) as client:
        local = Registry()
        assert [s.name for s in await import_mcp(local, client, names="add")] == ["add"]
        with pytest.raises(IntegrationError, match=r"no tools named \['nope'\]"):
            await import_mcp(Registry(), client, names=["add", "nope"])


async def test_policy_overrides_the_defaults():
    async with Client(mcp_server(remote_registry())) as client:
        local = Registry()
        specs = await import_mcp(
            local,
            client,
            names=["add"],
            kind="remote",
            tags={"x"},
            effects="additive",
            confirm=True,
        )
    (spec,) = specs
    assert (spec.kind, spec.tags, spec.effects, spec.confirm) == (
        "remote",
        frozenset({"x"}),
        "additive",
        True,
    )


async def test_import_is_all_or_nothing():
    local = Registry()

    @local.tool
    async def add(a: int) -> int:
        """Local add."""
        return a

    async with Client(mcp_server(remote_registry())) as client:
        with pytest.raises(DuplicateTool):
            await import_mcp(local, client)
    assert local.names() == ["add"]


def foreign_server() -> Server:
    """A server with names and shapes our own server never produces."""

    async def list_tools(ctx: Any, params: Any) -> t.ListToolsResult:
        return t.ListToolsResult(
            tools=[
                t.Tool(
                    name="files.read-text",
                    title="Read text",
                    input_schema={"type": "object", "properties": {"aa_ctx": {"type": "string"}}},
                ),
                t.Tool(name="blob", description="Mixed content", input_schema={"type": "object"}),
                t.Tool(name="nothing", description="No content", input_schema={"type": "object"}),
            ]
        )

    async def call_tool(ctx: Any, params: t.CallToolRequestParams) -> t.CallToolResult:
        if params.name == "files.read-text":
            text = f"ctx={(params.arguments or {}).get('aa_ctx')}"
            return t.CallToolResult(content=[t.TextContent(type="text", text=text)])
        if params.name == "blob":
            return t.CallToolResult(
                content=[
                    t.TextContent(type="text", text="see image"),
                    t.ImageContent(type="image", data="aGk=", mime_type="image/png"),
                ]
            )
        if params.name == "nothing":
            return t.CallToolResult(content=[])
        return t.CallToolResult(content=[], is_error=True)

    return Server("foreign", on_list_tools=list_tools, on_call_tool=call_tool)


async def test_foreign_names_content_and_context_clash():
    async with Client(foreign_server()) as client:
        local = Registry()
        await import_mcp(local, client)
        read = local.get("files_read_text")
        assert read.description == "Read text"
        assert read.context_param is None  # the schema owns `aa_ctx`
        assert await read.invoke({"aa_ctx": "mine"}) == "ctx=mine"
        assert await local.get("blob").invoke({}) == [
            {"type": "text", "text": "see image"},
            {"type": "image", "data": "aGk=", "mimeType": "image/png"},
        ]
        assert await local.get("nothing").invoke({}) is None


class PagedClient:
    """Duck-typed client serving tools/list in pages."""

    def __init__(self, cursors: list[str | None]) -> None:
        self.cursors = cursors
        self.asked: list[str | None] = []

    async def list_tools(self, *, cursor: str | None = None) -> t.ListToolsResult:
        self.asked.append(cursor)
        index = len(self.asked) - 1
        tool = t.Tool(name=f"t{index}", description="d", input_schema={"type": "object"})
        return t.ListToolsResult(tools=[tool], next_cursor=self.cursors[index])

    async def call_tool(self, name: str, arguments: Any, **_: Any) -> t.CallToolResult:
        return t.CallToolResult(content=[], is_error=True)


async def test_listing_follows_cursors_and_stops_on_a_repeat():
    client = PagedClient(["p1", "p2", None])
    specs = await import_mcp(Registry(), client)  # type: ignore[arg-type]
    assert [s.name for s in specs] == ["t0", "t1", "t2"]
    assert client.asked == [None, "p1", "p2"]
    with pytest.raises(IntegrationError, match="never ends"):
        await import_mcp(Registry(), PagedClient(["p1", "p1", "p1"]))  # type: ignore[arg-type]


async def test_remote_error_without_text_has_a_fallback_message():
    local = Registry()
    await import_mcp(local, PagedClient([None]))  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="MCP tool 't0' failed") as failed:
        await local.get("t0").invoke({})
    assert failed.value.details == {}


def test_mcp_client_targets():
    assert isinstance(mcp_client("https://tools.example/mcp", headers={"A": "b"}), Client)
    server = mcp_server(Registry())
    assert mcp_client(server).server is server
    params = StdioServerParameters(command="python", args=["-V"])
    assert mcp_client(params).server is params
    transport = _HttpTransport("http://x/mcp", {})
    assert mcp_client(transport).server is transport
    with pytest.raises(ToolUsageError, match="http"):
        mcp_client("tools.example/mcp")
    with pytest.raises(ToolUsageError, match="headers"):
        mcp_client(server, headers={"A": "b"})
    with pytest.raises(ToolUsageError, match="int"):
        mcp_client(42)


async def test_http_transport_exit_without_enter_is_a_no_op():
    assert await _HttpTransport("http://x/mcp", {}).__aexit__(None, None, None) is None


async def test_in_process_mcp_client_round_trip():
    async with mcp_client(mcp_server(remote_registry())) as client:
        local = Registry()
        await import_mcp(local, client, names=["add"])
        assert await local.get("add").invoke({"a": 1, "b": 2}) == 3


async def test_listing_gives_up_after_max_pages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_module, "MAX_PAGES", 2)
    with pytest.raises(IntegrationError, match="never ends"):
        await import_mcp(Registry(), PagedClient(["p1", "p2", None]))  # type: ignore[arg-type]


async def test_http_client_carries_headers_and_mcp_timeouts():
    async with client_module._new_http_client({"Authorization": "Bearer t"}) as http:
        assert http.headers["authorization"] == "Bearer t"
        assert (http.timeout.connect, http.timeout.read) == (30.0, 300.0)


async def test_http_transport_closes_its_client_when_connecting_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    import mcp.client.streamable_http as streamable

    opened: list[httpx2.AsyncClient] = []

    def factory(headers: Any) -> httpx2.AsyncClient:
        opened.append(httpx2.AsyncClient(headers=dict(headers)))
        return opened[-1]

    @asynccontextmanager
    async def refuse(url: str, *, http_client: Any) -> AsyncIterator[Any]:
        raise ConnectionError("down")
        yield  # pragma: no cover

    monkeypatch.setattr(client_module, "_new_http_client", factory)
    monkeypatch.setattr(streamable, "streamable_http_client", refuse)
    transport = _HttpTransport("http://x/mcp", {})
    with pytest.raises(ConnectionError):
        await transport.__aenter__()
    assert opened and opened[0].is_closed


async def test_remembered_resume_keys_are_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_module, "MAX_RESUMES", 1)
    remote = mcp_server(
        remote_registry(), executor=Executor(approval=DeferredApproval(MemoryApprovalStore()))
    )
    async with Client(remote) as client:
        local = Registry()
        (spec,) = await import_mcp(local, client, names=["delete"])
        executor = Executor(local)
        ctx = executor.context(run_id="run_l")
        for call_id in ("call_1", "call_2"):
            with pytest.raises(ToolError):
                await executor.run("delete", {"item": "x"}, ctx, call_id=call_id)
    assert list(spec.func._resume) == [("run_l", "call_2")]  # type: ignore[attr-defined]
