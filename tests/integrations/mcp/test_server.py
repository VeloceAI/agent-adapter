"""mcp_server: listing, calling, errors, pending approvals and ids (in-process client)."""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

import mcp.types as t
from mcp import Client, MCPError
from pydantic import BaseModel

from agent_adapter._version import __version__
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import MissingDependency, ToolError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor, defaults
from agent_adapter.integrations.mcp import (
    APPROVAL_ID_META,
    CALL_ID_META,
    ERROR_KIND_META,
    INTEGRATION,
    RUN_ID_META,
    mcp_server,
    mcp_tool,
)
from agent_adapter.integrations.mcp._handlers import Handlers, raw_context
from agent_adapter.integrations.mcp._scope import scoped_id
from tests.integrations.kit import FakeRunner

pytestmark = pytest.mark.contract


class Point(BaseModel):
    x: int
    y: int


def make_registry() -> Registry:
    reg = Registry()

    @reg.tool(effects="read_only", idempotent=True)
    async def add(a: int, b: int = 1) -> int:
        """Add two integers."""
        return a + b

    @reg.tool
    async def point(x: int) -> Point:
        """Make a point."""
        return Point(x=x, y=-x)

    @reg.tool
    async def whoami(ctx: ToolContext) -> dict[str, str]:
        """Report the ids the tool sees."""
        return {"run_id": ctx.run_id, "call_id": ctx.call_id}

    @reg.tool
    async def refuse(why: str) -> str:
        """Fail with a model-visible error."""
        raise ToolError(f"refused: {why}")

    @reg.tool
    async def crash() -> str:
        """Fail with a foreign error."""
        raise RuntimeError("secret internals")

    @reg.tool(confirm=True)
    async def delete(item: str) -> str:
        """Delete an item (needs approval)."""
        return f"deleted {item}"

    @reg.tool(targets="openai")
    async def only_openai() -> str:
        """Not exported to MCP."""
        return "no"

    return reg


async def call(server: Any, name: str, args: dict[str, Any], **meta: str) -> t.CallToolResult:
    async with Client(server) as client:
        return await client.call_tool(name, args, meta=meta or None)


async def test_list_tools_exports_the_mcp_target_with_schemas_and_hints():
    reg = make_registry()
    async with Client(mcp_server(reg)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"add", "point", "whoami", "refuse", "crash", "delete"}
    add = tools["add"]
    assert add.description == "Add two integers."
    assert add.input_schema["properties"]["b"] == {"type": "integer", "default": 1}
    assert add.output_schema == {
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
    }
    assert add.annotations == t.ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True
    )
    assert tools["delete"].annotations == t.ToolAnnotations(destructive_hint=True)
    assert tools["whoami"].annotations is None
    assert tools["point"].output_schema == reg.get("point").result_schema()
    assert tools["add"] == mcp_tool(reg.get("add"))


async def test_tools_added_later_are_listed():
    reg = make_registry()
    server = mcp_server(reg)

    @reg.tool
    async def late() -> str:
        """Added after the server."""
        return "late"

    assert "late" in {tool.name for tool in (await _list(server))}


async def _list(server: Any) -> list[t.Tool]:
    async with Client(server) as client:
        return list((await client.list_tools()).tools)


async def test_server_info_and_instructions():
    server = mcp_server(Registry(), name="tax-tools", instructions="Use for GST.")
    async with Client(server) as client:
        assert client.server_info.name == "tax-tools"
        assert client.server_info.version == __version__
        discovered = await client.session.discover()
    assert discovered.instructions == "Use for GST."


async def test_call_returns_structured_and_text_content():
    server = mcp_server(make_registry())
    result = await call(server, "add", {"a": 2})
    assert not result.is_error
    assert result.structured_content == {"result": 3}
    assert [block.text for block in result.content] == ["3"]
    result = await call(server, "point", {"x": 4})
    assert result.structured_content == {"x": 4, "y": -4}
    assert result.content[0].text == '{"x": 4, "y": -4}'


@pytest.mark.parametrize("name", ["missing", "only_openai"])
async def test_unknown_or_unexported_tool_is_a_protocol_error(name: str):
    async with Client(mcp_server(make_registry())) as client:
        with pytest.raises(MCPError, match=f"Unknown tool: {name}"):
            await client.call_tool(name, {})


async def test_invalid_args_are_tool_errors_with_a_kind():
    result = await call(mcp_server(make_registry()), "add", {"a": "two"})
    assert result.is_error
    assert result.meta is not None and result.meta[ERROR_KIND_META] == "tool-args-invalid"
    assert "two" not in result.content[0].text


async def test_tool_error_message_is_model_visible():
    result = await call(mcp_server(make_registry()), "refuse", {"why": "policy"})
    assert result.is_error and result.content[0].text == "refused: policy"
    assert result.meta is not None and result.meta[ERROR_KIND_META] == "tool-error"


async def test_foreign_errors_never_leak_their_message(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR, logger="agent_adapter.integrations.mcp.server"):
        raw = await call(mcp_server(make_registry(), executor=FakeRunner()), "crash", {})
    assert raw.is_error and raw.content[0].text == "RuntimeError"
    assert "raised over MCP" in caplog.text and "(RuntimeError)" in caplog.text
    assert "secret" not in caplog.text  # the exception text stays at DEBUG
    wrapped = await call(mcp_server(make_registry(), executor=Executor()), "crash", {})
    assert wrapped.is_error and "secret" not in wrapped.content[0].text
    assert wrapped.meta is not None and wrapped.meta[ERROR_KIND_META] == "tool-failed"


async def test_run_keys_are_scoped_and_call_keys_minted_unless_resuming():
    result = await call(
        mcp_server(make_registry()), "whoami", {}, **{RUN_ID_META: "run_x", CALL_ID_META: "c_1"}
    )
    seen = result.structured_content
    assert seen["run_id"] == scoped_id("run_", None, "run_x")
    assert seen["call_id"].startswith("call_")
    assert seen["call_id"] != scoped_id("call_", None, "c_1")


@pytest.mark.parametrize("bad", ["", "x" * 257, "tab\there"])
async def test_invalid_meta_ids_are_ignored(bad: str, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        result = await call(mcp_server(make_registry()), "whoami", {}, **{CALL_ID_META: bad})
    assert result.structured_content["call_id"].startswith("call_")
    assert "ignoring invalid agent-adapter/call_id" in caplog.text


def test_raw_context_reads_models_and_mappings():
    params = t.CallToolRequestParams(name="x", _meta={RUN_ID_META: "r", CALL_ID_META: "c"})
    ctx = raw_context(params)
    assert (ctx.run_id, ctx.call_id, ctx.extras["mcp"]) == ("r", "c", params)
    assert raw_context({"_meta": {CALL_ID_META: "c2"}}).call_id == "c2"
    assert raw_context({"meta": {RUN_ID_META: 5}}).run_id == ""
    assert raw_context(object()).call_id == ""
    assert INTEGRATION.bind_context(params).call_id == "c"


async def test_confirm_tool_without_a_port_is_denied_by_the_default_executor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(defaults, "_default", None)
    result = await call(mcp_server(make_registry()), "delete", {"item": "a"})
    assert result.is_error
    assert result.meta is not None and result.meta[ERROR_KIND_META] == "approval-denied"


async def test_pending_approval_names_the_id_and_resumes_after_the_decision():
    store = MemoryApprovalStore()
    server = mcp_server(make_registry(), executor=Executor(approval=DeferredApproval(store)))
    ids = {RUN_ID_META: "run_mcp", CALL_ID_META: "call_mcp"}
    pending = await call(server, "delete", {"item": "a"}, **ids)
    assert pending.is_error
    assert pending.meta is not None
    approval_id = pending.meta[APPROVAL_ID_META]
    assert pending.content[0].text == f"approval required: {approval_id}"
    assert pending.meta[ERROR_KIND_META] == "approval-pending"
    assert pending.meta[RUN_ID_META] == "run_mcp"  # keys, never the derived ids
    assert pending.meta[CALL_ID_META] != "call_mcp"  # not a resume: a fresh call key
    keys = {RUN_ID_META: "run_mcp", CALL_ID_META: pending.meta[CALL_ID_META]}
    (record,) = await store.list()
    assert record.request.call_id == scoped_id("call_", None, keys[CALL_ID_META])
    again = await call(server, "delete", {"item": "a"}, **keys)
    assert again.meta is not None and again.meta[APPROVAL_ID_META] == approval_id
    await store.decide(approval_id, ApprovalDecision(approved=True, decided_by="ops"))
    done = await call(server, "delete", {"item": "a"}, **keys)
    assert not done.is_error and done.structured_content == {"result": "deleted a"}
    spent = await call(server, "delete", {"item": "a"}, **keys)  # the resume key is spent
    assert spent.meta is not None and spent.meta[APPROVAL_ID_META] != approval_id


async def test_pending_without_ids_returns_the_minted_ids_for_the_resume():
    store = MemoryApprovalStore()
    server = mcp_server(make_registry(), executor=Executor(approval=DeferredApproval(store)))
    pending = await call(server, "delete", {"item": "b"})
    assert pending.meta is not None
    run_id, call_id = pending.meta[RUN_ID_META], pending.meta[CALL_ID_META]
    assert run_id.startswith("run_") and call_id.startswith("call_")
    await store.decide(pending.meta[APPROVAL_ID_META], ApprovalDecision(approved=False))
    denied = await call(
        server, "delete", {"item": "b"}, **{RUN_ID_META: run_id, CALL_ID_META: call_id}
    )
    assert denied.is_error and denied.meta is not None
    assert denied.meta[ERROR_KIND_META] == "approval-denied"


class GraphBubbleUp(Exception):
    """Named like LangGraph's interrupt base: control flow by name."""


async def test_control_flow_propagates_out_of_the_handler():
    reg = Registry()

    @reg.tool
    async def interrupt() -> str:
        """Raise a control-flow signal."""
        raise GraphBubbleUp("pause")

    @reg.tool
    async def cancel() -> str:
        """Raise CancelledError."""
        raise asyncio.CancelledError

    handlers = Handlers(reg, Executor())
    ctx = SimpleNamespace(request=None, protocol_version="2026-07-28", session=None)
    with pytest.raises(GraphBubbleUp):
        await handlers.call_tool(ctx, t.CallToolRequestParams(name="interrupt"))  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await handlers.call_tool(ctx, t.CallToolRequestParams(name="cancel"))  # type: ignore[arg-type]


def test_input_schema_lookup_for_header_validation():
    reg = make_registry()
    handlers = Handlers(reg, None)
    assert handlers.input_schema("add") == mcp_tool(reg.get("add")).input_schema
    assert handlers.input_schema("only_openai") is None
    assert handlers.input_schema("missing") is None


def test_mcp_server_rejects_bad_arguments():
    with pytest.raises(ToolUsageError, match="needs a Registry"):
        mcp_server([])  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="client_approval=True"):
        mcp_server(Registry(), request_state_keys=["k" * 32])


def test_export_tool_honours_targets():
    reg = make_registry()
    assert INTEGRATION.export_tool(reg.get("add")).name == "add"
    with pytest.raises(ToolUsageError, match="not exported to 'mcp'"):
        INTEGRATION.export_tool(reg.get("only_openai"))
    assert [tool.name for tool in INTEGRATION.export_tools(reg)] == [
        "add",
        "point",
        "whoami",
        "refuse",
        "crash",
        "delete",
    ]


def test_missing_mcp_raises_missing_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[mcp\]"):
        mcp_server(Registry())
    with pytest.raises(MissingDependency):
        mcp_tool(make_registry().get("add"))
