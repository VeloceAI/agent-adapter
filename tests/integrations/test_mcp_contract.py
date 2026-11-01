"""Contract kit C1-C8 for MCP: in-process and over stateless Streamable HTTP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import pytest

pytest.importorskip("mcp")

import httpx2
import mcp.types as t
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.mcp import (
    CALL_ID_META,
    ERROR_KIND_META,
    INTEGRATION,
    mcp_server,
)
from agent_adapter.integrations.mcp.annotations import unwrap_structured
from agent_adapter.schema.transforms import to_anthropic_input_schema
from tests.integrations._kit_support import wait_until
from tests.integrations.kit import Probe, contract_tests

pytestmark = pytest.mark.contract

BASE = "http://127.0.0.1:8000"


@dataclass
class Exported:
    tool: t.Tool  # what the integration exports
    server: Any  # the MCP server that runs it


class MCPHarness:
    """Calls go through a real `mcp.Client` against `mcp_server` (in-process dispatch)."""

    name = "mcp"
    edits_supported = True  # through the executor's port
    # MCP has no call id; a client key is honoured only to resume a pending approval.
    native_call_ids = False
    approval: Literal["native", "port"] = "port"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        registry = Registry()
        registry.add(spec)
        return Exported(INTEGRATION.export_tool(spec), mcp_server(registry, executor=executor))

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[t.Tool]:
        return INTEGRATION.export_tools(registry, executor=executor)

    @asynccontextmanager
    async def connect(self, native: Exported) -> AsyncIterator[Client]:
        async with Client(native.server) as client:
            yield client

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Any:
        async with self.connect(native) as client:
            return await client.call_tool(native.tool.name, args, meta={CALL_ID_META: call_id})

    def value(self, raw: Any) -> Any:
        if raw.is_error:
            kind = (raw.meta or {}).get(ERROR_KIND_META, "")
            return {"error": {"kind": kind, "message": raw.content[0].text}}
        return unwrap_structured(raw.structured_content, None)

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, t.CallToolResult) and raw.is_error

    async def approve_flow(
        self, native: Exported, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_kit_c6")

    def describe(self, native: Any) -> tuple[str, str, Mapping[str, Any]]:
        tool = native.tool if isinstance(native, Exported) else native
        return tool.name, tool.description or "", tool.input_schema

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_anthropic_input_schema(spec.json_schema())


class MCPHttpHarness(MCPHarness):
    """The same calls over stateless Streamable HTTP (`httpx2.ASGITransport`, no sockets)."""

    @asynccontextmanager
    async def connect(self, native: Exported) -> AsyncIterator[Client]:
        app = native.server.streamable_http_app(json_response=True, stateless_http=True)
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as http,
        ):

            @asynccontextmanager
            async def transport() -> AsyncIterator[Any]:
                async with streamable_http_client(f"{BASE}/mcp", http_client=http) as streams:
                    yield streams

            async with Client(transport()) as client:
                yield client


globals().update((item.__name__, item) for item in contract_tests(MCPHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(MCPHarness, runner_factory=Executor)
)
# Over a wire, a CancelledError the tool raises itself cannot reach the client as one (the
# SDK ends the request), so C7 runs in-process; `test_http_cancel_reaches_the_tool` covers
# the client side cancelling over HTTP.
globals().update(
    (f"{item.__name__}_http", item)
    for item in contract_tests(MCPHttpHarness, runner_factory=Executor)
    if not item.__name__.startswith("test_c7_")
)


async def test_http_cancel_reaches_the_tool():
    probe = Probe()
    harness = MCPHttpHarness()
    native = harness.export(probe.wait, Executor())
    task = asyncio.ensure_future(harness.call(native, {"label": "x"}, call_id="call_http_c7"))
    try:
        assert await wait_until(probe.started), "the tool never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await wait_until(probe.cancelled, 2.0), "the tool kept running"
    finally:
        probe.released.set()
