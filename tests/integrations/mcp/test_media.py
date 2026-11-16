"""Media over MCP: our server sends media blocks next to structuredContent; import keeps them."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("mcp")

import mcp.types as t
from mcp import Client
from mcp.server.lowlevel import Server

from agent_adapter.core.media import Media
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.mcp import import_mcp, mcp_server

pytestmark = pytest.mark.contract

PNG = Media(data=b"\x89PNG\r\n\x1a\nPIXELS", name="shot.png")
WAV = Media(data=b"RIFF\0\0\0\0WAVEfmt AUDIO", name="clip.wav")
PDF = Media(data=b"%PDF-1.7 PAGES", name="report.pdf")
LINK = Media(url="https://example.com/cat.png")


def media_registry() -> Registry:
    reg = Registry()

    @reg.tool
    def screenshot() -> Media:
        """Take a screenshot."""
        return PNG

    @reg.tool
    def weather_with_media() -> list[dict[str, int] | Media]:
        """Weather data plus media."""
        return [{"temp": 21}, PNG, WAV, PDF, LINK]

    return reg


async def test_server_sends_media_blocks_with_and_without_structured_content():
    async with Client(mcp_server(media_registry())) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert tools["screenshot"].output_schema is None  # no base64 in a schema
        alone = await client.call_tool("screenshot", {})
        both = await client.call_tool("weather_with_media", {})
    assert alone.structured_content is None
    assert alone.content == [t.ImageContent(type="image", data=PNG.base64(), mime_type="image/png")]
    assert both.structured_content == {"result": {"temp": 21}}
    kinds = [block.type for block in both.content]
    assert kinds == ["text", "image", "audio", "resource", "resource_link"]
    assert base64.b64decode(both.content[3].resource.blob) == PDF.data
    assert str(both.content[4].uri) == LINK.url


async def test_import_keeps_media_returned_with_structured_content():
    local = Registry()
    async with Client(mcp_server(media_registry())) as client:
        await import_mcp(local, client)
        alone = await local.get("screenshot").invoke({})
        both = await local.get("weather_with_media").invoke({})
    assert alone == PNG.model_copy(update={"name": ""})
    assert both[0] == {"temp": 21}
    assert [item.mime_type for item in both[1:]] == [
        "image/png",
        "audio/wav",
        "application/pdf",
        "image/png",
    ]
    assert both[3].data == PDF.data and both[3].name == "report.pdf"
    assert both[4].url == LINK.url


def foreign_server() -> Server:
    async def list_tools(ctx, params) -> t.ListToolsResult:
        return t.ListToolsResult(tools=[t.Tool(name="doc", input_schema={"type": "object"})])

    async def call_tool(ctx, params) -> t.CallToolResult:
        text = t.TextResourceContents(uri="file:///a/readme.md", text="# hi")
        return t.CallToolResult(content=[t.EmbeddedResource(type="resource", resource=text)])

    return Server("foreign", on_list_tools=list_tools, on_call_tool=call_tool)


async def test_import_reads_an_embedded_text_resource_as_media():
    local = Registry()
    async with Client(foreign_server()) as client:
        await import_mcp(local, client)
        value = await local.get("doc").invoke({})
    assert value == Media(data=b"# hi", mime_type="text/plain", name="readme.md")
