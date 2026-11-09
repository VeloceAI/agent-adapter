"""Call the tool server from Python: MCP with the official SDK, then plain REST."""

import asyncio
import os

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

URL = os.environ.get("AA_URL", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {os.environ.get('AA_TOKEN', 'dev-token')}"}


async def main() -> None:
    transport = streamable_http_client(
        f"{URL}/mcp", http_client=httpx2.AsyncClient(headers=HEADERS)
    )
    async with Client(transport) as client:
        names = [tool.name for tool in (await client.list_tools()).tools]
        assert "get_weather" in names, names
        result = await client.call_tool("get_weather", {"city": "Paris"})
        print("MCP:", "".join(p.text for p in result.content if isinstance(p, TextContent)))
    async with httpx2.AsyncClient(headers=HEADERS) as http:
        res = await http.post(
            f"{URL}/tools/get_weather/invoke", json={"arguments": {"city": "Paris"}}
        )
        print("REST:", res.json()["result"])


asyncio.run(main())
