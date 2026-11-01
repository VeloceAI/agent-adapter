"""Import an OpenAPI document as tools; a mock transport stands in for the server."""

import asyncio
from typing import Any

import httpx

from agent_adapter import Executor, Registry

DOCUMENT: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"title": "Rates", "version": "1"},
    "servers": [{"url": "https://rates.example.com/v1"}],
    "paths": {
        "/rates/{item}": {
            "get": {
                "operationId": "getRate",
                "summary": "Shipping rate for an item.",
                "parameters": [
                    {"name": "item", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "X-Region", "in": "header", "schema": {"type": "string"}},
                ],
            }
        }
    },
}


def server(request: httpx.Request) -> httpx.Response:
    assert request.headers["Authorization"] == "Bearer s3cr3t"
    region = request.headers.get("X-Region", "US")
    return httpx.Response(200, json={"item": request.url.path.rsplit("/", 1)[-1], "region": region})


def client_factory(**options: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(server), **options)


async def main() -> None:
    registry = Registry()
    # --8<-- [start:import]
    from agent_adapter.integrations.openapi import BearerAuth, import_openapi

    specs = import_openapi(
        registry,
        DOCUMENT,  # a mapping, JSON/YAML text, or a file path
        auth=BearerAuth("s3cr3t"),  # or BearerAuth(get_token), refreshed per call
        retries=2,  # policy options apply to every imported tool; 429/5xx are retried
        client_factory=client_factory,  # omit in production: httpx.AsyncClient
    )
    rate = await Executor(registry).run("getRate", {"item": "laptop", "X_Region": "EU"})
    # --8<-- [end:import]
    print([s.name for s in specs], list(specs[0].json_schema()["properties"]))
    print(rate)


if __name__ == "__main__":
    asyncio.run(main())
