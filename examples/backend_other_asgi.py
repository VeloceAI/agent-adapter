"""Other frameworks: mount the ASGI tool server, or call ToolService from aiohttp and gRPC."""

import json
from typing import Any

from backend_common import Backend, build_backend, invoke, registry, tool_server_options

from agent_adapter.server import RequestInfo, ServiceResponse, ToolService, create_app


# --8<-- [start:mount]
def starlette_app(backend: Backend | None = None) -> Any:
    """Starlette (and the same idea in Litestar, Quart, Sanic): mount the tool server."""
    from starlette.applications import Starlette
    from starlette.routing import Mount

    backend = backend or build_backend()
    tools = create_app(registry, mcp=False, **tool_server_options(backend))
    return Starlette(routes=[Mount("/agent", app=tools)])


# --8<-- [end:mount]


# --8<-- [start:service]
def tool_service(backend: Backend) -> ToolService:
    """The framework-free HTTP tool server: auth, approvals, problems; no web imports."""
    return ToolService(registry, **tool_server_options(backend))


async def forward(
    service: ToolService,
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    client: str | None,
) -> tuple[int, dict[str, str], bytes]:
    """Run one request through `service`; JSON routes only (SSE needs a streaming reply)."""
    info = RequestInfo(method=method, path=path, headers=headers, query={}, client=client)
    response: ServiceResponse = await service.handle(info, body)
    if response.stream is not None:  # SSE: use aiohttp's StreamResponse, not this shim
        return 501, {"Content-Type": "application/json"}, b'{"title": "SSE not proxied"}'
    out = {"Content-Type": response.media_type, **response.headers}
    return response.status, out, json.dumps(response.body).encode()


def aiohttp_app(service: ToolService) -> Any:
    """aiohttp is not ASGI: a catch-all route that forwards to `service`."""
    from aiohttp import web  # type: ignore[import-not-found,unused-ignore]

    async def route(request: web.Request) -> web.Response:
        status, headers, body = await forward(
            service,
            request.method,
            request.path,
            dict(request.headers),
            await request.read(),
            request.remote,  # the peer: the server refuses remote anonymous callers
        )
        return web.Response(status=status, headers=headers, body=body)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", route)
    return app


# --8<-- [end:service]


# --8<-- [start:grpc]
class ToolServicer:
    """A `grpc.aio` servicer method: unpack the request message, await the service layer."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    async def Invoke(self, request: Any, context: Any) -> dict[str, Any]:
        status, body = await invoke(self.backend, request.tool, json.loads(request.arguments))
        return {"status": status, "body": json.dumps(body)}  # build your reply message here


# --8<-- [end:grpc]
