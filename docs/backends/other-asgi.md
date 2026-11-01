# Other web frameworks

:material-check-circle:{ .status-ok } Available · `pip install "agent-adapter[server]"` for the mount

Any async framework can `await executor.run(...)` in its handlers, as the
[FastAPI](fastapi.md) guide does. To expose the whole registry over HTTP, mount the tool
server (ASGI), or forward requests to `ToolService` (anything else).

| File | Does |
|---|---|
| `examples/backend_common.py` | the shared service layer: `build_backend`, `invoke`, `tool_server_options` |
| `examples/backend_other_asgi.py` | a Starlette mount, an aiohttp shim over `ToolService`, a gRPC servicer |
| `tests/docs/test_backend_handler_examples.py` | runs the shim and the servicer offline; the mounts when the framework is installed |

Data enters as an HTTP request (or a gRPC message), goes through `ToolService` or `invoke`
to the executor, and leaves as JSON.

## Mount the tool server (ASGI)

```python
--8<-- "examples/backend_other_asgi.py:mount"
```

| Framework | Mount `tools = create_app(registry, ...)` with |
|---|---|
| Starlette | `Mount("/agent", app=tools)` (above) |
| Litestar | an `@asgi("/agent", is_mount=True)` handler that awaits `tools(scope, receive, send)` |
| Quart, Sanic (both ASGI apps) | an outer Starlette: `Mount("/agent", app=tools)`, then `Mount("/", app=your_app)` |
| Strawberry GraphQL | resolvers `await executor.run(...)`; mount `tools` beside the GraphQL app the same way |

The mounted server brings its own auth, approvals, idempotency and problems (see
[HTTP tool server](../cross-language/http-server.md)). With `mcp=True` (the default) its
lifespan must run: mount it in a framework that forwards lifespan events, or pass `mcp=False`.

## aiohttp and other non-ASGI servers

```python
--8<-- "examples/backend_other_asgi.py:service"
```

`ToolService.handle(RequestInfo, body)` is every route of the tool server with no web
imports. The shim passes the real peer address: an anonymous server refuses remote
callers by it. Event streams (SSE) need a streaming response, which this shim does not
proxy (it answers 501).

## gRPC

```python
--8<-- "examples/backend_other_asgi.py:grpc"
```

| Servicer | Call |
|---|---|
| `grpc.aio` (async) | `await invoke(...)` as above |
| sync `grpc.server` (thread pool) | `run_sync(invoke(...), timeout=...)`, or a long-lived loop (`LoopRunner`) |

gRPC has no adapter auth: authenticate in a server interceptor and refuse unknown callers
before the servicer runs. Map the reply's status to a gRPC status code.

## Rules

| Rule | Why |
|---|---|
| build one backend per process, at startup | executors and stores are shared by every request |
| `await` in async handlers, `run_sync` in sync ones | never `asyncio.run` inside a running loop |
| one auth path | the mounted server authenticates its own routes; your routes use yours |

See also: [FastAPI](fastapi.md), [Flask](flask.md), [Notebooks and apps](notebooks.md).
