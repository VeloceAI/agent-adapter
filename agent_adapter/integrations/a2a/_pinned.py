"""An httpx transport that connects only to addresses the host policy checked (no second lookup)."""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from agent_adapter.integrations.a2a._guard import HostPolicy

logger = logging.getLogger(__name__)


def pinned_transport(policy: HostPolicy, *, backend: Any = None) -> Any:
    """An `httpx.AsyncBaseTransport` over httpcore whose TCP connects go to vetted IPs only.

    TLS still uses the URL's host name for SNI and certificate checks. `backend` is the
    httpcore network backend that opens sockets (default: `httpcore.AnyIOBackend()`).
    """
    import httpcore
    import httpx

    pool = httpcore.AsyncConnectionPool(
        ssl_context=httpx.create_ssl_context(),
        network_backend=_backend_class()(policy, backend or httpcore.AnyIOBackend()),
    )
    return _transport_class()(pool)


@functools.cache
def _backend_class() -> type[Any]:
    import httpcore

    class PinnedBackend(httpcore.AsyncNetworkBackend):  # type: ignore[misc,unused-ignore]
        """Resolves and checks the host once, then connects to one of those addresses."""

        def __init__(self, policy: HostPolicy, inner: Any) -> None:
            self.policy = policy
            self.inner = inner

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[Any] | None = None,
        ) -> Any:
            addresses = await self.policy.vetted(host, port)
            for index, address in enumerate(addresses):
                try:
                    return await self.inner.connect_tcp(
                        address, port, timeout, local_address, socket_options
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                    if index == len(addresses) - 1:
                        raise
                    logger.info(
                        "A2A connect to %s failed (%s); next address", host, type(exc).__name__
                    )
            raise httpcore.ConnectError(f"no address for {host!r}")  # vetted never returns []

        async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> Any:
            raise httpcore.UnsupportedProtocol("unix sockets are not used for A2A")

        async def sleep(self, seconds: float) -> None:
            await self.inner.sleep(seconds)

    return PinnedBackend


@functools.cache
def _transport_class() -> type[Any]:
    import httpcore
    import httpx

    # Most specific first: the first match wins.
    errors: tuple[tuple[type[Exception], type[httpx.HTTPError]], ...] = (
        (httpcore.ConnectTimeout, httpx.ConnectTimeout),
        (httpcore.ReadTimeout, httpx.ReadTimeout),
        (httpcore.WriteTimeout, httpx.WriteTimeout),
        (httpcore.PoolTimeout, httpx.PoolTimeout),
        (httpcore.TimeoutException, httpx.TimeoutException),
        (httpcore.ConnectError, httpx.ConnectError),
        (httpcore.ReadError, httpx.ReadError),
        (httpcore.WriteError, httpx.WriteError),
        (httpcore.NetworkError, httpx.NetworkError),
        (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
        (httpcore.LocalProtocolError, httpx.LocalProtocolError),
        (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
        (httpcore.ProtocolError, httpx.ProtocolError),
    )

    def mapped(exc: Exception) -> Exception:
        for source, target in errors:
            if isinstance(exc, source):
                return target(str(exc))
        return exc

    class Body(httpx.AsyncByteStream):
        """An httpcore response body as an httpx stream, with httpx errors."""

        def __init__(self, stream: Any) -> None:
            self.stream = stream

        async def __aiter__(self) -> AsyncIterator[bytes]:
            try:
                async for chunk in self.stream:
                    yield chunk
            except Exception as exc:
                error = mapped(exc)
                if error is exc:
                    raise
                raise error from exc

        async def aclose(self) -> None:
            await self.stream.aclose()

    class PinnedTransport(httpx.AsyncBaseTransport):
        """httpx requests over an httpcore pool whose backend pins addresses."""

        def __init__(self, pool: Any) -> None:
            self.pool = pool

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            core = httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,  # type: ignore[arg-type,unused-ignore]
                extensions=request.extensions,
            )
            try:
                response = await self.pool.handle_async_request(core)
            except Exception as exc:
                error = mapped(exc)
                if error is exc:
                    raise
                raise error from exc
            return httpx.Response(
                status_code=response.status,
                headers=response.headers,
                stream=Body(response.stream),
                extensions=response.extensions,
            )

        async def aclose(self) -> None:
            await self.pool.aclose()

    return PinnedTransport
