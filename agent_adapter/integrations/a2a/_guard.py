"""Outbound HTTP guard for `connect_a2a`: no private hosts unless allowed, bounded responses."""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import logging
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

from agent_adapter.core.errors import IntegrationError

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
Resolver = Callable[[str, int], Awaitable[list[str]]]


class HostRefused(IntegrationError):
    """An outbound A2A request to a host the guard does not allow."""

    def __init__(self, message: str) -> None:
        super().__init__(message, integration="a2a")


class ResponseTooLarge(IntegrationError):
    """A remote A2A response over the byte limit."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"the A2A response is over {limit} bytes", integration="a2a")


async def resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


# Ranges `is_global` misses or that Python versions disagree on; never reachable from here.
_DENIED = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",
        "100.64.0.0/10",  # carrier-grade NAT
        "192.0.0.0/24",
        "198.18.0.0/15",
        "240.0.0.0/4",
        "fec0::/10",  # deprecated site-local
        "64:ff9b:1::/48",  # local-use NAT64 (RFC 8215)
        "2001::/32",  # Teredo
        "fc00::/7",
        "fe80::/10",
    )
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPATIBLE = ipaddress.ip_network("::/96")


def is_public(address: str) -> bool:
    """True for a globally routable unicast address, checking any IPv4 an IPv6 address embeds."""
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
        address.split("%", 1)[0]
    )
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and (ip in _NAT64 or ip in _V4_COMPATIBLE):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            ip = embedded
    if ip.is_multicast or any(ip in net for net in _DENIED if net.version == ip.version):
        return False
    return ip.is_global


class HostPolicy:
    """Which hosts outbound requests may reach."""

    def __init__(
        self,
        *,
        allow_private: bool = False,
        trusted_hosts: Iterable[str] = (),
        resolver: Resolver | None = None,
    ) -> None:
        self.allow_private = allow_private
        self.trusted = frozenset(host.lower() for host in trusted_hosts)
        self.resolver = resolver

    def check_url(self, url: str) -> tuple[str, int]:
        """The (host, port) of an http(s) URL, without any DNS lookup.

        Raises:
            HostRefused: not http(s), no host, or a bad port.
        """
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host:
            raise HostRefused(f"only http(s) URLs with a host are fetched, got {parts.scheme!r}")
        try:
            return host, parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise HostRefused("the URL has an invalid port") from exc

    async def vetted(self, host: str, port: int) -> list[str]:
        """The addresses to connect to: one lookup, every answer checked (`[host]` if trusted).

        Raises:
            HostRefused: the host does not resolve, or an answer is private or local.
        """
        host = host.lower()
        if self.allow_private or host in self.trusted:
            return [host]
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            try:
                addresses = await (self.resolver or resolve)(host, port)
            except OSError as exc:
                logger.warning("A2A host %s does not resolve: %s", host, type(exc).__name__)
                raise HostRefused(f"host {host!r} does not resolve") from exc
        if not addresses or not all(is_public(address) for address in addresses):
            logger.warning("A2A request to %s refused: private or local address", host)
            raise HostRefused(
                f"host {host!r} is private or local: pass allow_private=True or trusted_hosts="
            )
        return addresses

    async def check(self, url: str) -> None:
        """Refuse a non-http(s) URL, or one whose host is (or resolves to) a private address.

        Raises:
            HostRefused: the URL may not be fetched.
        """
        await self.vetted(*self.check_url(url))


def guarded_transport(
    inner: Any, policy: HostPolicy, *, max_response_bytes: int, pinned: bool = False
) -> Any:
    """An `httpx.AsyncBaseTransport` that checks every request's host and caps each body.

    `pinned=True` means `inner` checks the addresses it connects to (`pinned_transport`), so
    only the URL is checked here; otherwise the host is resolved and checked per request.
    """
    return _transport_class()(inner, policy, max_response_bytes, pinned)


@functools.cache
def _transport_class() -> type[Any]:
    import httpx

    class CappedStream(httpx.AsyncByteStream):
        """A response body that raises once it passes the limit."""

        def __init__(self, stream: Any, limit: int) -> None:
            self.stream = stream
            self.limit = limit

        async def __aiter__(self) -> AsyncIterator[bytes]:
            size = 0
            async for chunk in self.stream:
                size += len(chunk)
                if size > self.limit:
                    raise ResponseTooLarge(self.limit)
                yield chunk

        async def aclose(self) -> None:
            await self.stream.aclose()

    class GuardedTransport(httpx.AsyncBaseTransport):
        """Host policy before the request, byte cap on the response."""

        def __init__(self, inner: Any, policy: HostPolicy, limit: int, pinned: bool) -> None:
            self.inner = inner
            self.policy = policy
            self.limit = limit
            self.pinned = pinned

        async def handle_async_request(self, request: Any) -> Any:
            if self.pinned:
                self.policy.check_url(str(request.url))
            else:
                await self.policy.check(str(request.url))
            response = await self.inner.handle_async_request(request)
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > self.limit:
                await response.aclose()
                raise ResponseTooLarge(self.limit)
            response.stream = CappedStream(response.stream, self.limit)
            return response

        async def aclose(self) -> None:
            await self.inner.aclose()

    return GuardedTransport
