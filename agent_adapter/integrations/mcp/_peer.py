"""Loopback-only guard for anonymous apps built for a local host (ASGI peer address check)."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)

NON_LOCAL_CLIENT = "non-local-client"
# Starlette's TestClient sends this literal instead of an address; it is in-process, and a
# real server only puts an address here (a trusted proxy header could already claim 127.0.0.1).
TESTCLIENT_PEER = "testclient"

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def is_local_peer(host: str | None) -> bool:
    """True for a loopback address, no address (unix socket, in-process) or `testclient`."""
    if host is None or host == "" or host == TESTCLIENT_PEER:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False  # a name, not an address: never trusted as local
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def peer_of(scope: Scope) -> str | None:
    """The peer host of an ASGI scope (`scope["client"][0]`), None when absent."""
    client = scope.get("client")
    if not client:
        return None
    host = client[0]
    return host if isinstance(host, str) else str(host)


class LoopbackOnly:
    """ASGI middleware: a non-loopback HTTP peer gets `refuse`; a WebSocket one is closed."""

    def __init__(self, app: ASGIApp, *, refuse: ASGIApp, name: str) -> None:
        self.app = app
        self.refuse = refuse
        self.name = name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope.get("type")
        peer = peer_of(scope) if kind in ("http", "websocket") else None
        if is_local_peer(peer):
            await self.app(scope, receive, send)
            return
        logger.warning(
            "%s refused a request from non-local client %s: anonymous and local-only",
            self.name,
            peer,
        )
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # 403 at the handshake
        else:
            await self.refuse(scope, receive, send)
