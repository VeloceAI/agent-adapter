"""A local WebSocket server for provider fakes: each test's handler speaks the provider's
documented messages, and `Seen` records what our client sent."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

server_mod = pytest.importorskip("websockets.asyncio.server")


@dataclass
class Seen:
    path: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    frames: list[str | bytes] = field(default_factory=list)
    connections: int = 0

    def json(self) -> list[dict[str, Any]]:
        return [json.loads(f) for f in self.frames if isinstance(f, str) and f.startswith("{")]

    def audio(self) -> bytes:
        return b"".join(f for f in self.frames if isinstance(f, bytes))


Handler = Callable[[Any, Seen], Awaitable[None]]


@contextlib.asynccontextmanager
async def serve(handler: Handler) -> AsyncIterator[tuple[str, Seen]]:
    """Run `handler(connection, seen)` per connection; yields (`ws://` base URL, seen)."""
    seen = Seen()

    async def run(connection: Any) -> None:
        seen.connections += 1
        seen.path = connection.request.path
        seen.headers = dict(connection.request.headers)
        with contextlib.suppress(Exception):
            await handler(connection, seen)

    async with server_mod.serve(run, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}", seen


async def frames(connection: Any, seen: Seen) -> AsyncIterator[str | bytes]:
    """Each frame the client sends, recorded in `seen`."""
    async for frame in connection:
        seen.frames.append(frame)
        yield frame


def pcm(samples: int, value: int = 1) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples
