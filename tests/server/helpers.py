"""Shared fixtures for the server tests: a registry, tokens and a request helper."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.server.auth import APPROVER_SCOPE, Principal, RequestInfo, bearer_auth
from agent_adapter.server.service import ServiceResponse, ToolService, encode_body

ALICE = Principal("alice@example.com")
BOB = Principal("bob@example.com")
REVIEWER = Principal("reviewer", scopes=frozenset({APPROVER_SCOPE}))
TOKENS = {"t-alice": ALICE, "t-bob": BOB, "t-reviewer": REVIEWER}
SECRET = "hunter2-SECRET"


def make_registry(log: list[str] | None = None) -> Registry:
    seen = log if log is not None else []
    reg = Registry()

    @reg.tool(kind="math", tags=["math"], parallel_safe=True, effects="read_only")
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        seen.append(f"add {a} {b}")
        return a + b

    @reg.tool(kind="action", confirm=True)
    async def send_email(to: str, body: str) -> dict[str, Any]:
        """Send an email (needs approval)."""
        seen.append(f"send {to}")
        return {"sent": True, "to": to}

    @reg.tool(once_per_run=True)
    async def charge(amount: int) -> str:
        """Charge once per run."""
        seen.append(f"charge {amount}")
        return "charged"

    @reg.tool
    async def whoami(ctx: ToolContext) -> dict[str, Any]:
        """The ids and principal the tool sees."""
        principal = ctx.extras["principal"]
        return {
            "run_id": ctx.run_id,
            "call_id": ctx.call_id,
            "thread_id": ctx.thread_id,
            "principal": principal.id,
        }

    @reg.tool
    async def boom(token: str) -> str:
        """Fails with a secret in its message."""
        raise RuntimeError(f"password={token}")

    @reg.tool
    async def refuse_it() -> str:
        """Fails with a model-visible ToolError."""
        raise ToolError("nope, not today")

    @reg.tool(timeout=0.01)
    async def slow() -> str:
        """Never finishes in time."""
        await asyncio.sleep(1)
        return "late"

    @reg.tool
    async def nan() -> float:
        """Returns NaN (not JSON)."""
        return float("nan")

    @reg.tool(targets=["mcp"])
    async def mcp_only() -> str:
        """Not exported over HTTP."""
        return "x"

    return reg


def make_service(registry: Registry | None = None, **options: Any) -> ToolService:
    options.setdefault("authenticate", bearer_auth(TOKENS))
    return ToolService(registry if registry is not None else make_registry(), **options)


@dataclass
class Reply:
    status: int
    body: Any
    headers: dict[str, str]
    media_type: str
    raw: ServiceResponse


async def call(
    service: ToolService,
    method: str,
    path: str,
    *,
    token: str | None = "t-alice",
    body: Any = None,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
    json_body: bool = True,
) -> Reply:
    sent = dict(headers or {})
    if json_body and method == "POST" and not any(k.lower() == "content-type" for k in sent):
        sent["Content-Type"] = "application/json"
    if token is not None:
        sent["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
    info = RequestInfo(method, path, sent, dict(query or {}), "127.0.0.1")
    response = await service.handle(info, data)
    encoded = encode_body(response)
    decoded = json.loads(encoded) if encoded else None
    return Reply(response.status, decoded, dict(response.headers), response.media_type, response)


async def invoke(
    service: ToolService, tool: str, arguments: dict[str, Any], **kwargs: Any
) -> Reply:
    extra = kwargs.pop("extra", {})
    return await call(
        service, "POST", f"/tools/{tool}/invoke", body={"arguments": arguments, **extra}, **kwargs
    )


async def drain(response: ServiceResponse) -> list[dict[str, Any]]:
    """Read an SSE stream to its end; each frame as {id, event, data}."""
    assert response.stream is not None
    chunks = [chunk async for chunk in response.stream]
    frames = []
    for block in b"".join(chunks).decode().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        frames.append(
            {"id": fields.get("id"), "event": fields["event"], "data": json.loads(fields["data"])}
        )
    return frames
