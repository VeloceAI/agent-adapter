"""The container's ASGI app: `uvicorn app:app`; shared state in Redis when REDIS_URL is set."""

import os
from typing import Any

from agent_adapter import RedisApprovalStore, Registry
from agent_adapter.server import Principal, RedisIdempotencyStore, bearer_auth, create_app

registry = Registry()


@registry.tool(effects="read_only", idempotent=True)
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


def shared_state(redis_url: str | None) -> dict[str, Any]:
    """Approvals and Idempotency-Key replays shared by every pod (in memory without Redis)."""
    if not redis_url:
        return {}
    from redis.asyncio import Redis  # only the deployed image installs redis

    def client() -> Any:
        return Redis.from_url(redis_url)

    return {
        "approvals": RedisApprovalStore(client),
        "idempotency": RedisIdempotencyStore(client),
    }


def build_app(env: dict[str, str]) -> Any:
    """The app for one environment; the token comes from a Kubernetes Secret, never the image."""
    tokens = {env["TOOLS_TOKEN"]: Principal("service@example.com")}
    hosts = [host.strip() for host in env.get("ALLOWED_HOSTS", "").split(",") if host.strip()]
    return create_app(
        registry,
        authenticate=bearer_auth(tokens),
        allowed_hosts=hosts or None,
        mcp=False,
        **shared_state(env.get("REDIS_URL")),
    )


app = build_app(dict(os.environ)) if "TOOLS_TOKEN" in os.environ else None
