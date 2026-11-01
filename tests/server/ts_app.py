"""The fixture server the TypeScript client's live tests run against (`ts` nox session).

Tokens (env vars the harness sets for `node --test`):

| Env var | Token | Principal |
|---|---|---|
| `AA_TOKEN` | `test-token` | `ts-client` |
| `AA_OTHER_TOKEN` | `other-token` | `ts-other` (another principal: 403 / 404 on ts-client's runs) |
| `AA_APPROVER_TOKEN` | `approver-token` | `ts-approver` (holds `approvals:decide`) |

Tools: `add_numbers(a, b)` (read-only, parallel-safe), `send_email(to, body)` (needs
approval), `fail_always()` (raises `ToolError`). A cost ledger is configured.
"""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.cost.ledger import Ledger
from agent_adapter.server import APPROVER_SCOPE, Principal, bearer_auth, create_app

TOKENS = {
    "test-token": Principal("ts-client"),
    "other-token": Principal("ts-other"),
    "approver-token": Principal("ts-approver", scopes=frozenset({APPROVER_SCOPE})),
}
ENV = {
    "AA_TOKEN": "test-token",
    "AA_OTHER_TOKEN": "other-token",
    "AA_APPROVER_TOKEN": "approver-token",
}
# The live suite's scenario tools (clients/typescript/test/live.test.ts); strict: none skipped.
LIVE_ENV = {
    "AA_LIVE_STRICT": "1",
    "AA_LIVE_TOOL": "add_numbers",
    "AA_LIVE_ARGS": '{"a": 2, "b": 3}',
    "AA_LIVE_STRICT_TOOL": "add_numbers",
    "AA_LIVE_CONFIRM_TOOL": "send_email",
    "AA_LIVE_CONFIRM_ARGS": '{"to": "a@example.com", "body": "hi"}',
}


def build_registry() -> Registry:
    registry = Registry()

    @registry.tool(kind="math", tags=["math"], parallel_safe=True, effects="read_only")
    async def add_numbers(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    @registry.tool(kind="action", tags=["email"], confirm=True)
    async def send_email(to: str, body: str) -> dict[str, Any]:
        """Send an email (requires approval)."""
        return {"sent": True, "to": to}

    @registry.tool
    async def fail_always() -> str:
        """Always fails with a model-visible error."""
        raise ToolError("this tool always fails")

    return registry


def build_app() -> Any:
    """The app: no MCP mount (the `ts` session installs only the `server` extra)."""
    return create_app(
        build_registry(), mcp=False, authenticate=bearer_auth(TOKENS), ledger=Ledger()
    )
