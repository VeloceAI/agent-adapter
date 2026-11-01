"""Serverless: one `tool_handler` for Lambda; `handle_envelope` for async function hosts."""

from collections.abc import Callable, Mapping
from typing import Any

from backend_common import Reply, build_backend, error_reply, record_decision

from agent_adapter import AdapterError, MemoryApprovalStore, run_sync
from agent_adapter.handlers import handle_envelope, tool_handler

# Stand-in: every container and the review process share one store (SQL, Redis, DynamoDB).
SHARED_STORE = MemoryApprovalStore()
TOOLS = frozenset({"shipping_rate", "send_invoice"})  # what this function may run

# --8<-- [start:lambda]
# Module level: built once per container, reused by every warm invocation.
BACKEND = build_backend(store=SHARED_STORE)

# The Lambda entry point: an SQS batch, an EventBridge event or a direct invoke.
handler = tool_handler(BACKEND.executor, tools=TOOLS, timeout=30)
# --8<-- [end:lambda]


# --8<-- [start:function]
async def tool_function(body: Mapping[str, Any]) -> Reply:
    """Body of an async HTTP function (Cloud Run functions, Azure Functions v2, Modal)."""
    try:
        return 200, await handle_envelope(BACKEND.executor, body, tools=TOOLS)
    except AdapterError as exc:  # an RFC 9457 problem: 404, 422, 403 ...
        return error_reply(exc)


# --8<-- [end:function]


# --8<-- [start:review]
def review(
    approval_id: str, *, reviewer: str, approved: bool, send: Callable[[dict[str, Any]], None]
) -> Reply:
    """In the review app: store the decision, then queue a resume message for the handler."""
    reply = run_sync(
        record_decision(BACKEND, approval_id, reviewer=reviewer, approved=approved),
        timeout=30,
    )
    if reply[0] == 200 and approved:
        send({"approval_id": approval_id})  # the handler runs the call once
    return reply


# --8<-- [end:review]
