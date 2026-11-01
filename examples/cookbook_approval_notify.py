"""Cookbook: ask a reviewer over Slack or email, then resume the call they decided."""

import asyncio
import hashlib
import hmac
import json

from agent_adapter import (
    Decision,
    DeferredApproval,
    Event,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolCall,
)
from agent_adapter.tracing import CallbackEventSink

registry = Registry()
SIGNING_SECRET = b"demo-secret"  # in real code: from your secret store
REVIEWERS = {"U_ANA": "ana@example.com"}  # chat user id -> reviewer allowed to decide


@registry.tool(confirm=True, effects="destructive")
def wire_transfer(account: str, amount: float) -> str:
    """Send money to an account."""
    return f"sent {amount} to {account}"


# --8<-- [start:notify]
outbox: list[str] = []  # approval ids waiting for a message; a queue in real code
sent: list[str] = []  # stands in for the Slack or email API


def on_event(event: Event) -> None:
    """Event sinks must be quick and never raise: only note the id here."""
    if event.type == "tool.pending":
        outbox.append(event.data["approval_id"])


store = MemoryApprovalStore()  # a database-backed store across processes
executor = Executor(registry, approval=DeferredApproval(store), sink=CallbackEventSink(on_event))


async def send_reviews() -> None:
    """A worker: one message per pending approval, with what the reviewer must see."""
    while outbox:
        record = await store.get(outbox.pop(0))
        text = f"Approve {record.request.tool} {json.dumps(record.request.arguments)}?"
        sent.append(
            f"#approvals {record.approval_id}: {text}"
        )  # post_message(...) / send_mail(...)


# --8<-- [end:notify]


# --8<-- [start:webhook]
def sign(body: bytes) -> str:
    return hmac.new(SIGNING_SECRET, body, hashlib.sha256).hexdigest()


async def on_button(body: bytes, signature: str) -> str:
    """The chat app's callback: verify it, check the reviewer, record the decision."""
    if not hmac.compare_digest(sign(body), signature):
        return "403 bad signature"
    action = json.loads(body)
    reviewer = REVIEWERS.get(action["user"])
    if reviewer is None:
        return "403 not a reviewer"
    approved = action["value"] == "approve"
    decision = Decision(approved=approved, decided_by=reviewer, reason=action.get("reason", ""))
    await store.decide(action["approval_id"], decision)  # the first decision wins
    return "200 recorded"


# --8<-- [end:webhook]


async def main() -> None:
    # --8<-- [start:flow]
    ctx = executor.context(run_id="run_payout_9")  # keep run id and call id to resume
    call = ToolCall(name="wire_transfer", arguments={"account": "AC-1", "amount": 900}, id="c1")
    first = await executor.run_call(call, ctx)
    await send_reviews()
    print(first.status, sent[0].split(": ", 1)[1])

    approval_id = first.error.details["approval_id"] if first.error else ""
    forged = json.dumps({"approval_id": approval_id, "user": "U_ANA", "value": "approve"})
    print(await on_button(forged.encode(), "0" * 64))  # unsigned: refused
    print(await on_button(forged.encode(), sign(forged.encode())))  # signed by the chat app

    resumed = await executor.run_call(call, ctx)  # same run, same call id
    print(resumed.status, resumed.value)
    # --8<-- [end:flow]


if __name__ == "__main__":
    asyncio.run(main())
