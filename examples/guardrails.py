"""Guardrails: block, redact or approve personal data and secrets in tool calls and replies."""

import asyncio

from agent_adapter import (
    ApprovalDecision,
    CallableModel,
    CallbackApproval,
    Executor,
    LLMRequest,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.hooks import CallableGuard, GuardVerdict, PIIGuard

registry = Registry()

# --8<-- [start:guards]
redact_out = PIIGuard(on=("result", "llm_response"), action="redact")
block_secrets = PIIGuard(kinds=["secret", "card"], on="args")
review_contacts = PIIGuard(kinds=["email", "phone"], on="args", action="approve")
# --8<-- [end:guards]


@registry.tool
def find_member(name: str) -> dict[str, str]:
    """Look up a museum member."""
    return {"name": name, "email": "ana@example.com", "phone": "+44 20 7946 0958"}


# --8<-- [start:approve]
@registry.tool(confirm=review_contacts.confirm)
def send_newsletter(to: str, subject: str) -> str:
    """Send the newsletter; a reviewer approves when it goes to a personal address."""
    return f"sent {subject!r}"


# --8<-- [end:approve]


@registry.tool
def save_note(text: str) -> str:
    """Save a note to the member file."""
    return "saved"


def reviewer(request):
    print("approval asked for", request.tool)
    return ApprovalDecision(approved=True, decided_by="desk lead")


executor = Executor(
    registry,
    hooks=[block_secrets, review_contacts, redact_out],
    approval=CallbackApproval(reviewer),
)


async def tools() -> None:
    print(await executor.run("find_member", {"name": "Ana"}))
    print(await executor.run("send_newsletter", {"to": "all-members", "subject": "June"}))
    print(await executor.run("send_newsletter", {"to": "ana@example.com", "subject": "June"}))
    blocked = await executor.run_call(
        ToolCall(name="save_note", arguments={"text": "card 4111 1111 1111 1111"})
    )
    print(blocked.error.kind, "|", blocked.error.message)


def model_reply(request: LLMRequest) -> str:
    return "Ana's address is ana@example.com."


async def model() -> None:
    result = await run_tool_loop(
        CallableModel(model_reply),
        Registry(),
        [Message(role="user", content="Ana's email?")],
        executor=executor,
    )
    print(result.final.text)


# --8<-- [start:callable]
async def moderation_service(stage: str, payload: object) -> bool | str | GuardVerdict:
    """Stands in for your moderation endpoint: say no to ticket resale."""
    return "resale is not allowed" if "resell" in str(payload) else True


moderated = Executor(registry, hooks=[CallableGuard(moderation_service, on="args")])
# --8<-- [end:callable]


async def external() -> None:
    result = await moderated.run_call(
        ToolCall(name="save_note", arguments={"text": "resell two tickets"})
    )
    print(result.error.kind, "|", result.error.message)


async def main() -> None:
    await tools()
    await model()
    await external()


if __name__ == "__main__":
    asyncio.run(main())
