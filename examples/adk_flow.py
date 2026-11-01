"""A flow compiled to an ADK Workflow: run it, pause at a gate, resume with a decision."""

import asyncio
from typing import Annotated, Any

from google.adk.sessions import InMemorySessionService
from pydantic import BaseModel, Field

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    Flow,
    MemoryApprovalStore,
    Registry,
)
from agent_adapter.flow import Call, Merge
from agent_adapter.integrations.adk import compile_flow, resume_message, run_workflow

registry = Registry()


@registry.tool
def draft_notice(customer_id: str) -> str:
    """Draft a reply to a customer notice."""
    return f"reply for {customer_id}"


class Notice(BaseModel):
    customer_id: str
    draft: str = ""
    sent: str = ""
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


def send(notice: Notice) -> dict[str, Any]:
    return {"sent": notice.draft, "log": ["sent"]}


flow = (
    Flow("notice", state=Notice)
    .node(
        "draft", Call("draft_notice", args=lambda n: {"customer_id": n.customer_id}, into="draft")
    )
    .node("send", send, pause=lambda n: {"draft": n.draft})  # a human checks the draft
    .edge("draft", "send")
    .end("send")
)


async def main() -> None:
    # --8<-- [start:run]
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    workflow = compile_flow(flow, executor=executor)  # START -> aa_driver
    sessions = InMemorySessionService()  # ADK keeps the run here while it is paused

    paused = await run_workflow(workflow, {"customer_id": "cus_1024"}, session_service=sessions)
    assert paused.pending is not None
    print(paused.status, paused.pending.tool, paused.pending.arguments)

    decision = ApprovalDecision(approved=True, edited_args={"draft": "edited reply"})
    message = resume_message(paused.pending.metadata["interrupt_id"], decision)
    done = await run_workflow(workflow, message, session_service=sessions)
    print(done.status, done.state.sent, dict(done.visits))
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
