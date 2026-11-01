"""A first flow: classify a ticket, then either look up the policy or escalate."""

from pydantic import BaseModel

from agent_adapter import Executor, Flow, FlowRunner, Registry, run_sync
from agent_adapter.flow import Call, Route

registry = Registry()


@registry.tool
def lookup_policy(category: str) -> str:
    """Return the policy text for a ticket category."""
    return f"Our {category} policy: full refund within 30 days."


# --8<-- [start:state]
class Ticket(BaseModel):
    text: str
    category: str = ""
    reply: str = ""


# --8<-- [end:state]


# --8<-- [start:nodes]
def classify(ticket: Ticket) -> dict[str, str]:
    return {"category": "refund" if "refund" in ticket.text.lower() else "other"}


def escalate(ticket: Ticket) -> dict[str, str]:
    return {"reply": "A person will contact you today."}


# --8<-- [end:nodes]


# --8<-- [start:flow]
flow = (
    Flow("support", state=Ticket)
    .node("classify", classify)
    .node("policy", Call("lookup_policy", args=lambda t: {"category": t.category}, into="reply"))
    .node("escalate", escalate)
    .route(
        "classify",
        Route(
            lambda t: "policy" if t.category == "refund" else "escalate",
            targets=["policy", "escalate"],
        ),
    )
    .end("policy", "escalate")
)
# --8<-- [end:flow]

if __name__ == "__main__":
    # --8<-- [start:run]
    runner = FlowRunner(Executor(registry))
    result = run_sync(runner.run(flow, {"text": "I want a refund"}))
    print(result.status, "|", result.state.reply)
    print(dict(result.visits))

    other = run_sync(runner.run(flow, {"text": "My parcel is late"}))
    print(other.state.category, "|", other.state.reply)
    # --8<-- [end:run]
