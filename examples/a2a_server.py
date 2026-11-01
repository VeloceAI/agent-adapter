"""Serve registry tools as an A2A agent: bearer auth, one skill per tool, approvals parked."""

from agent_adapter import DeferredApproval, Executor, MemoryApprovalStore, Registry
from agent_adapter.integrations.a2a import a2a_app, static_bearer

# --8<-- [start:tools]
tools = Registry()


@tools.tool(effects="read_only")
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@tools.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


# --8<-- [end:tools]

# --8<-- [start:app]
store = MemoryApprovalStore()  # reviewers decide here (or through the HTTP server's API)
app = a2a_app(
    tools,
    url="https://tools.internal.example/a2a/",  # the public JSON-RPC URL, written in the card
    executor=Executor(approval=DeferredApproval(store)),
    authenticate=static_bearer({"s3cret-token": "agent-team"}),  # token -> caller id
    host="0.0.0.0",
    allowed_hosts=["tools.internal.example"],  # Host header allow-list
)
# uvicorn examples.a2a_server:app --host 0.0.0.0   (behind TLS)
# --8<-- [end:app]

if __name__ == "__main__":
    print([skill.id for skill in app.state.a2a_card.skills])
