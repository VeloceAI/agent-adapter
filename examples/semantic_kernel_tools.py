"""Semantic Kernel: registry tools as a plugin; an auto-invocation filter asks an approval port."""

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.filters import FilterTypes

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.semantic_kernel import approval_filter, kernel_plugin

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


def reviewer(request: ApprovalRequest) -> ApprovalDecision:
    """A stand-in for a human: approves invoices for one customer only."""
    approved = request.arguments["customer_id"] == "cus_1024"
    return ApprovalDecision(approved=approved, reason="" if approved else "unknown customer")


# --8<-- [start:wire]
kernel = Kernel()
kernel.add_plugin(kernel_plugin(registry, executor=Executor(registry)))  # agent_adapter-<tool>
kernel.add_filter(
    FilterTypes.AUTO_FUNCTION_INVOCATION, approval_filter(CallbackApproval(reviewer), registry)
)
# then: a chat service with FunctionChoiceBehavior.Auto() calls the tools through this kernel
# --8<-- [end:wire]


async def main() -> None:
    """Without a chat service: auto-invoke function calls as SK's function-calling loop does."""
    for name, arguments in [
        ("shipping_rate", '{"region": "eu"}'),
        ("send_invoice", '{"customer_id": "cus_1024", "amount": 49.5}'),
        ("send_invoice", '{"customer_id": "cus_2048", "amount": 49.5}'),
    ]:
        history = ChatHistory()
        call = FunctionCallContent(id="call_1", name=f"agent_adapter-{name}", arguments=arguments)
        await kernel.invoke_function_call(call, history)
        items = history.messages[0].items
        print(*[item.result for item in items if isinstance(item, FunctionResultContent)])


if __name__ == "__main__":
    asyncio.run(main())
