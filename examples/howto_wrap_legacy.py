"""Expose a legacy sync service class as tools: read methods, a guarded write, clean errors."""

import asyncio
import time
from typing import Any

from agent_adapter import CallbackApproval, Executor, Registry, ToolCall, ToolError


# --8<-- [start:legacy]
class InvoiceService:
    """Existing code: sync, blocking, raises its own exceptions. None of it changes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {
            "INV-1": {"amount": 1200.0, "status": "open"},
            "INV-2": {"amount": 300.0, "status": "open"},
        }

    def find(self, invoice_id: str) -> dict[str, Any]:
        """Return one invoice.

        Args:
            invoice_id: Invoice number, e.g. INV-1.
        """
        time.sleep(0.01)  # a blocking database call
        return self.rows[invoice_id]  # KeyError when unknown

    def total_open(self) -> float:
        """Return the sum of all open invoices."""
        return float(sum(row["amount"] for row in self.rows.values() if row["status"] == "open"))

    def void(self, invoice_id: str, reason: str) -> str:
        """Void an invoice.

        Args:
            invoice_id: Invoice number.
            reason: Why it is voided.
        """
        self.rows[invoice_id]["status"] = "void"
        return f"{invoice_id} voided: {reason}"


# --8<-- [end:legacy]

registry = Registry()
service = InvoiceService()

# --8<-- [start:methods]
# Register bound methods. Reads may run side by side; the write needs approval.
for method in (service.find, service.total_open):
    registry.wrap(
        method,
        name=f"invoice_{method.__name__}",
        kind="db",
        effects="read_only",
        parallel_safe=True,
        timeout=5,  # a sync tool keeps running in its thread after a timeout
    )
registry.wrap(service.void, name="invoice_void", confirm=True, effects="destructive")
# --8<-- [end:methods]


# --8<-- [start:shim]
def find_or_explain(invoice_id: str) -> dict[str, Any]:
    """Return one invoice, or tell the model it does not exist.

    Args:
        invoice_id: Invoice number, e.g. INV-1.
    """
    try:
        return service.find(invoice_id)
    except KeyError:
        raise ToolError(f"no invoice {invoice_id}; ask the user for the number") from None


registry.wrap(find_or_explain, name="invoice_lookup", effects="read_only")
# --8<-- [end:shim]

# --8<-- [start:run]
reviewer = CallbackApproval(lambda request: request.arguments["reason"] != "")
executor = Executor(registry, approval=reviewer)


async def main() -> None:
    print(registry.names())
    print(await executor.run("invoice_find", {"invoice_id": "INV-1"}))
    raw = await executor.run_call(ToolCall(name="invoice_find", arguments={"invoice_id": "X"}))
    print(raw.status, raw.error.kind if raw.error else "", raw.error.message if raw.error else "")
    told = await executor.run_call(ToolCall(name="invoice_lookup", arguments={"invoice_id": "X"}))
    print(told.error.kind if told.error else "", told.error.message if told.error else "")
    print(await executor.run("invoice_void", {"invoice_id": "INV-1", "reason": "duplicate"}))
    print(await executor.run("invoice_total_open"))
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
