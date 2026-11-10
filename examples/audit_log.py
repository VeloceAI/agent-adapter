"""Audit log: a signed record of a ticket desk's approvals and blocks, verified and exported.

A refund needs approval. One is approved, one is denied, and a guardrail blocks a lookup
that carries a card number. The audit log keeps those facts (never the arguments), signed
with a key only the operator holds. `verify` proves nothing was edited, removed or cut;
`export` gives an auditor a CSV. Changing one byte makes `verify` fail.
"""

import asyncio
import tempfile
from pathlib import Path

from agent_adapter import AdapterError, CallbackApproval, Executor, GuardrailHook, Registry
from agent_adapter.audit import audit, export, read_entries, verify

KEY = "demo-key-use-a-secret-manager-in-production"  # in production: AA_AUDIT_KEY
registry = Registry()


@registry.tool(confirm=True)
def refund(ticket: str, amount: int) -> str:
    """Refund a ticket (needs approval)."""
    return f"refunded {amount} for {ticket}"


@registry.tool
def lookup(ticket: str) -> str:
    """Find a ticket."""
    return f"ticket {ticket}: 2 seats"


def no_card_numbers(stage: str, arguments: dict[str, object]) -> bool | str:
    """Block any text argument that looks like a card number."""
    digits = [str(v).replace(" ", "") for v in arguments.values() if isinstance(v, str)]
    return "card number" if any(d.isdigit() and len(d) >= 12 for d in digits) else True


async def main(folder: Path) -> None:
    # --8<-- [start:attach]
    executor = Executor(
        registry,
        approval=CallbackApproval(lambda request: request.arguments["amount"] <= 50),
        hooks=[GuardrailHook(no_card_numbers)],
    )
    log = audit(executor, folder=folder, key=KEY)
    # --8<-- [end:attach]
    await executor.run("refund", {"ticket": "T-1", "amount": 20})
    for tool, arguments in (
        ("refund", {"ticket": "T-2", "amount": 90}),
        ("lookup", {"ticket": "4111 1111 1111 1111"}),
    ):
        try:
            await executor.run(tool, arguments)
        except AdapterError as exc:
            print(f"{tool}: {exc.kind}")
    log.stop()  # seals the last segment

    # --8<-- [start:verify]
    result = verify(folder, KEY)
    print(result.summary())
    print([entry["action"] for entry in read_entries(folder, KEY)])
    csv_text = export(folder, format="csv", key=KEY)
    print(csv_text.splitlines()[0])
    # --8<-- [end:verify]
    # The spaced form: bare "4111" can occur by chance in the hex macs and run ids.
    assert "4111 1111" not in csv_text  # arguments never reach the log

    # --8<-- [start:tamper]
    segment = next(folder.glob("audit-*.jsonl"))
    segment.write_bytes(segment.read_bytes().replace(b'"approved":false', b'"approved":true'))
    print(verify(folder, KEY).summary())
    # --8<-- [end:tamper]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(main(Path(tmp) / "audit"))
