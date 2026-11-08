"""Ship audit records elsewhere: a museum's ticket approvals sent to a collector, then checked.

`CallableAuditStore(fn)` signs and chains each record, then hands it to `fn`. Here `fn`
appends to a list standing in for a SIEM or a queue. The receiver checks the stream with
`verify_records`; a record lost on the way shows as a gap.
"""

import asyncio
import json

from agent_adapter import AutoApprove, Executor, Registry
from agent_adapter.audit import CallableAuditStore, audit, verify_records

KEY = "demo-key-use-a-secret-manager-in-production"  # in production: AA_AUDIT_KEY
registry = Registry()


@registry.tool(confirm=True)
def free_ticket(visitor: str) -> str:
    """Issue a free ticket (needs approval)."""
    return f"ticket for {visitor}"


async def main() -> None:
    # --8<-- [start:ship]
    collector: list[str] = []  # stands in for your SIEM, queue or bucket

    def ship(record: dict[str, object]) -> None:
        collector.append(json.dumps(record))  # e.g. POST it, with a timeout

    executor = Executor(registry, approval=AutoApprove())
    log = audit(executor, store=CallableAuditStore(ship, key=KEY))
    await executor.run("free_ticket", {"visitor": "ann@example.com"})
    log.stop()  # sends the seal

    received = [json.loads(line) for line in collector]
    print(verify_records(received, KEY).summary())
    print(verify_records(received[:2] + received[3:], KEY).summary())  # one lost on the way
    # --8<-- [end:ship]
    assert "ann@example.com" not in "".join(collector)


if __name__ == "__main__":
    asyncio.run(main())
