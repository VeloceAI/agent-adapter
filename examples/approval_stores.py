"""Durable approval stores: park a call in SQLite, decide it, resume it in another process."""

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_adapter import (
    Decision,
    DeferredApproval,
    Executor,
    RedisApprovalStore,
    Registry,
    SQLiteApprovalStore,
    ToolCall,
)

registry = Registry()


@registry.tool(confirm=True, effects="destructive")
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


RUN_ID = "run_support_42"  # the sender's ids: the resume must reuse them
CALL = ToolCall(name="refund", arguments={"order_id": "o-7", "amount": 500}, id="call_1")


# --8<-- [start:sqlite]
def executor_for(path: str) -> tuple[Executor, SQLiteApprovalStore]:
    """Every process on this host opens the same file; WAL lets them share it safely."""
    store = SQLiteApprovalStore(path, max_records=100_000, timeout=5.0)
    return Executor(registry, approval=DeferredApproval(store)), store


# --8<-- [end:sqlite]


async def web(path: str) -> str:
    """Process 1: the call pauses, a reviewer decides it."""
    # --8<-- [start:pause]
    executor, store = executor_for(path)
    first = await executor.run_call(CALL, executor.context(run_id=RUN_ID))
    print("web:", first.status)  # pending: nothing ran

    [waiting] = await store.list(run_id=RUN_ID, status="pending")
    edit = {"order_id": "o-7", "amount": 450}
    await store.decide(waiting.approval_id, Decision(approved=True, edited_args=edit))
    # --8<-- [end:pause]
    return waiting.approval_id


async def worker(path: str) -> str:
    """Process 2: the same run and call id resume with the decision, exactly once."""
    # --8<-- [start:resume]
    executor, _ = executor_for(path)  # a new process: only the file is shared
    resumed = await executor.run_call(CALL, executor.context(run_id=RUN_ID))
    replay = await executor.run_call(CALL, executor.context(run_id=RUN_ID))
    print("worker:", resumed.status, resumed.value, "| replay:", replay.status)
    # --8<-- [end:resume]
    return str(resumed.value)


# --8<-- [start:redis]
def shared_store(url: str = "redis://localhost:6379/0") -> RedisApprovalStore:
    """Needs the redis extra; one client per event loop; records expire after a week."""
    from redis.asyncio import Redis

    return RedisApprovalStore(
        lambda: Redis.from_url(url), prefix="{myapp}:approval:", ttl_s=7 * 24 * 3600
    )


# --8<-- [end:redis]


def main() -> None:
    if sys.argv[1:2] == ["worker"]:
        asyncio.run(worker(sys.argv[2]))
        return
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "approvals.db")
        approval_id = asyncio.run(web(path))
        child = subprocess.run(
            [sys.executable, __file__, "worker", path],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        print(child.stdout.strip())
        _, store = executor_for(path)
        print("consumed:", asyncio.run(store.get(approval_id)).consumed)


if __name__ == "__main__":
    main()
