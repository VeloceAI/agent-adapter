"""Database approval stores: park a call in SQL (or DynamoDB), decide it, resume it once."""

import asyncio
import tempfile
from pathlib import Path

from agent_adapter import Decision, DeferredApproval, Executor, Registry, ToolCall
from agent_adapter.approval import (
    ApprovalStore,
    DynamoDBApprovalStore,
    SQLAlchemyApprovalStore,
)

registry = Registry()


@registry.tool(confirm=True, effects="destructive")
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


RUN_ID = "run_support_42"  # the sender's ids: the resume must reuse them
CALL = ToolCall(name="refund", arguments={"order_id": "o-7", "amount": 500}, id="call_1")


# --8<-- [start:postgres]
def postgres_store(url: str = "postgresql+asyncpg://app@db/app") -> SQLAlchemyApprovalStore:
    """Needs the sqlalchemy extra; one engine per event loop; creates its table on first use."""
    return SQLAlchemyApprovalStore(url, table="aa_approvals")


# --8<-- [end:postgres]


# --8<-- [start:alembic]
def alembic_target_metadata(store: SQLAlchemyApprovalStore) -> list[object]:
    """In Alembic's env.py: add the store's table to your own metadata."""
    from sqlalchemy import MetaData

    app_metadata = MetaData()  # your models' metadata
    return [app_metadata, store.metadata]


# --8<-- [end:alembic]


# --8<-- [start:dynamodb]
def dynamodb_store(region: str = "eu-west-1") -> DynamoDBApprovalStore:
    """Needs the dynamodb extra; one client per event loop; items expire after a week."""
    import aioboto3

    session = aioboto3.Session()
    return DynamoDBApprovalStore(
        lambda: session.client("dynamodb", region_name=region),
        table="approvals",
        ttl_s=7 * 24 * 3600,
    )


# --8<-- [end:dynamodb]


async def pause_decide_resume(web_store: ApprovalStore, worker_store: ApprovalStore) -> str:
    """The web process pauses and a reviewer decides; a worker resumes exactly once."""
    # --8<-- [start:flow]
    web = Executor(registry, approval=DeferredApproval(web_store))
    first = await web.run_call(CALL, web.context(run_id=RUN_ID))
    print("web:", first.status)  # pending: nothing ran

    [waiting] = await web_store.list(run_id=RUN_ID, status="pending")
    edit = {"order_id": "o-7", "amount": 450}
    await web_store.decide(waiting.approval_id, Decision(approved=True, edited_args=edit))

    worker = Executor(registry, approval=DeferredApproval(worker_store))  # another host
    resumed = await worker.run_call(CALL, worker.context(run_id=RUN_ID))
    replay = await worker.run_call(CALL, worker.context(run_id=RUN_ID))
    print("worker:", resumed.status, resumed.value, "| replay:", replay.status)
    # --8<-- [end:flow]
    return str(resumed.value)


async def on_sqlite(folder: str) -> str:
    """The same SQL store code, on a local SQLite file (aiosqlite) instead of Postgres."""
    url = f"sqlite+aiosqlite:///{(Path(folder) / 'approvals.db').as_posix()}"
    web, worker = SQLAlchemyApprovalStore(url), SQLAlchemyApprovalStore(url)
    try:
        return await pause_decide_resume(web, worker)
    finally:
        await web.aclose()
        await worker.aclose()


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        print(asyncio.run(on_sqlite(folder)))


if __name__ == "__main__":
    main()
