"""Contract §7.3: every ApprovalStore's `find` returns only the unconsumed record of a call."""

import ast
from pathlib import Path

import pytest

from agent_adapter.approval.dynamodb_store import DynamoDBApprovalStore
from agent_adapter.approval.redis_store import RedisApprovalStore
from agent_adapter.approval.sqlalchemy_store import SQLAlchemyApprovalStore
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from tests.approval.stores import make

PACKAGE = Path(__file__).resolve().parents[2] / "agent_adapter"
STORE_METHODS = {"create", "get", "find", "decide", "consume", "list"}
# Every ApprovalStore implementation shipped in the package, and its conformance backend.
STORES = {
    MemoryApprovalStore.__name__: "memory",
    RedisApprovalStore.__name__: "redis-fake",
    SQLiteApprovalStore.__name__: "sqlite",
    SQLAlchemyApprovalStore.__name__: "sqlalchemy",
    DynamoDBApprovalStore.__name__: "dynamodb-fake",
}


def req(call_id: str = "c1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="r1", call_id=call_id, tool="pay", arguments={"amount": 1}
    )


def test_every_shipped_store_is_checked_here():
    found = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef) and "Protocol" not in map(ast.unparse, node.bases):
                methods = {n.name for n in node.body if isinstance(n, ast.AsyncFunctionDef)}
                if methods >= STORE_METHODS:
                    found.add(node.name)
    assert found == set(STORES)


@pytest.fixture(params=sorted(STORES))
async def store(request, tmp_path):
    kit = make(STORES[request.param], tmp_path)
    yield kit.store
    await kit.aclose()


async def test_find_returns_pending_and_decided_records_until_consumed(store):
    assert await store.find("r1", "c1") is None
    record = await store.create(req())
    assert (await store.find("r1", "c1")).approval_id == record.approval_id
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    decided = await store.find("r1", "c1")
    assert decided.approval_id == record.approval_id and decided.status == "approved"
    assert await store.consume(record.approval_id)
    assert await store.find("r1", "c1") is None


async def test_find_never_returns_a_consumed_record_beside_a_new_open_one(store):
    first = await store.create(req())
    await store.decide(first.approval_id, ApprovalDecision(approved=False, reason="no"))
    assert await store.consume(first.approval_id)
    second = await store.create(req())
    assert second.approval_id != first.approval_id
    found = await store.find("r1", "c1")
    assert found.approval_id == second.approval_id and not found.consumed
    await store.decide(second.approval_id, ApprovalDecision(approved=True))
    assert await store.consume(second.approval_id)
    assert await store.find("r1", "c1") is None
    assert await store.find("r1", "other") is None
