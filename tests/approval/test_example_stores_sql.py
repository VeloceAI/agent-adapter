"""examples/approval_stores_sql.py runs offline: SQL on SQLite, DynamoDB on the fake client."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_adapter.approval import DynamoDBApprovalStore, SQLAlchemyApprovalStore
from tests.approval.fake_dynamodb import FakeDynamoDB

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "approval_stores_sql.py"
RESUMED = "worker: ok refunded 450.0 on o-7 | replay: pending"


def example() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE))


def test_the_sql_flow_runs_on_sqlite(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out
    assert "web: pending" in out and RESUMED in out


def test_the_postgres_and_alembic_snippets_build_a_store() -> None:
    pytest.importorskip("sqlalchemy")
    ns = example()
    store = ns["postgres_store"]()
    assert isinstance(store, SQLAlchemyApprovalStore) and store.table_name == "aa_approvals"
    assert ns["alembic_target_metadata"](store)[1] is store.metadata


async def test_the_dynamodb_snippet_runs_the_flow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeDynamoDB()
    regions: list[str] = []

    def client(service: str, *, region_name: str) -> FakeDynamoDB:
        assert service == "dynamodb"
        regions.append(region_name)
        return fake

    session = SimpleNamespace(client=client)
    monkeypatch.setitem(sys.modules, "aioboto3", SimpleNamespace(Session=lambda: session))
    ns = example()
    web, worker = ns["dynamodb_store"](), ns["dynamodb_store"]()
    assert isinstance(web, DynamoDBApprovalStore) and web.ttl_s == 7 * 24 * 3600
    assert await ns["pause_decide_resume"](web, worker) == "refunded 450.0 on o-7"
    assert RESUMED in capsys.readouterr().out and set(regions) == {"eu-west-1"}
    await web.aclose()
    await worker.aclose()
