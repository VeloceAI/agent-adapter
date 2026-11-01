"""Every shipped ApprovalStore, built the same way for the shared conformance suite.

`redis-fake` mirrors the Lua scripts in Python and always runs. `redis-lua` runs the real
scripts under fakeredis + lupa, `redis-real` against `$AA_REDIS_URL`; both skip when absent.
`sqlalchemy` runs on SQLite through aiosqlite; `dynamodb-fake` always runs and
`dynamodb-moto` runs on moto; both of those skip when their packages are absent.
Every Redis client is wrapped in `Yielding`, so concurrent calls in one loop interleave
between commands as they do on a network connection.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.approval.dynamodb_store import DynamoDBApprovalStore
from agent_adapter.approval.redis_store import RedisApprovalStore
from agent_adapter.approval.sqlalchemy_store import SQLAlchemyApprovalStore
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.approval.store import ApprovalStore, MemoryApprovalStore
from tests.approval.fake_dynamodb import FakeDynamoDB, MotoClient
from tests.approval.fake_redis_store import FakeApprovalRedis

BACKENDS = (
    "memory",
    "sqlite",
    "redis-fake",
    "redis-lua",
    "redis-real",
    "sqlalchemy",
    "dynamodb-fake",
    "dynamodb-moto",
)
EVICTING = ("memory", "sqlite")  # the stores bounded by max_records


@dataclass
class Kit:
    """A store, and `twin()`: another store over the same records (another worker)."""

    name: str
    store: ApprovalStore
    twin: Callable[[], ApprovalStore]
    made: list[Any] = field(default_factory=list)
    finalizers: list[Callable[[], Any]] = field(default_factory=list)

    async def aclose(self) -> None:
        for store in [self.store, *self.made]:
            aclose = getattr(store, "aclose", None)
            if aclose is not None:
                await aclose()
        for finalize in self.finalizers:
            finalize()

    def other(self) -> ApprovalStore:
        store = self.twin()
        self.made.append(store)
        return store


def make(name: str, tmp_path: Path, **options: Any) -> Kit:
    """A fresh, empty store of backend `name` (skips when the backend is unavailable)."""
    if name == "memory":
        memory = MemoryApprovalStore(**options)
        return Kit(name, memory, lambda: memory)
    if name == "sqlite":
        path = tmp_path / "approvals.db"
        return Kit(name, SQLiteApprovalStore(path, **options), lambda: SQLiteApprovalStore(path))
    if name == "sqlalchemy":
        return _sqlalchemy_kit(tmp_path, **options)
    if name.startswith("dynamodb"):
        return _dynamodb_kit(name, **options)
    factory = _redis_factory(name)
    prefix = f"aa-test:{uuid.uuid4().hex}:"
    ttl = {"ttl_s": 600.0} if name == "redis-real" else {}
    options = {**ttl, **options}
    return Kit(
        name,
        RedisApprovalStore(factory, prefix=prefix, **options),
        lambda: RedisApprovalStore(factory, prefix=prefix, **options),
    )


def _sqlalchemy_kit(tmp_path: Path, **options: Any) -> Kit:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'approvals-sa.db').as_posix()}"
    return Kit(
        "sqlalchemy",
        SQLAlchemyApprovalStore(url, **options),
        lambda: SQLAlchemyApprovalStore(url, **options),
    )


def _dynamodb_kit(name: str, **options: Any) -> Kit:
    finalizers: list[Callable[[], Any]] = []
    if name == "dynamodb-fake":
        fake = FakeDynamoDB()
        factory: Callable[[], Any] = lambda: fake  # noqa: E731
    else:
        moto = pytest.importorskip("moto")
        boto3 = pytest.importorskip("boto3")
        mock = moto.mock_aws()
        mock.start()
        finalizers.append(mock.stop)
        table = "approvals"

        def factory() -> Any:
            return MotoClient(
                boto3.client(
                    "dynamodb",
                    region_name="us-east-1",
                    aws_access_key_id="testing",
                    aws_secret_access_key="testing",
                )
            )

        factory().client.create_table(**DynamoDBApprovalStore.table_definition(table))
    store_options = {"table": "approvals", **options}
    return Kit(
        name,
        DynamoDBApprovalStore(factory, **store_options),
        lambda: DynamoDBApprovalStore(factory, **store_options),
        finalizers=finalizers,
    )


class Yielding:
    """A Redis client that yields to the loop before each command (as a socket round trip does).

    Without it, fake clients answer each await at once, so racing coroutines never interleave
    and a missing compare-and-set would go unnoticed.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    async def eval(self, *args: Any) -> Any:
        await asyncio.sleep(0)
        return await self.client.eval(*args)

    async def hmget(self, *args: Any) -> Any:
        await asyncio.sleep(0)
        return await self.client.hmget(*args)

    async def aclose(self) -> None:
        await self.client.aclose()


def _redis_factory(name: str) -> Callable[[], Any]:
    factory = _raw_redis_factory(name)
    return lambda: Yielding(factory())


def _raw_redis_factory(name: str) -> Callable[[], Any]:
    if name == "redis-fake":
        fake = FakeApprovalRedis()
        return lambda: fake
    if name == "redis-lua":
        fakeredis = pytest.importorskip("fakeredis")
        pytest.importorskip("lupa")
        server = fakeredis.FakeServer()
        return lambda: fakeredis.FakeAsyncRedis(server=server)
    url = os.environ.get("AA_REDIS_URL", "")
    if not url:
        pytest.skip("set AA_REDIS_URL to run the conformance suite against a real Redis")
    redis_asyncio = pytest.importorskip("redis.asyncio")
    return lambda: redis_asyncio.Redis.from_url(url)
