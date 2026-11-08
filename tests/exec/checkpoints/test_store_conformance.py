"""Every shipped CheckpointStore keeps the same contract: compare-and-set steps, call logs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import (
    CheckpointConflict,
    ConfigError,
    ToolUsageError,
)
from agent_adapter.exec.checkpoints import (
    CallRecord,
    Checkpoint,
    CheckpointStore,
    MemoryCheckpointStore,
    RedisCheckpointStore,
    SQLAlchemyCheckpointStore,
    SQLiteCheckpointStore,
)

KINDS = ["memory", "sqlite", "redis-lua", "sqlalchemy"]


def make_store(kind: str, tmp_path: Path) -> Any:
    if kind == "memory":
        return MemoryCheckpointStore()
    if kind == "sqlite":
        return SQLiteCheckpointStore(tmp_path / "c.db")
    if kind == "redis-lua":
        fakeredis = pytest.importorskip("fakeredis")
        pytest.importorskip("lupa")
        server = fakeredis.FakeServer()
        return RedisCheckpointStore(lambda: fakeredis.FakeAsyncRedis(server=server))
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    return SQLAlchemyCheckpointStore(f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}")


@pytest.fixture(params=KINDS)
async def store(request: Any, tmp_path: Path) -> AsyncIterator[Any]:
    made = make_store(request.param, tmp_path)
    yield made
    close: Callable[[], Any] | None = getattr(made, "aclose", None)
    if close is not None:
        await close()


def cp(status: Any = "running", segment: str = "seg-a", name: str = "checkout") -> Checkpoint:
    return Checkpoint(kind="flow", name=name, status=status, segment=segment, data='{"x":1}')


def call(key: str, state: Any = "done", tool: str = "charge") -> CallRecord:
    result = {"value": "ok"} if state == "done" else None
    return CallRecord(key=key, tool=tool, state=state, result=result, identity=[key])


async def test_every_store_is_a_checkpoint_store(store: Any) -> None:
    assert isinstance(store, CheckpointStore)


async def test_the_first_save_is_step_0_and_latest_reads_it_back(store: Any) -> None:
    assert await store.latest("r1") is None
    await store.save("r1", 0, cp())
    found = await store.latest("r1")
    assert found is not None
    assert (found.run_id, found.step, found.checkpoint.status) == ("r1", 0, "running")
    assert found.checkpoint.saved_at > 0
    assert found.checkpoint.data == '{"x":1}'


async def test_a_save_must_be_exactly_one_past_the_stored_step(store: Any) -> None:
    with pytest.raises(CheckpointConflict):
        await store.save("r1", 1, cp())  # nothing stored: only step 0
    await store.save("r1", 0, cp())
    for bad in (0, 2, 5):
        with pytest.raises(CheckpointConflict, match="refused"):
            await store.save("r1", bad, cp())
    await store.save("r1", 1, cp(status="done"))
    assert (await store.latest("r1")).step == 1


async def test_two_workers_racing_for_the_same_step_one_wins(store: Any) -> None:
    await store.save("r1", 0, cp())
    outcomes = await asyncio.gather(
        *(store.save("r1", 1, cp(segment=f"seg-{i}")) for i in range(5)),
        return_exceptions=True,
    )
    wins = [o for o in outcomes if o is None]
    assert len(wins) == 1
    assert all(isinstance(o, CheckpointConflict) for o in outcomes if o is not None)


async def test_list_unfinished_holds_running_runs_only_oldest_first(store: Any) -> None:
    await store.save("a", 0, cp().model_copy(update={"saved_at": 10.0}))
    await store.save("b", 0, cp(name="other").model_copy(update={"saved_at": 5.0}))
    await store.save("c", 0, cp(status="paused"))
    await store.save("d", 0, cp(status="done"))
    assert [s.run_id for s in await store.list_unfinished()] == ["b", "a"]
    assert [s.run_id for s in await store.list_unfinished(name="checkout")] == ["a"]
    assert [s.run_id for s in await store.list_unfinished(limit=1)] == ["b"]
    await store.save("a", 1, cp(status="done"))
    assert [s.run_id for s in await store.list_unfinished()] == ["b"]
    with pytest.raises(ConfigError):
        await store.list_unfinished(limit=-1)


async def test_calls_are_kept_per_segment_and_the_latest_record_wins(store: Any) -> None:
    await store.save("r1", 0, cp(segment="seg-a"))
    await store.record_call("r1", "seg-a", call("k1", "started"))
    await store.record_call("r1", "seg-a", call("k2", "started"))
    await store.record_call("r1", "seg-a", call("k1", "done"))
    found = await store.calls("r1", "seg-a")
    assert [(c.key, c.state) for c in found] == [("k1", "done"), ("k2", "started")]
    assert found[0].result == {"value": "ok"}
    assert await store.calls("r1", "seg-b") == []


async def test_a_call_for_a_stale_segment_or_unknown_run_is_refused(store: Any) -> None:
    with pytest.raises(CheckpointConflict, match="stale"):
        await store.record_call("nope", "seg-a", call("k1"))
    await store.save("r1", 0, cp(segment="seg-a"))
    await store.save("r1", 1, cp(segment="seg-b"))
    with pytest.raises(CheckpointConflict, match="stale"):
        await store.record_call("r1", "seg-a", call("k1"))


async def test_moving_to_a_new_segment_drops_the_old_call_logs(store: Any) -> None:
    await store.save("r1", 0, cp(segment="seg-a"))
    await store.record_call("r1", "seg-a", call("k1"))
    await store.save("r1", 1, cp(segment="seg-a"))  # a claim keeps the segment and its calls
    assert len(await store.calls("r1", "seg-a")) == 1
    await store.save("r1", 2, cp(segment="seg-b"))
    assert await store.calls("r1", "seg-a") == []


async def test_delete_forgets_the_run_and_its_calls(store: Any) -> None:
    assert await store.delete("r1") is False
    await store.save("r1", 0, cp())
    await store.record_call("r1", "seg-a", call("k1"))
    assert await store.delete("r1") is True
    assert await store.latest("r1") is None
    assert await store.calls("r1", "seg-a") == []
    await store.save("r1", 0, cp())  # the id is free again


async def test_ids_that_look_alike_never_collide(store: Any) -> None:
    await store.save("a:b", 0, cp(segment="c"))
    await store.save("a", 0, cp(segment="b:c"))
    await store.record_call("a:b", "c", call("k1"))
    assert await store.calls("a", "b:c") == []
    await store.save("ünï\ud800", 0, cp())  # a lone surrogate survives
    assert (await store.latest("ünï\ud800")).run_id == "ünï\ud800"


@pytest.mark.parametrize("bad", ["", 5, None, "x" * 513])
async def test_bad_ids_are_refused_before_the_backend(store: Any, bad: Any) -> None:
    with pytest.raises(ToolUsageError):
        await store.save(bad, 0, cp())
    with pytest.raises(ToolUsageError):
        await store.latest(bad)
    with pytest.raises(ToolUsageError):
        await store.calls("r1", bad)


async def test_bad_steps_and_checkpoints_are_refused(store: Any) -> None:
    for step in (-1, True, 1.0, "0"):
        with pytest.raises(ToolUsageError):
            await store.save("r1", step, cp())
    with pytest.raises(ToolUsageError):
        await store.save("r1", 0, {"kind": "flow"})
