"""The durable call log: what each settled call logs, replay seeds, signed documents."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from agent_adapter.core.errors import CheckpointStoreCorrupt, RecoveryNeedsDecision
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolResult
from agent_adapter.exec._canon import canonical_bytes
from agent_adapter.exec._durable import (
    CallLog,
    check_in_doubt,
    idempotent_in,
    merged,
    open_document,
    seal_document,
    seeds_from,
)
from agent_adapter.exec._durable_seal import verified_calls
from agent_adapter.exec._journal import JournalSlot, _make_journal
from agent_adapter.exec.checkpoints import CallRecord, Checkpoint, MemoryCheckpointStore

KEY = b"k" * 32


async def log_for() -> tuple[MemoryCheckpointStore, CallLog]:
    store = MemoryCheckpointStore()
    cp = Checkpoint(kind="flow", name="f", status="running", segment="s", data="{}")
    await store.save("r", 0, cp)
    return store, CallLog(store, "r", "s", KEY)


def ok(value: Any = 1, status: Any = "ok") -> ToolResult:
    return ToolResult(call_id="c", name="t", status=status, value=value)


async def test_each_settled_call_logs_how_it_ended() -> None:
    store, log = await log_for()
    keyed = JournalSlot("t", "d1", 0, None)
    await log.started(("v",), keyed)
    assert [c.state for c in await store.calls("r", "s")] == ["started"]
    await log.settled(("v",), keyed, ok(), {"value": 1})
    pending = JournalSlot("t", "d2", 0, None)
    await log.settled(("v",), pending, ok(status="pending"), None)
    lost = JournalSlot("t", "d3", 0, None)
    await log.settled(("v",), lost, ok(), {"unsealable": "ran", "why": "a socket"})
    unkeyed = JournalSlot("t", "", 0, None, why="arguments not sealable")
    await log.started(("v", ("item:1", 0)), unkeyed)
    await log.settled(("v", ("item:1", 0)), unkeyed, ok(), None)
    logged = await store.calls("r", "s")
    assert verified_calls(KEY, "r", "s", logged) == logged  # each one signed, none missing
    assert [c.seq for c in logged] == [0, 1, 2, 3]
    states = [(c.state, c.why) for c in logged]
    assert states == [
        ("done", ""),
        ("pending", ""),
        ("unrecordable", "a socket"),
        ("unrecordable", "arguments not sealable"),
    ]


async def test_a_failed_write_marks_the_log_broken() -> None:
    _, log = await log_for()
    log.move_to("other")  # the store's segment is still "s"
    with pytest.raises(Exception, match="stale"):
        await log.started(("v",), JournalSlot("t", "d", 0, None))
    assert log.broken


async def test_a_journal_logs_only_calls_it_does_not_replay() -> None:
    store, log = await log_for()
    journal = _make_journal(observer=log, path=("v",))
    assert journal.durable
    slot = journal.claim("t", {"a": 1})
    assert slot is not None
    await journal.started(slot)
    await journal.settle(slot, ok())
    seed = seeds_from(await store.calls("r", "s"))["v"]
    again = _make_journal(seed, observer=log, path=("v",))
    replay = again.claim("t", {"a": 1})
    assert replay is not None and replay.result is not None
    await again.started(replay)  # replayed: nothing logged
    assert len(await store.calls("r", "s")) == 1
    plain = _make_journal()
    assert not plain.durable
    plain_slot = plain.claim("t", {"a": 1})
    assert plain_slot is not None
    await plain.started(plain_slot)
    await plain.settle(plain_slot, ok())


def record(identity: list[Any], tool: str = "t", key: str | None = None) -> CallRecord:
    digest = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return CallRecord(
        key=key or digest, tool=tool, state="done", result={"r": 1}, identity=identity
    )


def test_seeds_rebuild_nested_journals_and_skip_unfinished_calls() -> None:
    calls = [
        record(["v", [], "t", "d", 0]),
        record(["v", [["item:0", 0]], "t", "d", 0]),
        record(["v", [["item:0", 0]], "t", "d", 1]),
        CallRecord(key="x", tool="t", state="started", identity=["v", [], "t", "e", 0]),
    ]
    seeds = seeds_from(calls)
    assert seeds["v"]["calls"] == [["t", "d", 0, {"r": 1}]]
    [[name, n, child]] = seeds["v"]["children"]
    assert (name, n, len(child["calls"])) == ("item:0", 0, 2)


@pytest.mark.parametrize(
    "bad",
    [
        record(["v", [], "other", "d", 0]),
        record([5, [], "t", "d", 0]),
        record(["v", [["x"]], "t", "d", 0]),
        record(["v", [], "t", "d", 0], key="0" * 64),
        CallRecord(key="k", tool="t", state="done", result={}, identity=["short"]),
    ],
)
def test_a_logged_call_with_a_bad_identity_is_corruption(bad: CallRecord) -> None:
    with pytest.raises(CheckpointStoreCorrupt):
        seeds_from([bad])


def test_merged_journals_keep_the_first_and_add_the_rest() -> None:
    first = {
        "calls": [["t", "d", 0, "first"]],
        "children": [["c", 0, {"calls": [["t", "a", 0, 1]]}]],
    }
    extra = {
        "calls": [["t", "d", 0, "extra"], ["t", "e", 0, "new"]],
        "children": [["c", 0, {"calls": [["t", "b", 0, 2]]}], ["k", 0, {"calls": []}]],
    }
    out = merged(first, extra)
    assert out is not None
    assert out["calls"] == [["t", "d", 0, "first"], ["t", "e", 0, "new"]]
    kids = {name: data for name, _n, data in out["children"]}
    assert [c[1] for c in kids["c"]["calls"]] == ["a", "b"]
    assert merged(None, extra) is extra and merged(first, None) is first


def test_in_doubt_calls_rerun_only_when_idempotent_or_allowed() -> None:
    calls = [
        CallRecord(key="a", tool="read", state="started"),
        CallRecord(key="b", tool="pay", state="unrecordable", why="socket"),
        CallRecord(key="c", tool="pay", state="done", result={}),
    ]
    check = idempotent_in(None, {"read": True})
    with pytest.raises(RecoveryNeedsDecision) as caught:
        check_in_doubt("r", calls, check, rerun=False)
    assert caught.value.calls == (("pay", "socket"),)
    assert len(check_in_doubt("r", calls, check, rerun=True)) == 2
    assert check_in_doubt("r", calls[:1], check, rerun=False) == calls[:1]


def test_idempotent_reads_the_registry() -> None:
    registry = Registry()

    @registry.tool(idempotent=True)
    def safe() -> int:
        """Safe."""
        return 1

    @registry.tool
    def unsafe() -> int:
        """Unsafe."""
        return 1

    check = idempotent_in(registry)
    assert (check("safe"), check("unsafe"), check("unknown")) == (True, False, False)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("{nope", "not JSON"),
        ('{"format": "other"}', "of this kind"),
        ('{"format": "f", "payload": [], "mac": "x"}', "missing"),
        ('{"format": "f", "payload": {}, "mac": "x"}', "MAC"),
    ],
)
def test_signed_documents_are_checked(text: str, why: str) -> None:
    with pytest.raises(CheckpointStoreCorrupt, match=why):
        open_document(KEY, "f", text, "r")


def test_a_signed_document_reads_back() -> None:
    assert open_document(KEY, "f", seal_document(KEY, "f", {"a": [1]}), "r") == {"a": [1]}
