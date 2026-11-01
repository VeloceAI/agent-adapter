"""SQLiteApprovalStore across processes: racing consumers, and a pause resumed elsewhere."""

from __future__ import annotations

import runpy
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from tests.work import ROOT, run_child

EXAMPLE = ROOT / "examples" / "approval_stores.py"
RECORDS = 60  # decided records both children race to consume, in the same order

# Each child checks in through the store itself and waits for the other, so both race
# from the same moment; the result never depends on how long anything took.
CONSUMER = """import asyncio, json
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.core.types import ApprovalRequest

async def main(path, ids, me):
    store = SQLiteApprovalStore(path, timeout=60)
    ready = ApprovalRequest(approval_id="", run_id="ready", call_id=me, tool="barrier")
    await store.create(ready)
    while len(await store.list(run_id="ready")) < 2:
        await asyncio.sleep(0.001)
    return [approval_id for approval_id in ids if await store.consume(approval_id)]

print(json.dumps(asyncio.run(main({path!r}, {ids!r}, {me!r}))))
"""


def _children(scripts: list[str]) -> list[Any]:
    """Run every script in its own interpreter at once; their JSON results, in order."""
    results: list[Any] = [None] * len(scripts)
    errors: list[BaseException] = []

    def one(i: int) -> None:
        try:
            results[i] = run_child(scripts[i])
        except BaseException as exc:  # re-raised below, in the test's thread
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(scripts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


async def test_two_processes_racing_to_consume_win_each_decision_exactly_once(tmp_path: Path):
    path = str(tmp_path / "approvals.db")
    store = SQLiteApprovalStore(path)
    ids = []
    for i in range(RECORDS):
        record = await store.create(
            ApprovalRequest(approval_id="", run_id="run_1", call_id=f"c{i}", tool="pay")
        )
        await store.decide(record.approval_id, ApprovalDecision(approved=True))
        ids.append(record.approval_id)

    first, second = _children([CONSUMER.format(path=path, ids=ids, me=me) for me in "ab"])

    # Work counted, not timed: every decision is won once, by one process only.
    assert len(first) + len(second) == RECORDS
    assert set(first) | set(second) == set(ids) and not set(first) & set(second)
    assert all([(await store.get(approval_id)).consumed for approval_id in ids])
    assert [await store.consume(approval_id) for approval_id in ids] == [False] * RECORDS


RESUMER = """\
import asyncio, json, sys
sys.path.insert(0, {examples!r})
import approval_stores

print(json.dumps(asyncio.run(approval_stores.worker({path!r}))))
"""


async def test_a_call_paused_here_resumes_once_in_another_process(tmp_path: Path):
    example: dict[str, Any] = runpy.run_path(str(EXAMPLE))
    path = str(tmp_path / "approvals.db")
    approval_id = await example["web"](path)

    [value] = _children([RESUMER.format(examples=str(EXAMPLE.parent), path=path)])

    assert value == "refunded 450.0 on o-7"  # the reviewer's edit ran, in the other process
    _, store = example["executor_for"](path)
    spent = await store.get(approval_id)
    assert spent.consumed and spent.decision is not None
    assert spent.decision.edited_args == {"order_id": "o-7", "amount": 450}
    [replay] = await store.list(run_id=example["RUN_ID"], status="pending")
    assert replay.approval_id != approval_id  # the worker's replay opened a new review
    again = replay.request.model_copy(update={"approval_id": ""})  # the same call
    with pytest.raises(ApprovalPending) as info:
        await DeferredApproval(store).request(again)
    assert info.value.approval_id == replay.approval_id


def test_the_example_runs_both_processes(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "web: pending",
        "worker: ok refunded 450.0 on o-7 | replay: pending",
        "consumed: True",
    ]
