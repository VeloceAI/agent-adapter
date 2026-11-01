"""Every pending record a pause made is accounted for: kept, superseded, or withdrawn."""

import asyncio

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, FlowRunner, Map
from tests.flow.helpers import resume_until_done
from tests.flow.test_runner_fail_closed import Blob, Point

APPROVE = ApprovalDecision(approved=True)
KEY = b"s" * 32
registry = Registry()
RAN: list[str] = []


@registry.tool(confirm=True, parallel_safe=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


def setup():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    return FlowRunner(executor, checkpoint_key=KEY), executor, store


def two_pays(executor):
    async def go(state, ctx):
        paid = await asyncio.gather(
            executor.run("pay", {"amount": 1}, ctx), executor.run("pay", {"amount": 2}, ctx)
        )
        return {"note": " | ".join(paid)}

    return Flow("two_pays", state=Blob).node("go", go).end("go")


async def no_pending_left(store):
    assert await store.list(status="pending") == []
    assert all(record.consumed for record in await store.list())


async def test_two_concurrent_approvals_in_a_function_node_leave_no_record_pending():
    run, executor, store = setup()
    flow = two_pays(executor)
    paused = await run.run(flow, {})
    [superseded] = [r for r in await store.list() if r.approval_id != paused.pending.approval_id]
    assert superseded.status == "denied" and superseded.consumed
    assert superseded.decision is not None and superseded.decision.reason == "superseded"
    done, resumes = await resume_until_done(run, flow, paused)
    assert done.state.note == "paid 1 | paid 2" and resumes == 2
    assert sorted(RAN) == ["pay1", "pay2"]
    await no_pending_left(store)


async def test_two_pending_map_fn_items_leave_no_record_pending():
    run, executor, store = setup()

    async def per_item(item, state, ctx):
        return await executor.run("pay", {"amount": item}, ctx)

    flow = Flow("fn_items", state=Blob).node("go", Map("data", fn=per_item, into="out")).end("go")
    done, resumes = await resume_until_done(run, flow, await run.run(flow, {"data": [1, 2]}))
    assert done.state.out == ["paid 1", "paid 2"] and resumes == 2
    assert sorted(RAN) == ["pay1", "pay2"]
    await no_pending_left(store)


async def test_two_pending_map_tool_items_keep_their_records_for_the_resume():
    run, _, store = setup()
    flow = Flow("tool_items", state=Blob).node("go", Map("data", tool="pay", into="out")).end("go")
    paused = await run.run(flow, {"data": [{"amount": 1}, {"amount": 2}]})
    assert len(await store.list(status="pending")) == 2  # item 1's call id repeats on resume
    done, resumes = await resume_until_done(run, flow, paused)
    assert done.state.out == ["paid 1", "paid 2"] and resumes == 2
    assert len(await store.list()) == 2  # the kept record answered the second resume
    await no_pending_left(store)


async def test_an_unsealable_pause_withdraws_every_pending_record_of_the_node():
    run, executor, store = setup()
    with pytest.raises(FlowError, match="state-not-sealable"):
        await run.run(two_pays(executor), {"data": Point(1)})
    records = await store.list()
    assert len(records) == 2
    assert all(r.status == "denied" and r.consumed for r in records)
    assert RAN == []


async def test_an_unsealable_pause_withdraws_every_pending_map_item():
    run, executor, store = setup()

    async def per_item(item, state, ctx):
        return await executor.run("pay", {"amount": item}, ctx)

    flow = Flow("fn_bad", state=Blob).node("go", Map("data", fn=per_item, into="out")).end("go")
    with pytest.raises(FlowError, match="state-not-sealable"):
        await run.run(flow, {"data": [1, 2], "note": "x", "out": [Point(0)]})
    records = await store.list()
    assert len(records) == 2 and all(r.status == "denied" and r.consumed for r in records)
