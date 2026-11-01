"""Checkpoints are sealed: changed or foreign checkpoints never resume, and each resumes once."""

import dataclasses
import pickle

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalPending,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, Map
from agent_adapter.flow._resume import SPENT
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.flow.helpers import PAID, State, registry

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")
KEY = b"k" * 32


@pytest.fixture(autouse=True)
def _reset():
    PAID.clear()


def map_flow(name="sealed"):
    return Flow(name, state=State).node("go", Map("items", tool="pay", into="log")).end("go")


def gate_flow():
    return Flow("sealed_gate", state=State).node("go", lambda s: {"n": 1}, pause=True).end("go")


async def paused_two_items(store):
    run = FlowRunner(Executor(registry, approval=DeferredApproval(store)), checkpoint_key=KEY)
    flow = map_flow()
    first = await run.run(flow, {"items": [{"amount": 1}, {"amount": 2}]})
    second = await run.resume(flow, first.checkpoint, APPROVE)  # item 0 runs; item 1 pends
    assert second.status == "paused" and second.checkpoint._answers
    return run, flow, second.checkpoint


@pytest.mark.parametrize(
    "change",
    [
        "answers",
        "items",
        "journal",
        "decisions",
        "writes",
        "state",
        "mac",
    ],
)
async def test_a_changed_checkpoint_is_refused_and_runs_nothing(change):
    store = MemoryApprovalStore()
    run, flow, checkpoint = await paused_two_items(store)
    PAID.clear()
    entry = checkpoint._answers["go"][0]
    forged_request = entry[0].model_copy(update={"arguments": {"amount": 2}})
    changes = {
        "answers": {
            "_answers": {"go": (*checkpoint._answers["go"], (forged_request, APPROVE, False))}
        },
        "items": {"_items": {"go": {0: ("ok", "paid 1"), 1: ("ok", "paid 2")}}},
        "journal": {"_journal": {"go": ()}, "_items": {}},
        "decisions": {"decisions": {"go": APPROVE}},
        "writes": {"_writes": {"other": ({"note": "x"},)}},
        "state": {"state": checkpoint.state.model_copy(update={"n": 7})},
        "mac": {"_mac": ""},
    }[change]
    forged = dataclasses.replace(checkpoint, **changes)
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(flow, forged, APPROVE)
    assert PAID == []


async def test_a_gate_grant_cannot_be_forged_into_the_checkpoint():
    run = FlowRunner(Executor(registry), approval=ScriptedApproval([ApprovalPending("apr_g")]))
    paused = await run.run(gate_flow(), {})
    forged = dataclasses.replace(paused.checkpoint, decisions={"go": APPROVE})
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(gate_flow(), forged, DENY)


async def test_a_checkpoint_from_another_key_is_refused_and_a_shared_key_resumes():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    flow = map_flow("keys")
    paused = await FlowRunner(executor, checkpoint_key=KEY).run(flow, {"items": [{"amount": 5}]})
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)  # per-process key
    other = FlowRunner(executor, checkpoint_key=KEY)
    assert (await other.resume(flow, paused.checkpoint, APPROVE)).status == "done"
    assert PAID == [5]


async def test_a_pickled_checkpoint_still_resumes():
    run = FlowRunner(Executor(registry, approval=DeferredApproval(MemoryApprovalStore())))
    flow = map_flow("pickled")
    paused = await run.run(flow, {"items": [{"amount": 6}]})
    copy = pickle.loads(pickle.dumps(paused.checkpoint))
    assert (await run.resume(flow, copy, APPROVE)).state.log == ["paid 6"]


async def test_answers_must_match_the_approval_store():
    store = MemoryApprovalStore()
    _, flow, checkpoint = await paused_two_items(store)
    PAID.clear()
    answered = checkpoint._answers["go"][0][0]
    other_store = MemoryApprovalStore()
    await other_store.create(answered)  # the same id, but pending: never consumed there
    elsewhere = FlowRunner(
        Executor(registry, approval=DeferredApproval(other_store)), checkpoint_key=KEY
    )
    with pytest.raises(FlowError, match="checkpoint-tampered: answer"):
        await elsewhere.resume(flow, checkpoint, APPROVE)
    assert PAID == []


async def test_an_answer_another_port_holds_is_not_refused():
    store = MemoryApprovalStore()
    _, flow, checkpoint = await paused_two_items(store)
    elsewhere = FlowRunner(
        Executor(registry, approval=DeferredApproval(MemoryApprovalStore())), checkpoint_key=KEY
    )
    done = await elsewhere.resume(flow, checkpoint, APPROVE)  # the seal binds the answers
    assert done.status == "done" and sorted(PAID) == [1, 2]


def test_the_checkpoint_key_is_checked():
    for bad in (b"short", "not bytes"):
        with pytest.raises(ToolUsageError, match="checkpoint_key"):
            FlowRunner(Executor(registry), checkpoint_key=bad)  # type: ignore[arg-type]


async def test_reprs_never_show_pending_arguments():
    run = FlowRunner(Executor(registry, approval=DeferredApproval(MemoryApprovalStore())))
    flow = Flow("reprs", state=State).node("go", Call("pay", args={"amount": 424242})).end("go")
    paused = await run.run(flow, {})
    assert paused.pending.arguments == {"amount": 424242}  # the explicit accessor
    assert "424242" not in repr(paused) and "424242" not in repr(paused.checkpoint)
    assert "arguments redacted" in repr(paused)


async def test_a_wrapped_deferred_port_still_resumes_once_via_its_store():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=RecordingApproval(DeferredApproval(store)))
    flow = map_flow("wrapped")
    paused = await FlowRunner(executor).run(flow, {"items": [{"amount": 7}]})
    assert (await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)).status == "done"
    SPENT.clear()  # only the store remembers now
    with pytest.raises(ApprovalAlreadyDecided):
        await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)
    assert PAID == [7]


async def test_a_port_without_a_store_still_resumes_once_across_runners():
    executor = Executor(registry, approval=ScriptedApproval([ApprovalPending("apr_s")]))
    flow = map_flow("no_store")
    paused = await FlowRunner(executor).run(flow, {"items": [{"amount": 8}]})
    assert (await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)).status == "done"
    with pytest.raises(ApprovalAlreadyDecided):
        await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)
    assert PAID == [8]


async def test_a_refused_settle_does_not_spend_the_checkpoint():
    store = MemoryApprovalStore()
    run = FlowRunner(Executor(registry), approval=DeferredApproval(store))
    paused = await run.run(gate_flow(), {})
    await store.decide(paused.pending.approval_id, DENY)  # decided out of band
    with pytest.raises(ApprovalAlreadyDecided):
        await run.resume(gate_flow(), paused.checkpoint, APPROVE)  # disagrees with the store
    denied = await run.resume(gate_flow(), paused.checkpoint, DENY)  # F2: still usable
    assert denied.status == "denied"


async def test_a_failing_store_does_not_spend_the_checkpoint():
    store = MemoryApprovalStore()

    class Flaky(DeferredApproval):
        fail = True

    port = Flaky(store)
    original_get = store.get

    async def get(approval_id):
        if Flaky.fail:
            Flaky.fail = False
            raise ConnectionError("store down")
        return await original_get(approval_id)

    run = FlowRunner(Executor(registry), approval=port)
    paused = await run.run(gate_flow(), {})
    store.get = get  # type: ignore[method-assign]
    with pytest.raises(ConnectionError):
        await run.resume(gate_flow(), paused.checkpoint, APPROVE)
    assert (await run.resume(gate_flow(), paused.checkpoint, APPROVE)).state.n == 1  # F2b
