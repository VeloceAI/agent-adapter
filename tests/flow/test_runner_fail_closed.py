"""A pause that cannot be sealed fails closed: its approvals are withdrawn, nothing re-runs."""

import logging
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalPending, FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, Map, Merge
from tests.fakes.sinks import RecordingSink

KEY = b"f" * 32
registry = Registry()
RAN: list[str] = []


class Blob(BaseModel):
    data: Any = None
    note: str = ""
    out: Annotated[list[Any], Merge.append] = []


class Point:
    def __init__(self, x: int) -> None:
        self.x = x


@registry.tool(confirm=True, parallel_safe=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@registry.tool(parallel_safe=True)
async def opaque() -> object:
    """Returns a value with no canonical form."""
    RAN.append("opaque")
    return Point(0)


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


def pay_flow(name="fc"):
    return Flow(name, state=Blob).node("go", Call("pay", args={"amount": 1})).end("go")


def runner(store=None, **options):
    store = store if store is not None else MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    return FlowRunner(executor, checkpoint_key=KEY, **options), store


async def assert_withdrawn(store, count=1):
    records = await store.list()
    assert len(records) == count
    assert all(r.status == "denied" and r.consumed for r in records)


async def test_a_flow_that_never_pauses_runs_with_an_unsealable_state():
    run, _ = runner()
    flow = Flow("free", state=Blob).node("go", lambda s: {"note": "ok"}).end("go")
    done = await run.run(flow, {"data": Point(1)})
    assert done.status == "done" and done.state.note == "ok"


async def test_a_gate_checks_its_payload_and_the_state_before_asking_its_port():
    run, store = runner(approval=DeferredApproval(gate_store := MemoryApprovalStore()))
    opaque_payload = Flow("gate", state=Blob).node(
        "go", lambda s: None, pause=lambda s: {"what": Point(2)}
    )
    with pytest.raises(FlowError, match=r"state-not-sealable.*pending\.arguments\.what"):
        await run.run(opaque_payload.end("go"), {})
    flow = Flow("gate", state=Blob).node("go", lambda s: None, pause=True).end("go")
    with pytest.raises(FlowError, match=r"state-not-sealable.*state\.data"):
        await run.run(flow, {"data": Point(1)})
    assert await gate_store.list() == [] and await store.list() == []


async def test_a_tool_pending_in_the_runner_store_is_withdrawn_and_a_retry_runs_nothing():
    run, store = runner()
    ctx = ToolContext(run_id="run_fc")
    with pytest.raises(FlowError, match=r"state-not-sealable.*state\.data"):
        await run.run(pay_flow(), {"data": Point(1)}, ctx)
    await assert_withdrawn(store)
    [old] = await store.list()
    with pytest.raises(ApprovalAlreadyDecided):  # the withdrawn record can no longer be approved
        await store.decide(old.approval_id, ApprovalDecision(approved=True))
    retry = await run.run(pay_flow(), {}, ToolContext(run_id="run_fc"))
    assert retry.status == "paused" and retry.pending.approval_id != old.approval_id
    assert RAN == []  # a retry asks afresh; the withdrawn approval runs nothing


async def test_a_tool_pending_in_the_nodes_own_store_is_withdrawn():
    own_store = MemoryApprovalStore()
    own = Executor(registry, approval=DeferredApproval(own_store))

    async def pays(state, ctx):
        return {"note": await own.run("pay", {"amount": 2}, ctx)}

    run, store = runner()
    flow = Flow("own", state=Blob).node("go", pays).end("go")
    with pytest.raises(FlowError, match="state-not-sealable"):
        await run.run(flow, {"data": Point(1)})
    assert await store.list() == []
    await assert_withdrawn(own_store)
    assert RAN == []


async def test_a_store_less_port_logs_an_error_and_emits_an_event(caplog):
    class Storeless:
        async def request(self, request):
            raise ApprovalPending("apr_nostore", tool=request.tool, arguments={"secret": "s3"})

    sink = RecordingSink()
    run = FlowRunner(Executor(registry, approval=Storeless()), checkpoint_key=KEY, sink=sink)
    with caplog.at_level(logging.ERROR), pytest.raises(FlowError, match="state-not-sealable"):
        await run.run(pay_flow(), {"data": Point(1)})
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("apr_nostore" in r.getMessage() for r in errors)
    assert "s3" not in caplog.text
    [event] = [
        e for e in sink.of_type("approval.decided") if e.data["approval_id"] == "apr_nostore"
    ]
    assert event.data["approved"] is False and event.data["withdrawn"] is False


async def test_an_unsealable_sibling_update_fails_closed_without_running_again():
    runs: list[str] = []

    def sibling(state):
        runs.append("sibling")
        return {"data": [Point(1)]}

    flow = (
        Flow("sib", state=Blob)
        .node("start", lambda s: None)
        .node("sibling", sibling)
        .node("go", Call("pay", args={"amount": 3}))
        .edge("start", "sibling")
        .edge("start", "go")
        .end("sibling", "go")
    )
    run, store = runner()
    with pytest.raises(FlowError, match=r"state-not-sealable.*writes\.sibling"):
        await run.run(flow, {})
    await assert_withdrawn(store)
    assert runs == ["sibling"] and RAN == []


async def test_an_unsealable_map_item_fails_closed():
    executor = Executor(registry, approval=DeferredApproval(store := MemoryApprovalStore()))

    async def per_item(item, state, ctx):
        RAN.append(f"item{item}")
        if item == 0:
            return Point(0)
        return await executor.run("pay", {"amount": item}, ctx)

    flow = Flow("items", state=Blob).node("go", Map("data", fn=per_item, into="out")).end("go")
    run = FlowRunner(executor, checkpoint_key=KEY)
    with pytest.raises(FlowError, match=r"state-not-sealable.*items\.go"):
        await run.run(flow, {"data": [0, 1]})
    await assert_withdrawn(store)
    assert sorted(RAN) == ["item0", "item1"]


async def test_an_unsealable_call_result_fails_closed_naming_the_call():
    executor = Executor(registry, approval=DeferredApproval(store := MemoryApprovalStore()))

    async def go(state, ctx):
        await executor.run("opaque", {}, ctx)
        return {"note": await executor.run("pay", {"amount": 4}, ctx)}

    flow = Flow("result", state=Blob).node("go", go).end("go")
    run = FlowRunner(executor, checkpoint_key=KEY)
    with pytest.raises(FlowError, match=r"state-not-sealable.*opaque.*result not sealable"):
        await run.run(flow, {})
    await assert_withdrawn(store)
    assert RAN == ["opaque"]
