"""Edges of sub-flows and the node cache: cancellation, denial beside a pause, bad entries."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from agent_adapter.approval import AutoDeny, DeferredApproval, MemoryApprovalStore
from agent_adapter.cache import MemoryCache
from agent_adapter.core.errors import FlowDefinitionError, FlowError
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, NodeCache, SubFlow, _cache
from agent_adapter.flow._nested import SubFlowPaused
from agent_adapter.flow._result import FlowCheckpoint
from agent_adapter.flow._seal import mac
from agent_adapter.flow._subflow import run_subflow
from agent_adapter.flow.visit import NodeEnv
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import PAID, registry, runner
from tests.flow.subflow_helpers import Order, Pack, Shop, child_flow, parent_flow


async def test_cancelling_a_run_ends_the_child_flow_as_cancelled():
    started = asyncio.Event()

    async def wait(state):
        started.set()
        await asyncio.sleep(30)

    child = Flow("slow", state=Order).node("wait", wait).end("wait")
    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    task = asyncio.create_task(run.run(parent_flow(child), {}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    ends = {e.data["flow"]: e.data["status"] for e in sink.events if e.type == "flow.finished"}
    assert ends == {"slow": "cancelled", "parent": "cancelled"}


async def test_control_flow_from_an_input_function_passes_through():
    def stop(state):
        raise asyncio.CancelledError

    run, _, _ = runner()
    with pytest.raises(asyncio.CancelledError):
        await run.run(parent_flow(child_flow(), input=stop), {})


async def test_a_denied_child_gate_withdraws_what_a_sibling_left_pending():
    PAID.clear()  # a module-wide list other tests fill
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    run = FlowRunner(executor, approval=AutoDeny())  # gates are denied, tools wait
    flow = (
        Flow("both", state=Order)
        .node("start", lambda s: None)
        .node("gated", SubFlow(child_flow(gate=True), into={"note": "note"}))
        .node("payer", Call("pay", args={"amount": 7}, into="item"))
        .edge("start", ["gated", "payer"])
        .end("gated", "payer")
    )
    done = await run.run(flow, {})
    assert done.status == "denied" and done.denied_at == "gated/check"
    assert await store.list(status="pending") == [] and PAID == []


async def test_a_subflow_visit_without_a_child_graph_is_refused():
    env = NodeEnv("parent", Executor(registry), ())
    node = parent_flow(child_flow()).validate().nodes["start"]
    with pytest.raises(FlowError, match="not a runnable SubFlow node"):
        await run_subflow(env, node, Order(), Executor(registry).context(), None)


def test_a_paused_child_without_a_request_is_refused():
    empty = FlowCheckpoint("c", Order(), (), {}, {}, 0, None)
    with pytest.raises(FlowError, match="holds no request"):
        SubFlowPaused(empty, ())


def test_a_whole_state_field_of_exactly_the_child_model_is_accepted():
    class Exact(BaseModel):
        packed: Pack = Pack()

    child = Flow("labeler", state=Pack).node("label", lambda s: None).end("label")
    flow = Flow("host", state=Exact).node("sub", SubFlow(child, into="packed")).end("sub")
    assert flow.validate().nodes["sub"].child is not None


def test_a_cached_subflow_with_a_gate_two_levels_down_is_refused():
    middle = Flow("middle", state=Order).node("deep", SubFlow(child_flow(gate=True))).end("deep")
    flow = Flow("top", state=Order).node("sub", SubFlow(middle), cache=NodeCache()).end("sub")
    with pytest.raises(FlowDefinitionError, match="sub-flow with pause gates"):
        flow.validate()


async def test_a_signed_checkpoint_with_children_for_a_plain_node_cannot_load():
    run, _, _ = runner(approval=DeferredApproval(MemoryApprovalStore()))
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {})
    document = json.loads(run.dump_checkpoint(paused.checkpoint))
    payload = document["payload"]
    payload["children"]["start"] = payload["children"].pop("sub")
    document["mac"] = mac(run._sealer.key, payload)
    with pytest.raises(FlowError, match="is not a SubFlow node of this graph"):
        run.load_checkpoint(flow, json.dumps(document))


def quote_flow() -> Flow[Shop]:
    return (
        Flow("quote", state=Shop)
        .node("q", lambda s: {"label": "priced"}, cache=NodeCache(scope="global"))
        .end("q")
    )


async def test_inputs_without_a_canonical_form_are_not_cached():
    class Odd(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        thing: object = None
        label: str = ""

    ran = []
    flow = (
        Flow("odd", state=Odd)
        .node("q", lambda s: ran.append(1) or {"label": "x"}, cache=NodeCache(scope="global"))
        .end("q")
    )
    run, _, _ = runner(cache=MemoryCache())
    await run.run(flow, {"thing": object()})
    await run.run(flow, {"thing": object()})
    assert ran == [1, 1]


@pytest.mark.parametrize(
    "body",
    [["not", "a", "body"], {"items": {"a": 1}, "view": None}, {"items": [], "view": [1]}],
)
async def test_a_signed_entry_of_the_wrong_shape_is_a_miss(body):
    port = MemoryCache()
    run = FlowRunner(Executor(registry, cache=port), checkpoint_key=b"k" * 16)
    await run.run(quote_flow(), {})
    [key] = list(port._data)
    store = _cache.NodeCacheStore(port, key, b"k" * 16, None, None, {})
    await port.set(key, json.dumps({"v": 1, "body": body, "mac": store._mac(body)}).encode())
    assert await store.get("quote", "q") is None


class Stopping(MemoryCache):
    def __init__(self, on: str) -> None:
        super().__init__()
        self.on = on

    async def get(self, key):
        if self.on == "get":
            raise asyncio.CancelledError
        return await super().get(key)

    async def set(self, key, value, ttl=None):
        if self.on == "set":
            raise asyncio.CancelledError
        await super().set(key, value, ttl)


@pytest.mark.parametrize("on", ["get", "set"])
async def test_cancellation_from_the_cache_port_passes_through(on):
    run = FlowRunner(Executor(registry, cache=Stopping(on)))
    with pytest.raises(asyncio.CancelledError):
        await run.run(quote_flow(), {})
