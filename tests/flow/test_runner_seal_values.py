"""Values the seal must keep exactly: huge ints, time zones, keys of defaulted arguments,
classes a loaded checkpoint may build, and the code identity of the graph's nodes."""

import datetime as dt
import functools
import json
import uuid
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, Field

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec._canon import type_name
from agent_adapter.flow import Call, Flow, FlowRunner, _seal
from agent_adapter.flow._digest import identity

APPROVE = ApprovalDecision(approved=True)
KEY = b"v" * 32
HUGE = 10**5000
registry = Registry()
RAN: list[str] = []
BUILT: list[str] = []


class Blob(BaseModel):
    data: Any = None
    note: str = ""


class Gadget(BaseModel):
    def model_post_init(self, context: Any) -> None:
        BUILT.append("gadget")


class Hit(BaseModel):
    text: str = ""


@registry.tool(confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@registry.tool
async def grow(n: int) -> int:
    """Add one to a number of any size."""
    RAN.append("grow")
    return n + 1


@registry.tool
async def charge(
    amount: int, request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
) -> str:
    """Charge once per request id."""
    RAN.append(f"charge{amount}")
    return request_id


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()
    BUILT.clear()


def setup():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    return FlowRunner(executor, checkpoint_key=KEY), executor, store


def pay_flow():
    return Flow("values", state=Blob).node("go", Call("pay", args={"amount": 1})).end("go")


async def test_a_huge_int_in_the_state_pauses_dumps_loads_and_resumes():
    run, _, _ = setup()
    paused = await run.run(pay_flow(), {"data": [HUGE, -HUGE]})
    loaded = run.load_checkpoint(pay_flow(), run.dump_checkpoint(paused.checkpoint))
    done = await run.resume(pay_flow(), loaded, APPROVE)
    assert done.state.data == [HUGE, -HUGE] and RAN == ["pay1"]


async def test_a_huge_int_tool_argument_inside_a_node_works_and_replays():
    run, executor, _ = setup()

    async def go(state, ctx):
        grown = await executor.run("grow", {"n": HUGE}, ctx)
        return {"data": grown, "note": await executor.run("pay", {"amount": 2}, ctx)}

    flow = Flow("huge_arg", state=Blob).node("go", go).end("go")
    paused = await run.run(flow, {})
    assert paused.status == "paused" and paused.checkpoint._journal["go"]["calls"]
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.data == HUGE + 1 and RAN == ["grow", "pay2"]


async def test_a_huge_int_tool_argument_outside_flows_works():
    assert await Executor(registry).run("grow", {"n": HUGE}) == HUGE + 1


async def test_a_call_with_a_defaulted_argument_replays_once():
    run, executor, _ = setup()

    async def go(state, ctx):
        request_id = await executor.run("charge", {"amount": 7}, ctx)
        return {"note": f"{request_id}: {await executor.run('pay', {'amount': 3}, ctx)}"}

    flow = Flow("defaults", state=Blob).node("go", go).end("go")
    paused = await run.run(flow, {})
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert RAN == ["charge7", "pay3"] and done.state.note.endswith(": paid 3")


async def test_a_zoned_datetime_keeps_its_zone_and_fold_through_json():
    when = dt.datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
    run, _, _ = setup()
    paused = await run.run(pay_flow(), {"data": when})
    loaded = run.load_checkpoint(pay_flow(), run.dump_checkpoint(paused.checkpoint))
    back = loaded.state.data
    assert back == when and back.fold == 1 and back.tzinfo is ZoneInfo("America/New_York")
    assert back.utcoffset() == dt.timedelta(hours=-5)


async def test_a_loaded_checkpoint_builds_no_class_its_state_model_does_not_reach():
    run, _, _ = setup()
    paused = await run.run(pay_flow(), {})
    document = json.loads(run.dump_checkpoint(paused.checkpoint))
    document["payload"]["state"]["fields"]["data"] = {"$model": type_name(Gadget), "fields": {}}
    document["mac"] = _seal.mac(KEY, document["payload"])  # validly sealed, unexpected class
    with pytest.raises(FlowError, match=r"cannot be loaded.*not allowed"):
        run.load_checkpoint(pay_flow(), json.dumps(document))
    assert BUILT == []


async def test_dump_refuses_a_class_that_loading_would_refuse():
    run, _, _ = setup()
    paused = await run.run(pay_flow(), {"data": Hit(text="x")})
    with pytest.raises(FlowError, match=r"Hit.*seal_decoder"):
        run.dump_checkpoint(paused.checkpoint)
    done = await run.resume(pay_flow(), paused.checkpoint, APPROVE)  # in memory it resumes
    assert done.state.data == Hit(text="x")


def first_pays(state):
    return {"note": "first"}


def second_pays(state):
    return {"note": "second"}


def gated(action, tool="pay"):
    return (
        Flow("identity", state=Blob)
        .node("go", Call(tool, args={"amount": 1}))
        .node("after", action)
        .edge("go", "after")
        .end("after")
    )


@pytest.mark.parametrize("changed", ["function", "tool"])
async def test_a_checkpoint_is_refused_when_a_nodes_code_identity_changed(changed):
    run, _, store = setup()
    paused = await run.run(gated(first_pays), {})
    other = gated(second_pays) if changed == "function" else gated(first_pays, tool="grow")
    with pytest.raises(FlowError, match="another version of this flow"):
        await run.resume(other, paused.checkpoint, APPROVE)
    assert (await store.get(paused.pending.approval_id)).status == "pending" and RAN == []
    done = await run.resume(gated(first_pays), paused.checkpoint, APPROVE)
    assert done.state.note == "first"


class Step:
    def __call__(self, state):
        return None


def test_code_identities_name_the_function_behind_partials_and_callables():
    assert identity(first_pays) == f"{__name__}:first_pays"
    assert identity(functools.partial(first_pays)) == f"partial({__name__}:first_pays)"
    assert identity(Step()) == f"{__name__}:Step"
