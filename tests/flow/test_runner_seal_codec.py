"""The seal's canonical form: non-JSON state, exact types, the graph digest and the JSON form."""

import contextlib
import dataclasses
import json
import pickle
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalPending,
    FlowError,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec._canon import encode
from agent_adapter.flow import Call, Flow, FlowRunner, Merge, _seal

APPROVE = ApprovalDecision(approved=True)
KEY = b"s" * 32
registry = Registry()
RAN: list[str] = []


class Blob(BaseModel):
    data: Any = None
    note: str = ""
    out: Annotated[list[Any], Merge.append] = []


class Point:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x


@registry.tool(confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@registry.tool
async def refuse(why: str) -> str:
    """Always refuses."""
    RAN.append(f"refuse {why}")
    raise ToolError(f"refused: {why}", tool="refuse")


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


def pay_flow(name="codec", extra_node=False):
    flow = Flow(name, state=Blob).node("go", Call("pay", args={"amount": 1}))
    if extra_node:
        flow = flow.node("after", lambda s: None).edge("go", "after").end("after")
        return flow
    return flow.end("go")


def setup(**options):
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    return FlowRunner(executor, checkpoint_key=KEY, **options), store


async def test_a_state_that_cannot_be_sealed_fails_the_pause_and_withdraws_its_approval():
    run, store = setup()
    with pytest.raises(FlowError, match=r"state-not-sealable.*state\.data.*object"):
        await run.run(pay_flow(), {"data": object()})
    assert await store.list(status="pending") == [] and RAN == []


async def test_a_node_that_puts_an_unsealable_value_in_the_state_fails_the_next_pause():
    run, store = setup()
    flow = (
        Flow("later", state=Blob)
        .node("first", lambda s: {"data": [object()]})
        .node("go", Call("pay", args={"amount": 1}))
        .edge("first", "go")
        .end("go")
    )
    with pytest.raises(FlowError, match=r"state-not-sealable.*state\.data\[0\]"):
        await run.run(flow, {})
    assert await store.list(status="pending") == [] and RAN == []


async def test_bytes_and_text_seal_differently():
    run, _ = setup()
    paused = await run.run(pay_flow(), {"data": b"abc"})
    as_text = dataclasses.replace(paused.checkpoint, state=Blob(data="abc"))
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(pay_flow(), as_text, APPROVE)
    tuple_state = await run.run(pay_flow(), {"data": (1, 2)})
    as_list = dataclasses.replace(tuple_state.checkpoint, state=Blob(data=[1, 2]))
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(pay_flow(), as_list, APPROVE)
    assert RAN == []


async def test_invalid_utf8_bytes_pause_and_resume():
    run, _ = setup()
    paused = await run.run(pay_flow(), {"data": b"\xff\xfe\x80"})
    done = await run.resume(pay_flow(), paused.checkpoint, APPROVE)
    assert done.status == "done" and done.state.data == b"\xff\xfe\x80" and RAN == ["pay1"]


async def test_a_seal_encoder_seals_custom_values_and_a_decoder_loads_them():
    def encoder(value):
        return {"x": value.x}

    def decoder(name, value):
        assert name.endswith(":Point")
        return Point(value["x"])

    run, _ = setup(seal_encoder=encoder, seal_decoder=decoder)
    paused = await run.run(pay_flow(), {"data": Point(3)})
    text = run.dump_checkpoint(paused.checkpoint)
    loaded = run.load_checkpoint(pay_flow(), text)
    assert loaded.state.data == Point(3)
    assert (await run.resume(pay_flow(), loaded, APPROVE)).state.data == Point(3)
    no_decoder, _ = setup(seal_encoder=encoder)
    with pytest.raises(FlowError, match=r"cannot be loaded.*seal_decoder"):
        no_decoder.load_checkpoint(pay_flow(), text)


def test_seal_hooks_must_be_callable():
    for option in ("seal_encoder", "seal_decoder"):
        with pytest.raises(ToolUsageError, match=option):
            FlowRunner(Executor(registry), **{option: "nope"})


async def test_a_mac_that_is_not_text_is_refused():
    run, _ = setup()
    paused = await run.run(pay_flow(), {})
    forged = dataclasses.replace(paused.checkpoint, _mac=paused.checkpoint._mac.encode())
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(pay_flow(), forged, APPROVE)
    assert RAN == []


async def test_a_checkpoint_of_another_graph_with_the_same_name_is_refused_unspent():
    run, store = setup()
    paused = await run.run(pay_flow(), {})
    with pytest.raises(FlowError, match="another version of this flow"):
        await run.resume(pay_flow(extra_node=True), paused.checkpoint, APPROVE)
    record = await store.get(paused.pending.approval_id)
    assert record.status == "pending" and RAN == []  # the approval was not spent
    assert (await run.resume(pay_flow(), paused.checkpoint, APPROVE)).status == "done"


async def test_a_checkpoint_round_trips_through_json_and_resumes_once():
    run, store = setup()
    paused = await run.run(pay_flow(), {"data": {"when": b"\x00", 1: (2, 3)}})
    text = run.dump_checkpoint(paused.checkpoint)
    json.loads(text)  # plain JSON text
    other = FlowRunner(Executor(registry, approval=DeferredApproval(store)), checkpoint_key=KEY)
    loaded = other.load_checkpoint(pay_flow(), text)
    assert loaded.state == paused.checkpoint.state and loaded.pending == paused.pending
    done = await other.resume(pay_flow(), loaded, APPROVE)
    assert done.state.data == {"when": b"\x00", 1: (2, 3)} and RAN == ["pay1"]
    with pytest.raises(ApprovalAlreadyDecided):
        await other.resume(pay_flow(), other.load_checkpoint(pay_flow(), text), APPROVE)


@pytest.mark.parametrize("change", ["state", "mac", "format", "json", "deep", "graph"])
async def test_a_tampered_json_checkpoint_is_refused_before_anything_is_built(change, monkeypatch):
    run, _ = setup()
    paused = await run.run(pay_flow(), {"note": "a"})
    document = json.loads(run.dump_checkpoint(paused.checkpoint))
    if change == "state":
        document["payload"]["state"]["fields"]["note"] = "b"
    elif change == "mac":
        document["mac"] = 7
    elif change == "format":
        document["format"] = "other"
    elif change == "graph":
        other = FlowRunner(Executor(registry), checkpoint_key=KEY)
        document = json.loads(other.dump_checkpoint(paused.checkpoint))
    text = {"json": "{", "deep": "[" * 100_000 + "]" * 100_000}.get(change) or json.dumps(document)

    def never(*args, **kwargs):
        raise AssertionError("built before the MAC was checked")

    monkeypatch.setattr("agent_adapter.flow._seal._build", never)
    flow = pay_flow(extra_node=change == "graph")
    with pytest.raises(FlowError) as raised:
        run.load_checkpoint(flow, text)
    if change in ("json", "deep"):
        assert _seal.TAMPERED in str(raised.value)


async def test_dump_refuses_a_checkpoint_this_runner_did_not_seal():
    run, _ = setup()
    paused = await run.run(pay_flow(), {})
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        FlowRunner(Executor(registry)).dump_checkpoint(paused.checkpoint)


async def test_the_journal_holds_plain_json_and_replays_a_rebuilt_exception():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    caught: list[BaseException] = []

    async def go(state, ctx):
        try:
            await executor.run("refuse", {"why": "x"}, ctx)
        except ToolError as exc:
            caught.append(exc)
        return {"note": await executor.run("pay", {"amount": 2}, ctx)}

    flow = Flow("journal_json", state=Blob).node("go", go).end("go")
    run = FlowRunner(executor, checkpoint_key=KEY)
    paused = await run.run(flow, {})
    journal = dict(pickle.loads(pickle.dumps(paused.checkpoint))._journal)
    assert json.loads(json.dumps(journal)) == journal  # plain JSON: no live exceptions
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "paid 2" and RAN == ["refuse x", "pay2"]
    first, replayed = caught
    assert type(replayed) is ToolError and replayed is not first
    assert str(replayed) == "refused: x" and replayed.tool == "refuse"


async def test_a_swapped_exception_in_the_journal_is_refused():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))

    async def go(state, ctx):
        with contextlib.suppress(ToolError):
            await executor.run("refuse", {"why": "y"}, ctx)
        return {"note": await executor.run("pay", {"amount": 4}, ctx)}

    flow = Flow("journal_swap", state=Blob).node("go", go).end("go")
    run = FlowRunner(executor, checkpoint_key=KEY)
    paused = await run.run(flow, {})
    journal = json.loads(json.dumps(dict(paused.checkpoint._journal)))
    journal["go"]["calls"][0][3]["error"]["exception"] = "builtins.SystemExit"
    forged = dataclasses.replace(paused.checkpoint, _journal=journal)
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run.resume(flow, forged, APPROVE)
    assert RAN == ["refuse y"]


class Other(BaseModel):
    data: Any = None


@pytest.mark.parametrize("part", ["state", "state-allowed", "pending", "decisions"])
async def test_a_validly_sealed_payload_of_the_wrong_types_is_not_loaded(part):
    run, _ = setup()
    paused = await run.run(pay_flow(), {})
    document = json.loads(run.dump_checkpoint(paused.checkpoint))
    wrong = {
        "state": Other(),
        "state-allowed": ApprovalDecision(approved=True),  # an allowed class, but not the state
        "pending": "text",
        "decisions": {"go": "yes"},
    }[part]
    part = part.removesuffix("-allowed")
    document["payload"][part] = encode(wrong)
    document["mac"] = _seal.mac(KEY, document["payload"])  # same key, other content
    with pytest.raises(FlowError, match="cannot be loaded"):
        run.load_checkpoint(pay_flow(), json.dumps(document))


async def test_a_port_that_puts_custom_objects_into_pending_arguments_fails_clearly():
    class Odd:
        async def request(self, request):
            raise ApprovalPending("apr_odd", tool="pay", arguments={"x": object()})

    run = FlowRunner(Executor(registry, approval=Odd()), checkpoint_key=KEY)
    with pytest.raises(FlowError, match=r"state-not-sealable.*pending"):
        await run.run(pay_flow(), {})
