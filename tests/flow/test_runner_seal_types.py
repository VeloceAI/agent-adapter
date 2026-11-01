"""Journal keys of nested defaults, polymorphic models in the seal, static args in the digest."""

import enum
import uuid

import pytest
from pydantic import BaseModel, Field, RootModel, field_serializer

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.exec import Executor
from agent_adapter.exec._canon import reachable_types, type_name
from agent_adapter.exec._journal import _make_journal
from agent_adapter.exec.pipeline import _claim
from agent_adapter.flow import Call, Flow, FlowRunner, Map
from agent_adapter.flow._digest import graph_digest

APPROVE = ApprovalDecision(approved=True)
KEY = b"t" * 32
registry = Registry()
RAN: list[str] = []


class Order(BaseModel):
    amount: int
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class Animal(BaseModel):
    name: str = ""


class Dog(Animal):
    good: bool = True


class Puppy(Dog):
    weeks: int = 8


class Cat(BaseModel):
    lives: int = 9


class Fussy(BaseModel):
    x: int = 0

    @field_serializer("x")
    def _refuse(self, x: int) -> int:
        raise ValueError("no dumps")


class Home(BaseModel):
    pet: Animal | None = None
    note: str = ""
    other: object = None
    items: list[int] = []


@registry.tool
async def ynested(order: Order) -> str:
    """Place an order (its id defaults per call)."""
    RAN.append("ynested")
    return order.request_id


@registry.tool
async def adopt() -> Animal:
    """Adopt an animal (a Dog, declared as Animal)."""
    RAN.append("adopt")
    return Dog(name="rex")


@registry.tool
async def fussy(arg: Fussy) -> int:
    """Take an argument that cannot be dumped."""
    return arg.x


@registry.tool(confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


def setup():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    return FlowRunner(executor, checkpoint_key=KEY), executor


def pay_flow(amount=1):
    return Flow("home", state=Home).node("go", Call("pay", args={"amount": amount})).end("go")


async def test_a_nested_default_factory_argument_replays_once():
    run, executor = setup()

    async def go(state, ctx):
        request_id = await executor.run("ynested", {"order": {"amount": 1}}, ctx)
        return {"note": f"{request_id}: {await executor.run('pay', {'amount': 1}, ctx)}"}

    flow = Flow("nested_default", state=Home).node("go", go).end("go")
    paused = await run.run(flow, {})
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert RAN == ["ynested", "pay1"] and done.state.note.endswith(": paid 1")


async def test_a_subclass_in_a_model_field_dumps_loads_and_resumes():
    run, _ = setup()
    paused = await run.run(pay_flow(), {"pet": Puppy(name="bo")})
    loaded = run.load_checkpoint(pay_flow(), run.dump_checkpoint(paused.checkpoint))
    done = await run.resume(pay_flow(), loaded, APPROVE)
    assert type(done.state.pet) is Puppy and done.state.pet.name == "bo" and RAN == ["pay1"]


async def test_a_tool_returning_a_subclass_of_its_annotation_replays():
    run, executor = setup()

    async def go(state, ctx):
        pet = await executor.run("adopt", {}, ctx)
        return {"pet": pet, "note": await executor.run("pay", {"amount": 2}, ctx)}

    flow = Flow("adopt", state=Home).node("go", go).end("go")
    paused = await run.run(flow, {})
    loaded = run.load_checkpoint(flow, run.dump_checkpoint(paused.checkpoint))
    done = await run.resume(flow, loaded, APPROVE)
    assert done.state.pet == Dog(name="rex") and RAN == ["adopt", "pay2"]


async def test_an_unrelated_class_is_still_refused():
    run, _ = setup()
    paused = await run.run(pay_flow(), {"other": Cat()})
    with pytest.raises(FlowError, match="Cat"):
        run.dump_checkpoint(paused.checkpoint)


def test_reachable_types_add_loaded_subclasses_but_never_every_model():
    found = reachable_types(Home)
    assert {type_name(c) for c in (Animal, Dog, Puppy)} <= set(found)
    assert type_name(Cat) not in found
    assert type_name(Cat) not in reachable_types(BaseModel)


async def test_changed_static_call_args_are_another_version_of_the_flow():
    run, _ = setup()
    assert graph_digest(pay_flow(1).validate()) != graph_digest(pay_flow(2).validate())
    paused = await run.run(pay_flow(1), {})
    with pytest.raises(FlowError, match="another version of this flow"):
        await run.resume(pay_flow(2), paused.checkpoint, APPROVE)
    assert RAN == []


def test_unsealable_static_args_still_digest():
    flow = Flow("odd", state=Home).node("go", Call("pay", args={"x": object()})).end("go")
    assert len(graph_digest(flow.validate())) == 64


def test_map_options_are_in_the_digest():
    def mapped(on_error):
        action = Map("items", tool="pay", into="note", on_error=on_error)
        return Flow("m", state=Home).node("go", action).end("go").validate()

    assert graph_digest(mapped("fail")) != graph_digest(mapped("skip"))


def test_a_journal_key_falls_back_to_the_arguments_when_dumping_fails():
    journal, call = _make_journal(), ToolCall(name="fussy", arguments={"arg": {"x": 1}})
    slot = _claim(journal, call, None, registry)
    again = _claim(_make_journal(), call, None, registry)
    assert slot is not None and again is not None
    assert slot.digest == again.digest != ""


class Tags(RootModel[list[str]]):
    pass


class Colour(enum.Enum):
    RED = "red"


def test_library_bases_never_expand_to_every_loaded_subclass():
    for base in (RootModel, enum.ReprEnum, enum.Enum, enum.StrEnum, BaseModel):
        found = reachable_types(base)
        assert type_name(Tags) not in found and type_name(Colour) not in found, base


def test_a_class_defined_in_pydantic_or_enum_is_never_expanded():
    library = type("Base", (BaseModel,), {"__module__": "pydantic.somewhere"})
    child = type("Child", (library,), {"__module__": __name__})
    assert type_name(child) not in reachable_types(library)
    flag = enum.Flag("Perm", {"R": 1, "W": 2}, module="enum")
    assert set(reachable_types(flag)) == {type_name(flag)}


def test_a_user_model_and_enum_still_reach_their_own_subclasses():
    class Base(BaseModel):
        pass

    class Leaf(Base):
        pass

    assert type_name(Leaf) in reachable_types(Base)
    assert type_name(Tags) in reachable_types(Tags)
