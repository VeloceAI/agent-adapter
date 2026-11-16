"""LangGraph for comparison: the same workload as a `StateGraph` with a checkpointer.

Needs `langgraph` (the `langgraph` extra). Saves are synchronous (`durability="sync"`), the
strongest setting; the default (`"async"`) saves in the background.
"""

from __future__ import annotations

import contextlib
import copy
import operator
import threading
from collections.abc import Iterator, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent_adapter.errors import CheckpointNotFound
from tests.resume_contract.harness import ACTIVE, Crash, Outcome, World, effect, enter

THREAD = "trip-1"
CONFIG: Any = {"configurable": {"thread_id": THREAD}}
DURABILITY: Any = "sync"
RACE: threading.Barrier | None = None


class Trip(TypedDict):
    order: str
    log: Annotated[list[str], operator.add]


def pay(state: Trip) -> dict[str, Any]:
    enter("pay", dict(state))
    effect("charge")
    return {"log": [f"charged {state['order']}"]}


def ship(state: Trip) -> dict[str, Any]:
    enter("ship", dict(state))
    effect("ship")
    return {"log": [f"shipped {state['order']}"]}


def quote(state: Trip) -> dict[str, Any]:
    enter("quote", dict(state))
    ACTIVE.world.boundary("inside quote")
    return {"log": [f"quote {state['order']}"]}


def gather(state: Trip) -> dict[str, Any]:
    enter("gather", dict(state))
    return {}


def refund(state: Trip) -> dict[str, Any]:
    enter("refund", dict(state))
    amount = interrupt({"refund": 5})  # the resume value: an amount, or 0 to deny
    if RACE is not None:  # consume-once: hold both rival resumes here, if both got this far
        with contextlib.suppress(threading.BrokenBarrierError):
            RACE.wait(timeout=5)
    if not amount:
        return {"log": ["refund denied"]}
    effect("refund")
    return {"log": [f"refunded {amount} on {state['order']}"]}


def close(state: Trip) -> dict[str, Any]:
    enter("close", dict(state))
    effect("email")
    effect("audit")
    return {"log": [f"mailed {state['order']}", f"audited {state['order']}"]}


def build(state: Any = Trip, nodes: Sequence[tuple[str, Any]] = ()) -> StateGraph[Any]:
    graph = StateGraph(state)
    steps = nodes or (
        ("pay", pay), ("ship", ship), ("quote", quote), ("gather", gather),
        ("refund", refund), ("close", close),
    )  # fmt: skip
    for name, fn in steps:
        graph.add_node(name, fn)
    if nodes:
        names = [name for name, _ in nodes]
        for a, b in zip([START, *names], [*names, END], strict=True):
            graph.add_edge(a, b)
        return graph
    graph.add_edge(START, "pay")
    graph.add_edge("pay", "ship")
    graph.add_edge("pay", "quote")
    graph.add_edge(["ship", "quote"], "gather")
    graph.add_edge("gather", "refund")
    graph.add_edge("refund", "close")
    graph.add_edge("close", END)
    return graph


GRAPH = build()


class FaultSaver(InMemorySaver):
    """An in-memory checkpointer whose writes are boundaries; death makes it unreachable."""

    world: World = World()

    def snapshot(self) -> FaultSaver:
        other = FaultSaver()
        other.storage = copy.deepcopy(self.storage)
        other.writes = copy.deepcopy(self.writes)
        other.blobs = copy.deepcopy(self.blobs)
        return other

    def _reach(self) -> None:
        if self.world.dead:
            raise Crash("checkpointer unreachable after death")

    def get_tuple(self, config: Any) -> Any:
        self._reach()
        return super().get_tuple(config)

    def list(self, config: Any, **options: Any) -> Iterator[Any]:
        self._reach()
        return super().list(config, **options)

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        self._reach()
        self.world.boundary("before put")
        saved = super().put(config, checkpoint, metadata, new_versions)
        self.world.boundary("after put")
        return saved

    def put_writes(self, config: Any, writes: Any, task_id: str, task_path: str = "") -> None:
        self._reach()
        label = f"writes {sorted(channel for channel, _ in writes)}"
        self.world.boundary(f"before {label}")
        super().put_writes(config, writes, task_id, task_path)
        self.world.boundary(f"after {label}")


class LangGraphSubject:
    """LangGraph's `StateGraph` with a checkpointer, driven like our subjects."""

    name = "LangGraph"
    run_id = THREAD
    effects = ("charge", "ship", "refund", "email", "audit")

    def fresh(self) -> LangGraphSubject:
        return LangGraphSubject()

    def durable(self) -> FaultSaver:
        return FaultSaver()

    def view(self, inner: FaultSaver, world: World, *, yielding: bool = False) -> FaultSaver:
        inner.world = world
        return inner

    def dropped(self, inner: FaultSaver) -> list[FaultSaver]:
        """Copies that each lost one task's pending writes (not the newest) of the latest step."""
        latest = inner.storage[THREAD][""]
        if not latest:
            return []
        key = (THREAD, "", max(latest))
        tasks = list(dict.fromkeys(task for task, _ in inner.writes.get(key, {})))
        copies = []
        for task in tasks[:-1]:
            copy_ = inner.snapshot()
            copy_.writes[key] = {k: v for k, v in copy_.writes[key].items() if k[0] != task}
            copies.append(copy_)
        return copies

    def refused(self, exc: BaseException) -> bool:
        return True  # LangGraph has no corruption error: any loud failure counts

    def _out(self, result: Any) -> Outcome:
        if "__interrupt__" in result:
            return Outcome("paused")
        return Outcome("done", final={"order": result["order"], "log": result["log"]})

    async def start(self, saver: FaultSaver) -> Outcome:
        app = GRAPH.compile(checkpointer=saver)
        return self._out(app.invoke({"order": "o-1", "log": []}, CONFIG, durability=DURABILITY))

    async def resume(self, saver: FaultSaver, token: Any) -> Outcome:
        app = GRAPH.compile(checkpointer=saver)
        return self._out(app.invoke(Command(resume=5), CONFIG, durability=DURABILITY))

    async def recover(self, saver: FaultSaver, *, rerun: bool) -> Outcome:
        if saver.get_tuple(CONFIG) is None:
            raise CheckpointNotFound(THREAD)
        app = GRAPH.compile(checkpointer=saver)
        return self._out(app.invoke(None, CONFIG, durability=DURABILITY))

    async def ended(self, saver: FaultSaver) -> Outcome:
        raise AssertionError("LangGraph never refuses a recovery as ended")

    async def position(self, saver: FaultSaver) -> Any:
        """The latest checkpoint's values, the tasks with no recorded writes, and the rest."""
        found = saver.get_tuple(CONFIG)
        if found is None:
            raise CheckpointNotFound(THREAD)
        snapshot = GRAPH.compile(checkpointer=saver).get_state(CONFIG)
        values = {
            k: v for k, v in found.checkpoint["channel_values"].items() if k in Trip.__annotations__
        }
        todo = tuple(t.name for t in snapshot.tasks if t.result is None)
        done = tuple(t.name for t in snapshot.tasks if t.result is not None)
        return todo, done, values, dict(snapshot.values)

    def continued(self, position: Any, world: World) -> str | None:
        """Tasks with no writes rerun from the checkpoint state; finished ones never rerun."""
        todo, done, values, applied = position
        if not world.entered or "__start__" in todo + done:
            return None
        node, seen = world.entered[0]
        if (node in todo and seen == values) or (not todo and node not in done and seen == applied):
            return None
        return f"entered {node} with {seen}; durable: to run {todo} on {values}, done {done}"
